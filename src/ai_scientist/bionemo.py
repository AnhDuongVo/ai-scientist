"""Client for NVIDIA BioNeMo biology NIMs (Boltz-2, RFdiffusion, ProteinMPNN, ESMFold/OpenFold).

All hosted NIMs share host health.api.nvidia.com with Bearer auth; a self-hosted NIM is the same path
without the /v1 prefix and no auth. Long jobs return HTTP 202 with an `nvcf-reqid` header; this client
polls the status endpoint. A `SimulatedBioNeMo` with the same interface backs the offline tests and demo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

HOSTED = "https://health.api.nvidia.com/v1"
PATHS = {
    "boltz2": "/biology/mit/boltz2/predict",
    "proteinmpnn": "/biology/ipd/proteinmpnn/predict",
    "rfdiffusion": "/biology/ipd/rfdiffusion/generate",
}


@dataclass
class BioNeMoSettings:
    api_key: str | None = field(default_factory=lambda: os.getenv("NGC_API_KEY") or os.getenv("NVIDIA_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("BIONEMO_BASE_URL", HOSTED))
    poll_seconds: int = 300
    timeout: int = 600


class BioNeMo(Protocol):
    def boltz2(self, polymers: list[dict], ligands: list[dict] | None = None, **kw: Any) -> dict: ...
    def rfdiffusion(self, input_pdb: str, contigs: str, hotspot_res: list[str] | None = None, **kw: Any) -> dict: ...
    def proteinmpnn(self, input_pdb: str, **kw: Any) -> dict: ...


class NIMBioNeMo:
    """Live BioNeMo client. Needs `requests` and a key (hosted) or a running NIM (BIONEMO_BASE_URL)."""

    def __init__(self, settings: BioNeMoSettings | None = None):
        import requests

        self.s = settings or BioNeMoSettings()
        self._requests = requests
        self.hosted = self.s.base_url.startswith("https://health.api.nvidia.com")
        if self.hosted and not self.s.api_key:
            raise RuntimeError(
                "Set NGC_API_KEY (or NVIDIA_API_KEY) for hosted BioNeMo, or BIONEMO_BASE_URL for a local NIM."
            )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.hosted and self.s.api_key:
            h["Authorization"] = f"Bearer {self.s.api_key}"
            h["NVCF-POLL-SECONDS"] = str(self.s.poll_seconds)
        return h

    def _post(self, tool: str, body: dict) -> dict:
        url = self.s.base_url.rstrip("/") + PATHS[tool]
        resp = self._requests.post(url, headers=self._headers(), json=body, timeout=self.s.timeout)
        if resp.status_code == 202:  # async: poll the status endpoint on the same host as base_url
            rid = resp.headers.get("nvcf-reqid")
            host = self.s.base_url.rstrip("/").removesuffix("/v1")
            status_url = f"{host}/v1/status/{rid}"
            deadline = time.time() + self.s.timeout
            while time.time() < deadline:
                s = self._requests.get(status_url, headers=self._headers(), timeout=60)
                if s.status_code == 202:
                    time.sleep(5)
                    continue
                s.raise_for_status()
                return s.json()
            raise TimeoutError(f"BioNeMo {tool} did not finish within {self.s.timeout}s")
        resp.raise_for_status()
        return resp.json()

    def boltz2(self, polymers, ligands=None, **kw):
        body = {
            "polymers": polymers,
            "ligands": ligands or [],
            "output_format": "mmcif",
            "recycling_steps": kw.get("recycling_steps", 3),
            "sampling_steps": kw.get("sampling_steps", 50),
            "diffusion_samples": kw.get("diffusion_samples", 1),
        }
        if any(lig.get("predict_affinity") for lig in (ligands or [])):
            body["diffusion_samples_affinity"] = kw.get("diffusion_samples_affinity", 1)
        return self._post("boltz2", body)

    def rfdiffusion(self, input_pdb, contigs, hotspot_res=None, **kw):
        body = {"input_pdb": input_pdb, "contigs": contigs, "diffusion_steps": kw.get("diffusion_steps", 50)}
        if hotspot_res:
            body["hotspot_res"] = hotspot_res
        return self._post("rfdiffusion", body)

    def proteinmpnn(self, input_pdb, **kw):
        body = {
            "input_pdb": input_pdb,
            "num_seq_per_target": kw.get("num_seq_per_target", 8),
            "sampling_temp": kw.get("sampling_temp", [0.1]),
        }
        if kw.get("input_pdb_chains"):
            body["input_pdb_chains"] = kw["input_pdb_chains"]
        return self._post("proteinmpnn", body)


@dataclass
class SimulatedBioNeMo:
    """Deterministic in-process simulator so the whole pipeline runs with no GPU, key or network.

    Returns well-formed but meaningless structures and scores; it exercises the orchestration and the
    ranking, not the biology. Switch to NIMBioNeMo for real design runs.
    """

    seed: int = 0

    def rfdiffusion(self, input_pdb, contigs, hotspot_res=None, **kw):
        import random

        rng = random.Random(self.seed + kw.get("random_seed", 0) + len(contigs))
        n = rng.randint(50, 70)  # backbones differ in length and coordinates per seed
        jitter = rng.random()
        atoms = "\n".join(
            f"ATOM  {i + 1:>5}  CA  GLY B{i + 1:>4}    {i * 3.8 + jitter:8.3f}   0.000   0.000  1.00  0.00           C"
            for i in range(n)
        )
        return {"output_pdb": f"{input_pdb}\n{atoms}\nEND\n", "elapsed_ms": 1200}

    def proteinmpnn(self, input_pdb, **kw):
        n = kw.get("num_seq_per_target", 8)
        import random

        rng = random.Random(self.seed + hash(input_pdb) % 100000)
        designs, scores = [], []
        for i in range(n):
            seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(55))
            score = round(0.8 + 0.4 * rng.random(), 4)  # lower = better (neg-log-prob convention)
            designs.append(f">T=0.1, score={score}, seq={i}\n{seq}")
            scores.append(score)
        return {"mfasta": "\n".join(designs), "scores": scores}

    def boltz2(self, polymers, ligands=None, **kw):
        import random

        rng = random.Random(self.seed + sum(len(p.get("sequence", "")) for p in polymers))
        k = kw.get("diffusion_samples", 1)
        cif = "data_fake\n_struct.title fake\n"
        out = {
            "structures": [{"structure": cif, "format": "mmcif", "name": f"m{i}"} for i in range(k)],
            "confidence_scores": [round(0.4 + 0.5 * rng.random(), 4) for _ in range(k)],
        }
        if any(lig.get("predict_affinity") for lig in (ligands or [])):
            out["affinities"] = {
                "affinity_probability_binary": [round(rng.random(), 4)],
                "affinity_pic50": [round(4 + 4 * rng.random(), 3)],
            }
        return out
