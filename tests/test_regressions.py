import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ai_scientist.bionemo import BioNeMoSettings, NIMBioNeMo, SimulatedBioNeMo
from ai_scientist.pipeline import design_candidates
from ai_scientist.samples import DEMO_TARGET


@pytest.mark.parametrize("seed", [0, 42])
def test_request_payload_preserves_seed_and_chains(monkeypatch, seed):
    client = NIMBioNeMo(BioNeMoSettings(base_url="http://localhost:8000"))
    captured = []
    monkeypatch.setattr(client, "_post", lambda tool, body: captured.append((tool, body)) or {})
    client.rfdiffusion("pdb", "A1-20/0 50-70", random_seed=seed)
    client.proteinmpnn("pdb", input_pdb_chains=["B"])
    assert captured[0][1]["random_seed"] == seed
    assert captured[1][1]["input_pdb_chains"] == ["B"]


def test_async_missing_request_id_fails(monkeypatch):
    client = NIMBioNeMo(BioNeMoSettings(base_url="http://localhost:8000"))
    client._requests = SimpleNamespace(post=lambda *a, **kw: SimpleNamespace(status_code=202, headers={}))
    with pytest.raises(ValueError, match="nvcf-reqid"):
        client.rfdiffusion("pdb", "contigs")


def test_http_error_is_propagated():
    client = NIMBioNeMo(BioNeMoSettings(base_url="http://localhost:8000"))

    def fail():
        raise RuntimeError("HTTP 401")

    client._requests = SimpleNamespace(post=lambda *a, **kw: SimpleNamespace(status_code=401, raise_for_status=fail))
    with pytest.raises(RuntimeError, match="401"):
        client.rfdiffusion("pdb", "contigs")


def test_simulation_stable_across_processes():
    code = "import json; from ai_scientist.bionemo import SimulatedBioNeMo; print(json.dumps(SimulatedBioNeMo().proteinmpnn('pdb')))"
    outputs = [
        subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ["1", "2"]
    ]
    assert outputs[0] == outputs[1]


class SpyBio(SimulatedBioNeMo):
    selected = None

    def proteinmpnn(self, input_pdb, **kw):
        self.selected = kw["input_pdb_chains"]
        return super().proteinmpnn(input_pdb, **kw)


def test_binder_chain_is_selected():
    bio = SpyBio()
    candidates = design_candidates(DEMO_TARGET, bio, 1, 2)
    assert bio.selected == ["B"] and len(candidates) == 2


def test_wrong_sequence_length_is_rejected():
    class WrongBio(SpyBio):
        def proteinmpnn(self, input_pdb, **kw):
            return {"mfasta": ">score=1.0\n" + "A" * 20}

    with pytest.raises(ValueError, match="map"):
        design_candidates(DEMO_TARGET, WrongBio(), 1, 1)


def test_target_chain_cannot_be_designed_as_binder():
    with pytest.raises(ValueError, match="differ"):
        design_candidates(DEMO_TARGET.model_copy(update={"binder_chain": "A"}), SpyBio(), 1, 1)


async def test_nat_plugin_registration_and_workflow(monkeypatch):
    monkeypatch.delenv("BIONEMO_BASE_URL", raising=False)
    pytest.importorskip("nat")
    from ai_scientist.nat_plugin.register import DesignBindersConfig, design_binders

    class Builder:
        async def get_llm(self, *args, **kw):
            return object()

    async with design_binders(
        DesignBindersConfig(llm_name="fake", backbones=1, seqs_per_backbone=1), Builder()
    ) as info:
        assert info is not None
        report = await info.single_fn("Example target")
        assert "Candidate binders" in report and "Not validated" in report


def test_successful_async_response_parsing():
    client = NIMBioNeMo(BioNeMoSettings(base_url="http://localhost:8000"))
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"output_pdb": "ATOM"})

    client._requests = SimpleNamespace(
        post=lambda *a, **kw: SimpleNamespace(status_code=202, headers={"nvcf-reqid": "request-id"}), get=get
    )
    assert client.rfdiffusion("pdb", "contigs")["output_pdb"] == "ATOM"
    assert calls == ["http://localhost:8000/v1/status/request-id"]


def test_demo_cannot_send_placeholder_to_live_backend():
    from typer.testing import CliRunner

    from ai_scientist.cli import app

    result = CliRunner().invoke(app, ["demo", "--real-bio"])
    assert result.exit_code != 0
    assert "placeholder" in result.output
