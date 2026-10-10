"""Explicit opt-in NVIDIA RFdiffusion/ProteinMPNN contract check using a real input fixture."""

import json
import os
from pathlib import Path

import pytest

from ai_scientist.bionemo import NIMBioNeMo

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_NVIDIA") != "1", reason="Live calls require explicit RUN_LIVE_NVIDIA=1"
)


def test_live_backbone_and_sequence_contract():
    path = os.getenv("BIONEMO_TARGET_JSON")
    if not path:
        pytest.fail("BIONEMO_TARGET_JSON must point to a real scientific target fixture")
    target = json.loads(Path(path).read_text())
    bio = NIMBioNeMo()
    backbone = bio.rfdiffusion(target["pdb"], target["contigs"], target.get("hotspot_res"), random_seed=0)
    assert isinstance(backbone["output_pdb"], str) and "ATOM" in backbone["output_pdb"]
    chain = target["binder_chain"]
    residues = {
        line[22:27] for line in backbone["output_pdb"].splitlines() if line.startswith("ATOM") and line[21:22] == chain
    }
    assert residues, "Configured binder chain is absent from generated structure"
    output = bio.proteinmpnn(backbone["output_pdb"], num_seq_per_target=1, input_pdb_chains=[chain])
    assert isinstance(output["mfasta"], str) and ">" in output["mfasta"]
