"""Data models for the AI scientist pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Target(BaseModel):
    name: str
    pdb: str = Field(description="Target structure as a PDB string")
    binder_chain_target: str = Field(default="A", description="Chain id of the target in the PDB")
    binder_chain: str | None = Field(
        default=None, description="Explicit generated binder chain; inferred only if unambiguous"
    )
    contigs: str = Field(description="RFdiffusion contig string, e.g. 'A1-100/0 50-100'")
    hotspot_res: list[str] = Field(default_factory=list, description="e.g. ['A50','A51']")
    target_sequence: str = Field(
        default="",
        description="Target amino-acid sequence; when set, the binder is "
        "co-folded WITH the target so the confidence reflects the complex, not the binder alone",
    )
    description: str = ""


class LiteratureBrief(BaseModel):
    summary: str = Field(description="What is known about the target and binder design for it")
    key_points: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list, description="Paper ids cited, e.g. MED:12345678")


class Candidate(BaseModel):
    id: str
    sequence: str
    backbone_index: int
    mpnn_score: float = Field(description="ProteinMPNN score (lower is better)")
    complex_confidence: float | None = None  # Boltz-2 confidence; of the binder-target complex if co-folded
    co_folded: bool = False  # True if the target sequence was included (otherwise binder-only)
    affinity_probability: float | None = None  # only when a ligand with predict_affinity was supplied
    composite: float = 0.0
    rank: int = 0


class ScientistReport(BaseModel):
    target: str
    brief: LiteratureBrief
    n_backbones: int
    n_candidates: int
    top: list[Candidate]
    method: str = ""
    caveats: list[str] = Field(default_factory=list)
