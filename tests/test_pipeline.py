from ai_scientist.bionemo import SimulatedBioNeMo
from ai_scientist.literature import Paper
from ai_scientist.llm import FakeLLM
from ai_scientist.pipeline import design_candidates, literature_brief, parse_mfasta, rank, report_markdown, run
from ai_scientist.samples import DEMO_TARGET
from ai_scientist.schemas import Candidate


def test_parse_mfasta():
    mf = ">T=0.1, score=0.95, seq=0\nACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY\n>T=0.1, score=1.2, seq=1\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIV"
    out = parse_mfasta(mf)
    assert len(out) == 2 and out[0][0] == 0.95 and out[0][1].startswith("ACDEF")


def test_design_and_rank_deterministic():
    cands = design_candidates(DEMO_TARGET, SimulatedBioNeMo(seed=1), n_backbones=2, seqs_per_backbone=3)
    assert len(cands) == 6
    assert len({c.sequence for c in cands}) >= 4  # backbones produce distinct designs
    ranked = rank(cands)
    assert ranked[0].rank == 1
    assert all(ranked[i].composite >= ranked[i + 1].composite for i in range(len(ranked) - 1))
    assert all(0 <= c.composite <= 1 for c in ranked)


def test_rank_prefers_better_metrics():
    good = Candidate(id="g", sequence="A", backbone_index=0, mpnn_score=0.8, complex_confidence=0.95)
    bad = Candidate(id="b", sequence="A", backbone_index=0, mpnn_score=1.3, complex_confidence=0.4)
    assert rank([bad, good])[0].id == "g"


async def test_literature_brief_filters_citations():
    papers = [
        Paper(
            id="111",
            source="MED",
            title="PD-L1 binder",
            year="2024",
            authors="A",
            journal="J",
            abstract="A nanobody binds PD-L1.",
        )
    ]
    llm = FakeLLM(
        {
            "literature_brief": {
                "summary": "PD-L1 is druggable.",
                "key_points": ["binders exist"],
                "citations": ["MED:111", "MED:999"],
            }
        }
    )  # 999 is hallucinated
    brief = await literature_brief(DEMO_TARGET, llm, papers=papers)
    assert brief.citations == ["MED:111"]  # the fake id is dropped


async def test_full_run_offline():
    llm = FakeLLM({"literature_brief": {"summary": "s", "key_points": [], "citations": []}})
    report = await run(DEMO_TARGET, llm, SimulatedBioNeMo(), n_backbones=2, seqs_per_backbone=2, top_k=3, papers=[])
    assert report.n_candidates == 4 and len(report.top) == 3
    md = report_markdown(report)
    assert "Candidate binders" in md and "Caveats" in md and "require wet-lab validation" in md.lower()
