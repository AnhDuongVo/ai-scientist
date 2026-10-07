"""AI scientist: an agent that designs candidate protein binders for a target, end to end.

    literature (Europe PMC + reasoning LLM) -> design backbones (RFdiffusion)
      -> sequences (ProteinMPNN) -> co-fold and score the complex (Boltz-2) -> rank -> report for a scientist

This orchestrates NVIDIA BioNeMo NIMs the way the Protein Binder Design blueprint does. The scientific
scoring is deliberately simple and documented; the output is a traceable, ranked report,
not a new method. Everything runs offline with SimulatedBioNeMo so the pipeline and ranking are testable.
"""

from __future__ import annotations

import re

from .bionemo import BioNeMo
from .literature import Paper, search
from .llm import LLM, complete_structured
from .schemas import Candidate, LiteratureBrief, ScientistReport, Target

SYSTEM = (
    "You are a careful computational biologist. You summarise only what the provided abstracts support "
    "and you cite them by id. You never invent experimental results."
)
BRIEF = """Summarise what is known about designing protein binders for this target, from these abstracts only.
Give a short summary, 3-6 key points, and cite the paper ids (e.g. MED:12345678) you used.

Target: {target}

Abstracts:
{abstracts}
"""

AA = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{20,}")


def parse_mfasta(mfasta: str) -> list[tuple[float, str]]:
    """[(score, sequence)] from ProteinMPNN multi-FASTA output (header: '... score=1.23 ...')."""
    out = []
    score = None
    for line in mfasta.splitlines():
        if line.startswith(">"):
            m = re.search(r"score=([-\d.]+)", line)
            score = float(m.group(1)) if m else None
        elif AA.fullmatch(line.strip()):
            out.append((score if score is not None else 0.0, line.strip()))
    return out


async def literature_brief(
    target: Target, llm: LLM, papers: list[Paper] | None = None, max_papers: int = 12
) -> LiteratureBrief:
    papers = (
        papers
        if papers is not None
        else search(f'{target.name} AND (binder OR "binding protein" OR inhibitor)', limit=max_papers)
    )
    if not papers:
        return LiteratureBrief(summary=f"No abstracts retrieved for {target.name}.", key_points=[], citations=[])
    abstracts = "\n\n".join(f"[{p.source}:{p.id}] {p.title}\n{p.abstract[:1200]}" for p in papers[:max_papers])
    brief = await complete_structured(
        llm,
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": BRIEF.format(target=f"{target.name}. {target.description}", abstracts=abstracts),
            },
        ],
        LiteratureBrief,
        role="reasoning",
        step="literature_brief",
    )
    valid = {f"{p.source}:{p.id}" for p in papers}
    brief.citations = [c for c in brief.citations if c in valid]
    return brief


def design_candidates(
    target: Target, bio: BioNeMo, n_backbones: int = 4, seqs_per_backbone: int = 4
) -> list[Candidate]:
    """RFdiffusion backbones -> ProteinMPNN sequences -> Boltz-2 co-fold of the binder-target complex."""
    candidates: list[Candidate] = []
    target_seq = target.target_sequence.strip()  # when provided, we co-fold the complex
    for b in range(n_backbones):
        backbone = bio.rfdiffusion(target.pdb, target.contigs, target.hotspot_res, random_seed=b)
        pdb = backbone["output_pdb"]
        designs = parse_mfasta(bio.proteinmpnn(pdb, num_seq_per_target=seqs_per_backbone)["mfasta"])
        for s, (score, seq) in enumerate(designs):
            polymers = [{"id": "A", "molecule_type": "protein", "sequence": seq}]
            if target_seq:
                polymers.append({"id": "T", "molecule_type": "protein", "sequence": target_seq})
            fold = bio.boltz2(polymers, diffusion_samples=1)
            conf = fold.get("confidence_scores") or [None]
            aff = fold.get("affinities", {})
            candidates.append(
                Candidate(
                    id=f"b{b}s{s}",
                    sequence=seq,
                    backbone_index=b,
                    mpnn_score=score,
                    complex_confidence=conf[0],
                    co_folded=bool(target_seq),
                    affinity_probability=(aff.get("affinity_probability_binary") or [None])[0],
                )
            )
    return candidates


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Composite score: higher complex confidence and affinity are better, lower MPNN score is better."""
    has_aff = any(c.affinity_probability is not None for c in candidates)
    for c in candidates:
        conf = c.complex_confidence or 0.0
        mpnn = max(0.0, 1.5 - c.mpnn_score) / 1.5  # map typical 0.8-1.2 to ~[0.2,0.47], lower score -> higher
        if has_aff:
            aff = c.affinity_probability or 0.0
            c.composite = round(0.5 * conf + 0.3 * aff + 0.2 * mpnn, 4)
        else:  # protein binder with no ligand: confidence + sequence quality only
            c.composite = round(0.7 * conf + 0.3 * mpnn, 4)
    ranked = sorted(candidates, key=lambda c: -c.composite)
    for i, c in enumerate(ranked, 1):
        c.rank = i
    return ranked


async def run(
    target: Target,
    llm: LLM,
    bio: BioNeMo,
    n_backbones: int = 4,
    seqs_per_backbone: int = 4,
    top_k: int = 5,
    papers: list[Paper] | None = None,
) -> ScientistReport:
    brief = await literature_brief(target, llm, papers)
    candidates = design_candidates(target, bio, n_backbones, seqs_per_backbone)
    ranked = rank(candidates)
    return ScientistReport(
        target=target.name,
        brief=brief,
        n_backbones=n_backbones,
        n_candidates=len(candidates),
        top=ranked[:top_k],
        method=(
            f"RFdiffusion ({n_backbones} backbones, contigs '{target.contigs}') -> ProteinMPNN "
            f"({seqs_per_backbone}/backbone) -> Boltz-2 "
            + ("co-fold of binder+target" if target.target_sequence else "fold of binder alone")
            + " -> composite rank (0.7 complex confidence + 0.3 inverse MPNN score)."
        ),
        caveats=[
            "Computational screen only; all candidates require wet-lab validation.",
            "Composite weights are a heuristic, not a validated scoring function.",
            (
                "Target sequence supplied: confidence reflects the binder-target complex."
                if target.target_sequence
                else "No target sequence supplied: Boltz-2 folded the binder ALONE, so confidence does not "
                "measure binding. Set target.target_sequence for a complex-level signal."
            ),
        ],
    )


def report_markdown(report: ScientistReport) -> str:
    b = report.brief
    lines = [f"# Candidate binders for {report.target}", "", "## Literature brief", "", b.summary, ""]
    if b.key_points:
        lines += ["Key points:", ""] + [f"- {p}" for p in b.key_points] + [""]
    if b.citations:
        lines += ["Citations: " + ", ".join(b.citations), ""]
    lines += [
        "## Method",
        "",
        report.method,
        "",
        f"## Top {len(report.top)} of {report.n_candidates} candidates",
        "",
        "| Rank | ID | Composite | Confidence | Co-folded | Affinity prob. | MPNN score | Sequence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in report.top:
        seq = c.sequence if len(c.sequence) <= 40 else c.sequence[:37] + "..."
        lines.append(
            f"| {c.rank} | {c.id} | {c.composite} | "
            f"{c.complex_confidence if c.complex_confidence is not None else '-'} | "
            f"{'yes' if c.co_folded else 'no'} | "
            f"{c.affinity_probability if c.affinity_probability is not None else '-'} | {c.mpnn_score} | `{seq}` |"
        )
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in report.caveats]
    lines += ["", "_AI-generated research hypotheses. Not validated. For laboratory follow-up by a scientist._"]
    return "\n".join(lines) + "\n"
