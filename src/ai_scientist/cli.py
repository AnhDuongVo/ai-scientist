"""`sci --help`: AI scientist that designs candidate protein binders on NVIDIA BioNeMo."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from .bionemo import BioNeMoSettings, NIMBioNeMo, SimulatedBioNeMo
from .config import get_settings
from .llm import FakeLLM, NIMClient
from .pipeline import report_markdown, run
from .samples import DEMO_TARGET
from .schemas import Target

app = typer.Typer(add_completion=False, help="AI scientist: literature-informed protein binder design on BioNeMo.")
console = Console()


def _bio(fake: bool):
    return SimulatedBioNeMo() if fake else NIMBioNeMo(BioNeMoSettings())


def _llm(offline: bool):
    """Real NIM client, or a FakeLLM so `demo` runs with no key when the literature step is also faked."""
    if offline:
        return FakeLLM(
            {
                "literature_brief": {
                    "summary": "(offline demo: literature step faked, no API key set)",
                    "key_points": [],
                    "citations": [],
                }
            }
        )
    return NIMClient(get_settings())


def _save(out_dir: Path, report, md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(md)
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2))


@app.command()
def demo(
    out_dir: Path = typer.Option(Path("runs/demo")),
    backbones: int = typer.Option(3),
    seqs: int = typer.Option(3),
    real_bio: bool = typer.Option(False, help="Call real BioNeMo NIMs (needs NGC_API_KEY or a local NIM)"),
    real_lit: bool = typer.Option(False, help="Query Europe PMC (needs network)"),
):
    """Run the bundled demo target end to end (BioNeMo faked unless --real-bio)."""
    if real_bio:
        raise typer.BadParameter(
            "The demo target is a placeholder. Use sci design target.json --real-bio with a real scientific target."
        )
    papers = None if real_lit else []
    llm = _llm(offline=not real_lit)
    report = asyncio.run(run(DEMO_TARGET, llm, _bio(not real_bio), backbones, seqs, papers=papers))
    md = report_markdown(report)
    _save(out_dir, report, md)
    console.print(Markdown(md))
    console.print(f"[green]Saved to {out_dir}/[/]")


@app.command()
def design(
    target_json: Path = typer.Argument(..., exists=True, help="Target JSON (see schemas.Target)"),
    out_dir: Path = typer.Option(Path("runs/latest")),
    backbones: int = typer.Option(4),
    seqs: int = typer.Option(4),
    real_bio: bool = typer.Option(True),
    real_lit: bool = typer.Option(True),
):
    """Design binders for your own target."""
    target = Target.model_validate_json(target_json.read_text())
    llm = NIMClient(get_settings())
    papers = None if real_lit else []
    report = asyncio.run(run(target, llm, _bio(not real_bio), backbones, seqs, papers=papers))
    _save(out_dir, report, report_markdown(report))
    console.print(f"[green]Verdict: {len(report.top)} ranked candidates. Saved to {out_dir}/[/]")


@app.command()
def search_lit(query: str, limit: int = typer.Option(10)):
    """Search Europe PMC and print citations (no LLM, needs network)."""
    from .literature import search

    for p in search(query, limit):
        console.print(f"{p.citation()}  {p.url}")


@app.command()
def models():
    """List reasoning-model IDs served by the configured endpoint."""
    from openai import OpenAI

    settings = get_settings()
    with OpenAI(base_url=settings.base_url, api_key=settings.require_api_key()) as client:
        for model_id in sorted(model.id for model in client.models.list().data):
            console.print(model_id)


if __name__ == "__main__":
    app()
