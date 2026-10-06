"""Literature search over Europe PMC (keyless REST API). Returns citable abstracts for the reasoning step."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass
class Paper:
    id: str
    source: str
    title: str
    year: str
    authors: str
    journal: str
    abstract: str
    doi: str | None = None

    @property
    def url(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return f"https://europepmc.org/article/{self.source}/{self.id}"

    def citation(self) -> str:
        return f"[{self.source}:{self.id}] {self.authors} ({self.year}). {self.title}. {self.journal}."


def search(query: str, limit: int = 15, client: httpx.Client | None = None) -> list[Paper]:
    """Search Europe PMC, abstracts included (resultType=core), most recent first."""
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        resp = client.get(
            EUROPE_PMC,
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": limit,
                "sort": "P_PDATE_D desc",
            },
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
    finally:
        if own:
            client.close()
    papers = []
    for r in results:
        if not r.get("abstractText"):
            continue
        papers.append(
            Paper(
                id=r.get("id", ""),
                source=r.get("source", "MED"),
                title=r.get("title", "").strip(),
                year=str(r.get("pubYear", "")),
                authors=r.get("authorString", ""),
                journal=r.get("journalTitle", ""),
                abstract=r.get("abstractText", ""),
                doi=r.get("doi"),
            )
        )
    return papers
