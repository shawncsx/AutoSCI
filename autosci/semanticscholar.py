"""AutoSCI — Semantic Scholar ingester (v0.9).

Semantic Scholar (S2) covers 200M+ papers across all fields with rich
metadata: paperId, title, abstract, year, venue, citation count, fields
of study, and even a citation graph (references + citations).

Free API endpoints (no key needed for ~1 req/s, with API key 100 req/s):
    https://api.semanticscholar.org/graph/v1/paper/search
    https://api.semanticscholar.org/graph/v1/paper/{id}?fields=...
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


log = logging.getLogger("autosci.semanticscholar")


S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"


@dataclass
class S2Paper:
    paper_id: str
    title: str
    abstract: str
    year: int
    venue: str
    citation_count: int
    reference_count: int
    influential_citation_count: int
    fields_of_study: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    external_ids: dict = field(default_factory=dict)

    @property
    def doi(self) -> str:
        return self.external_ids.get("DOI", "") or ""

    @property
    def arxiv_id(self) -> str:
        return self.external_ids.get("ArXiv", "") or ""

    @property
    def cite_key(self) -> str:
        return f"[s2_{self.paper_id}]"


def fetch_s2(
    query: str,
    *,
    max_results: int = 50,
    year_from: str = "",
    year_to: str = "",
    fields: str = "title,abstract,year,venue,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,authors,externalIds",
    api_key: str = "",
    client: Optional[httpx.Client] = None,
    timeout: float = 20.0,
) -> list[S2Paper]:
    """Search S2 and return up to ``max_results`` papers."""
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    params: list[tuple[str, str]] = [
        ("query", query),
        ("limit", str(min(max_results, 100))),
        ("fields", fields),
    ]
    if year_from or year_to:
        # S2 expects YYYY-YYYY format
        yf = year_from or "1900"
        yt = year_to or str(int(time.strftime("%Y")) + 1)
        params.append(("year", f"{yf}-{yt}"))
    headers: dict = {}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        r = client.get(S2_SEARCH, params=params, headers=headers)
        r.raise_for_status()
        j = r.json() or {}
    except httpx.HTTPError as e:
        log.warning("s2 search '%s' failed: %s", query, e)
        return []
    out: list[S2Paper] = []
    for entry in (j.get("data") or []):
        if not entry.get("abstract"):
            continue  # skip papers with no abstract
        authors_raw = entry.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if isinstance(a, dict)]
        fields_raw = entry.get("fieldsOfStudy") or []
        fields_list = [f for f in fields_raw if isinstance(f, str)]
        ext_ids = entry.get("externalIds") or {}
        if isinstance(ext_ids, list):
            ext_ids = {x.get("source", ""): x.get("id", "") for x in ext_ids if isinstance(x, dict)}
        out.append(S2Paper(
            paper_id=entry.get("paperId", ""),
            title=entry.get("title", ""),
            abstract=entry.get("abstract", ""),
            year=int(entry.get("year") or 0),
            venue=entry.get("venue", "") or "",
            citation_count=int(entry.get("citationCount") or 0),
            reference_count=int(entry.get("referenceCount") or 0),
            influential_citation_count=int(entry.get("influentialCitationCount") or 0),
            fields_of_study=fields_list,
            authors=authors,
            external_ids=ext_ids,
        ))
    time.sleep(1.1)  # S2 asks for 1 req/s without key
    return out


def write_as_digests(papers: list[S2Paper], out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for p in papers:
        if p.paper_id in seen or not p.paper_id:
            continue
        seen.add(p.paper_id)
        ext = p.external_ids
        doi = ext.get("DOI", "")
        arxiv = ext.get("ArXiv", "")
        ext_line = []
        if doi:
            ext_line.append(f"**DOI:** {doi}")
        if arxiv:
            ext_line.append(f"**arXiv:** {arxiv}")
        body = (
            f"# s2:{p.paper_id}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {p.title}\n\n"
            f"**Authors:** {', '.join(p.authors[:8]) or 'unknown'}\n\n"
            f"**Year:** {p.year}\n\n"
            f"**Venue:** {p.venue or '—'}\n\n"
            f"**Citations:** {p.citation_count} (influential: {p.influential_citation_count}) &nbsp;&nbsp; "
            f"**References:** {p.reference_count}\n\n"
            + ("\n".join(ext_line) + "\n\n" if ext_line else "")
            + f"**Fields of study:** {', '.join(p.fields_of_study[:8]) or '—'}\n\n"
            f"**Abstract:**\n\n{p.abstract}\n"
        )
        target = out_dir / f"s2_{p.paper_id.replace('/', '_').replace(':', '_')}.digest.md"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written
