"""AutoSCI — OpenAlex ingester (v0.8).

OpenAlex (https://api.openalex.org) indexes 250M+ scholarly works with
no API key required. We use it as a *complement* to arXiv: arXiv is
strong on physics/CS/ML, while OpenAlex covers clinical, life-sciences,
materials, etc. that arXiv may not have.

We deliberately use the polite `mailto=` parameter so OpenAlex gives us
a more generous rate limit (10 req/s vs 1 req/s).

OpenAlex returns works with a `abstract_inverted_index` (a word→position
map, NOT a plain abstract). We reconstruct the abstract by sorting by
position. This is the standard recipe.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import httpx


log = logging.getLogger("autosci.openalex")


OPENALEX_API = "https://api.openalex.org/works"


@dataclass
class OpenAlexWork:
    openalex_id: str
    doi: str
    title: str
    abstract: str
    year: int
    authors: list[str]
    venue: str
    cited_by_count: int
    concepts: list[str] = field(default_factory=list)

    @property
    def cite_key(self) -> str:
        # Use last segment of openalex_id (W...) for short, stable keys.
        short = self.openalex_id.split("/")[-1] if "/" in self.openalex_id else self.openalex_id
        return f"[openalex_{short}]"


def _reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex gives us a {token: [positions]} map. Sort by position."""
    if not inv:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inv.items():
        for p in positions or []:
            pairs.append((p, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def fetch_openalex(
    query: str,
    max_results: int = 50,
    *,
    mailto: str = "autosci@example.com",
    client: Optional[httpx.Client] = None,
    timeout: float = 20.0,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> list[OpenAlexWork]:
    """Search OpenAlex for ``query``."""
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    params: list[tuple[str, str]] = [
        ("search", query),
        ("per_page", str(min(max_results, 200))),
        ("mailto", mailto),
    ]
    if year_from or year_to:
        yf = year_from or 0
        yt = year_to or 9999
        params.append(("filter", f"publication_year:{yf}-{yt}"))

    try:
        r = client.get(OPENALEX_API, params=params)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("openalex query '%s' failed: %s", query, e)
        return []
    data = r.json() or {}
    results = data.get("results") or []
    out: list[OpenAlexWork] = []
    for w in results:
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        if not abstract:
            # OpenAlex often lacks abstracts for some works; skip
            continue
        year = w.get("publication_year") or 0
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = 0
        authorships = w.get("authorships") or []
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in authorships
            if isinstance(a, dict)
        ]
        venue = (w.get("primary_location") or {}).get("source", {}) or {}
        venue_name = (venue.get("display_name") or "") if isinstance(venue, dict) else ""
        concepts = [
            (c.get("display_name") or "")
            for c in (w.get("concepts") or [])[:6]
            if isinstance(c, dict)
        ]
        out.append(OpenAlexWork(
            openalex_id=w.get("id", ""),
            doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            title=w.get("title") or "",
            abstract=abstract,
            year=year,
            authors=[a for a in authors if a],
            venue=venue_name,
            cited_by_count=int(w.get("cited_by_count") or 0),
            concepts=[c for c in concepts if c],
        ))
    time.sleep(0.2)  # polite — we already have a mailto so 5/s is OK
    return out


def write_as_digests(works: Iterable[OpenAlexWork], out_dir: Path | str) -> list[Path]:
    """Persist OpenAlex works as AutoSCI digests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for w in works:
        if w.openalex_id in seen:
            continue
        seen.add(w.openalex_id)
        body = (
            f"# openalex:{w.openalex_id}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {w.title}\n\n"
            f"**Authors:** {', '.join(w.authors) or 'unknown'}\n\n"
            f"**Year:** {w.year}\n\n"
            f"**Venue:** {w.venue or 'unknown'}\n\n"
            f"**DOI:** {w.doi or '—'}\n\n"
            f"**Cited by:** {w.cited_by_count}\n\n"
            f"**Concepts:** {', '.join(w.concepts) or '—'}\n\n"
            f"**Abstract:**\n\n{w.abstract}\n"
        )
        short = w.openalex_id.split("/")[-1] if "/" in w.openalex_id else w.openalex_id
        target = out_dir / f"openalex_{short}.digest.md"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written
