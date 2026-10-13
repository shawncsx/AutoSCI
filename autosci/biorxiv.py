"""AutoSCI — bioRxiv ingester (v0.9).

bioRxiv hosts the bulk of biology preprints (microbiology,
bioinformatics, etc.) that never reach arXiv. Their public API is:

    https://api.biorxiv.org/details/biorxiv/{server}/{from_date}/{to_date}/{cursor}

It returns JSON with abstract + authors + DOI.

The endpoint is rate-limited (50 req/s with API key, 1 req/s without)
and only returns up to ~100 records per call. We loop over multiple
date windows to fetch a meaningful batch.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import urllib.parse as urllib_parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


log = logging.getLogger("autosci.biorxiv")


BIORXIV_API = "https://api.biorxiv.org"


@dataclass
class BiorxivPaper:
    biorxiv_id: str       # e.g. "2024.05.15.123456v1"
    doi: str
    title: str
    authors: list[str]
    abstract: str
    category: str
    posted: str
    link: str

    @property
    def cite_key(self) -> str:
        return f"[biorxiv_{self.biorxiv_id.replace('/', '_').replace('.', '_')}]"


def _chunk_date_range(start: dt.date, end: dt.date, days: int = 30) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end] into `days`-wide chunks."""
    out: list[tuple[dt.date, dt.date]] = []
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=days), end)
        out.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)
    return out


def fetch_biorxiv_window(
    start: dt.date,
    end: dt.date,
    *,
    server: str = "biorxiv",  # or "medrxiv"
    max_records: int = 100,
    client: Optional[httpx.Client] = None,
    timeout: float = 30.0,
) -> list[BiorxivPaper]:
    """Fetch all bioRxiv papers in [start, end] (max 100 per call)."""
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    url = f"{BIORXIV_API}/details/{server}/{start.isoformat()}/{end.isoformat()}/0"
    try:
        r = client.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("biorxiv fetch %s..%s failed: %s", start, end, e)
        return []
    try:
        data = r.json() or {}
    except json.JSONDecodeError:
        return []
    out: list[BiorxivPaper] = []
    for entry in (data.get("collection") or [])[:max_records]:
        try:
            abstract = entry.get("abstract") or ""
            if not abstract:
                continue
            out.append(BiorxivPaper(
                biorxiv_id=entry.get("biorxiv_id") or entry.get("doi") or "",
                doi=(entry.get("doi") or "").replace("https://doi.org/", ""),
                title=entry.get("title") or "",
                authors=[a.strip() for a in (entry.get("authors") or "").split(";") if a.strip()],
                abstract=abstract,
                category=(entry.get("category") or ""),
                posted=entry.get("posted") or "",
                link=f"https://www.{server}.org/content/{entry.get('biorxiv_id') or ''}",
            ))
        except Exception as e:
            log.debug("bad biorxiv entry: %s", e)
    time.sleep(0.5)  # be polite
    return out


def fetch_biorxiv_search(
    query: str,
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    max_records: int = 50,
    client: Optional[httpx.Client] = None,
    timeout: float = 30.0,
) -> list[BiorxivPaper]:
    """Search bioRxiv for a term.

    Note: bioRxiv's /pubs/{start}/{end}/{term} endpoint has been
    intermittently 403-ing in 2024+. We fall back to fetching the
    full date range and filtering client-side.
    """
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    if end is None:
        end = dt.date.today()
    if start is None:
        start = end - dt.timedelta(days=180)  # 6 months back by default

    # Try the new endpoint first
    url = f"{BIORXIV_API}/pubs/biorxiv/{start.isoformat()}/{end.isoformat()}/{urllib_parse.quote(query)}/0"
    try:
        r = client.get(url)
        if r.status_code == 200:
            data = r.json() or {}
            out: list[BiorxivPaper] = []
            for entry in (data.get("collection") or [])[:max_records]:
                paper = _entry_to_paper(entry)
                if paper and query.lower() in (paper.title + " " + paper.abstract).lower():
                    out.append(paper)
            return out
        log.warning("biorxiv /pubs returned %d; falling back to /details range scan",
                    r.status_code)
    except httpx.HTTPError as e:
        log.warning("biorxiv /pubs failed: %s; falling back to /details range scan", e)

    # Fallback: pull all papers in the date range and filter client-side
    out = []
    for s, e in _chunk_date_range(start, end, days=30):
        papers = fetch_biorxiv_window(s, e, max_records=100, client=client, timeout=timeout)
        for p in papers:
            if query.lower() in (p.title + " " + p.abstract).lower():
                out.append(p)
                if len(out) >= max_records:
                    return out
        time.sleep(0.5)
    return out


def _entry_to_paper(entry: dict) -> BiorxivPaper | None:
    try:
        abstract = entry.get("abstract") or ""
        if not abstract:
            return None
        return BiorxivPaper(
            biorxiv_id=entry.get("biorxiv_id") or entry.get("doi") or "",
            doi=(entry.get("doi") or "").replace("https://doi.org/", ""),
            title=entry.get("title") or "",
            authors=[a.strip() for a in (entry.get("authors") or "").split(";") if a.strip()],
            abstract=abstract,
            category=(entry.get("category") or ""),
            posted=entry.get("posted") or "",
            link=f"https://www.biorxiv.org/content/{entry.get('biorxiv_id') or ''}",
        )
    except Exception:
        return None


def write_as_digests(papers: list[BiorxivPaper], out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for p in papers:
        if p.biorxiv_id in seen or not p.biorxiv_id:
            continue
        seen.add(p.biorxiv_id)
        body = (
            f"# biorxiv:{p.biorxiv_id}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {p.title}\n\n"
            f"**Authors:** {', '.join(p.authors) or 'unknown'}\n\n"
            f"**Category:** {p.category}\n\n"
            f"**Posted:** {p.posted}\n\n"
            f"**DOI:** {p.doi or '—'}\n\n"
            f"**Link:** {p.link}\n\n"
            f"**Abstract:**\n\n{p.abstract}\n"
        )
        target = out_dir / f"biorxiv_{p.biorxiv_id.replace('/', '_').replace('.', '_').replace(':', '_')}.digest.md"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written
