"""AutoSCI — ArxivIngester (v0.4 new).

Hit the public arXiv API to fetch abstracts for a query. Used by the
IdeaMiner to enrich the KB when the local corpus doesn't cover a
proposed direction. We deliberately use the public Atom API (no key)
with a polite 3s sleep between calls; we also dedupe on arxiv-id.

The fetched abstracts are written as synthetic ``*.digest.md`` files
into a dedicated folder so the existing KB indexer picks them up.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import httpx


log = logging.getLogger("autosci.arxiv")


ARXIV_API = "https://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    year: int
    categories: list[str]
    link: str

    @property
    def cite_key(self) -> str:
        # [arxiv_2402.16901_...]
        return f"[arxiv_{self.arxiv_id.replace('/', '_').replace('.', '_')}]"


def _parse_atom(xml_text: str) -> list[ArxivEntry]:
    root = ET.fromstring(xml_text)
    out: list[ArxivEntry] = []
    for entry in root.findall("atom:entry", NS):
        # arXiv id lives in the atom namespace (not arxiv)
        arxiv_id_full = (entry.findtext("atom:id", default="", namespaces=NS) or "").strip()
        # arXiv id like "http://arxiv.org/abs/2402.16901v1" → "2402.16901"
        arxiv_id = arxiv_id_full.split("/abs/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        if not arxiv_id:
            continue
        title = " ".join((entry.findtext("atom:title", default="", namespaces=NS) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=NS) or "").split())
        year_str = (entry.findtext("atom:published", default="", namespaces=NS) or "")[:4]
        try:
            year = int(year_str)
        except ValueError:
            year = 0
        authors = [
            (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
            for a in entry.findall("atom:author", NS)
        ]
        cats = [
            (c.attrib.get("term", "")).strip()
            for c in entry.findall("arxiv:category", NS)
        ]
        link = ""
        for l in entry.findall("atom:link", NS):
            if l.attrib.get("type") == "text/html":
                link = l.attrib.get("href", "")
        out.append(ArxivEntry(
            arxiv_id=arxiv_id, title=title, summary=summary,
            authors=[a for a in authors if a], year=year,
            categories=cats, link=link,
        ))
    return out


def fetch_arxiv(
    query: str,
    max_results: int = 5,
    *,
    client: Optional[httpx.Client] = None,
    timeout: float = 15.0,
    categories: Optional[list[str]] = None,
) -> list[ArxivEntry]:
    """Search arXiv for ``query`` and return up to ``max_results`` entries.

    Note: arXiv's public API does NOT accept quoted multi-word queries; we
    send them as plain tokens (each word matched by OR by default) which
    is what most users actually want.
    """
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    params: list[tuple[str, str]] = [
        ("search_query", query),
        ("start", "0"),
        ("max_results", str(max_results)),
    ]
    if categories:
        # Restrict to specific archive categories if requested
        params.append(("search_query", " OR ".join(f"cat:{c}" for c in categories)))

    try:
        resp = client.get(ARXIV_API, params=params)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("arxiv fetch failed: %s", e)
        return []

    entries = _parse_atom(resp.text)
    time.sleep(3.0)  # be polite — arXiv asks for >= 3s between unauthenticated calls
    return entries


def write_as_digests(entries: Iterable[ArxivEntry], out_dir: Path | str) -> list[Path]:
    """Persist fetched arXiv entries as AutoSCI digests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for e in entries:
        if e.arxiv_id in seen:
            continue
        seen.add(e.arxiv_id)
        body = (
            f"# arxiv:{e.arxiv_id}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {e.title}\n\n"
            f"**Authors:** {', '.join(e.authors) or 'unknown'}\n\n"
            f"**Year:** {e.year}\n\n"
            f"**arXiv:** {e.link or e.arxiv_id}\n\n"
            f"**Categories:** {', '.join(e.categories) or 'unknown'}\n\n"
            f"**Abstract:**\n\n{e.summary}\n"
        )
        target = out_dir / f"arxiv_{e.arxiv_id.replace('/', '_').replace('.', '_')}.digest.md"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written
