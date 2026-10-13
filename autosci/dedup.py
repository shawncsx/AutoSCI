"""AutoSCI — Cross-source dedup (v0.9).

When we ingest papers from 5+ sources (arXiv, OpenAlex, bioRxiv, PubMed,
Semantic Scholar) many of the same papers appear under different IDs
(arXiv-id vs DOI vs PMID vs S2-paperId). The dedup uses three keys in
priority order:

    1. DOI  — most authoritative
    2. arXiv-id — strong for CS/ML/quant bio
    3. title-hash — fallback for when no IDs match

The dedup also *merges* records when the same paper appears in
multiple sources: it keeps the richest abstract (longest), prefers
s2 venue info if present, and concatenates all cite_keys under a
single canonical entry.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


log = logging.getLogger("autosci.dedup")


def _normalize_doi(doi: str) -> str:
    return (doi or "").strip().lower().replace("https://doi.org/", "")


def _normalize_arxiv_id(s: str) -> str:
    """Strip URL prefix and version suffix from an arXiv id."""
    s = (s or "").strip()
    # strip 'http://arxiv.org/abs/' etc.
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    # strip version suffix case-insensitively (e.g. v1, V1)
    s = re.sub(r"[vV]\d+$", "", s)
    return s.lower()


def title_hash(title: str) -> str:
    """Stable hash from a normalised title (lowercase, alnum + CJK only)."""
    if not title:
        return ""
    s = title.lower()
    s = re.sub(r"[\s\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    if len(s) < 8:
        return ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class DedupedRecord:
    canonical_id: str           # e.g. doi:10.xxxx, arxiv:2402.16901, title-hash:abc
    source: list[str] = field(default_factory=list)
    title: str = ""
    abstract: str = ""
    year: int = 0
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    s2_id: str = ""
    cite_keys: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "source": self.source,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "s2_id": self.s2_id,
            "cite_keys": self.cite_keys,
        }


def digest_to_record(digest_path: Path) -> DedupedRecord | None:
    """Parse a *.digest.md file we already wrote for arxiv/openalex/biorxiv/
    pubmed/s2 sources, and produce a DedupedRecord.
    """
    try:
        text = digest_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    # The first line of our digests is `# <source>:<id>`
    first = text.splitlines()[0] if text else ""
    m = re.match(r"^#\s*(\w+):(.+?)\s*$", first)
    source, raw_id = (m.group(1), m.group(2)) if m else ("unknown", digest_path.stem)
    # Pull title/abstract/year from the structured `## [开头/摘要]` block
    abstract_match = re.search(r"\*\*Abstract:\*\*\s*(.+?)(?=\n## |\Z)", text, re.S)
    abstract = (abstract_match.group(1).strip() if abstract_match else "").strip()
    title_match = re.search(r"\*\*Title:\*\*\s*(.+?)\n", text)
    title = title_match.group(1).strip() if title_match else ""
    year_match = re.search(r"\*\*Year:\*\*\s*(\d{4})", text)
    year = int(year_match.group(1)) if year_match else 0
    doi_match = re.search(r"\*\*DOI:\*\*\s*([^\n]+)", text)
    doi = _normalize_doi(doi_match.group(1)) if doi_match and doi_match.group(1) != "—" else ""
    arxiv_match = re.search(r"\*\*(?:arXiv:|arXiv ID:|ArXiv:)\*\*\s*([^\n]+)", text)
    arxiv_id = _normalize_arxiv_id(arxiv_match.group(1)) if arxiv_match else ""
    # Also try to extract arXiv id from the H1 line: `# arxiv:2402.16901`
    if not arxiv_id:
        arxiv_h1 = re.search(r"^# arxiv:([\w\.\-]+)", text, re.M)
        if arxiv_h1:
            arxiv_id = _normalize_arxiv_id(arxiv_h1.group(1))
    pmid_match = re.search(r"^# pubmed:(\d+)", text, re.M)
    pmid = pmid_match.group(1) if pmid_match else ""
    s2_match = re.search(r"^# s2:([0-9a-f]+)", text, re.M)
    s2_id = s2_match.group(1) if s2_match else ""
    venue_match = re.search(r"\*\*Venue:\*\*\s*([^\n]+)", text)
    venue = venue_match.group(1).strip() if venue_match else ""

    # Build canonical id
    if doi:
        canonical_id = f"doi:{doi}"
    elif arxiv_id:
        canonical_id = f"arxiv:{arxiv_id}"
    elif pmid:
        canonical_id = f"pmid:{pmid}"
    elif s2_id:
        canonical_id = f"s2:{s2_id}"
    else:
        canonical_id = f"title:{title_hash(title)}"
    return DedupedRecord(
        canonical_id=canonical_id,
        source=[source],
        title=title,
        abstract=abstract,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        pmid=pmid,
        s2_id=s2_id,
        cite_keys=[f"[{source}_{raw_id.replace('/', '_').replace('.', '_').replace(':', '_')}]"],
    )


def dedup_digests(digest_dir: Path) -> tuple[list[DedupedRecord], list[Path]]:
    """Walk ``digest_dir`` and return (deduped_records, paths_to_delete).

    The caller should then delete the returned paths to remove duplicates
    and rewrite a single canonical digest per record.
    """
    by_canonical: dict[str, DedupedRecord] = {}
    paths_to_delete: list[Path] = []
    files = sorted(digest_dir.glob("*.digest.md"))
    for fp in files:
        rec = digest_to_record(fp)
        if rec is None or not rec.abstract:
            continue
        if rec.canonical_id in by_canonical:
            existing = by_canonical[rec.canonical_id]
            # merge
            existing.source = list(set(existing.source) | set(rec.source))
            existing.cite_keys = list(set(existing.cite_keys) | set(rec.cite_keys))
            if rec.abstract and len(rec.abstract) > len(existing.abstract):
                existing.abstract = rec.abstract
            if not existing.venue and rec.venue:
                existing.venue = rec.venue
            if not existing.year and rec.year:
                existing.year = rec.year
            for f in ("doi", "arxiv_id", "pmid", "s2_id"):
                if not getattr(existing, f) and getattr(rec, f):
                    setattr(existing, f, getattr(rec, f))
            paths_to_delete.append(fp)
        else:
            by_canonical[rec.canonical_id] = rec
    log.info("dedup: %d digests → %d unique (dropped %d)", len(files), len(by_canonical), len(paths_to_delete))
    return list(by_canonical.values()), paths_to_delete


def write_deduped_digests(records: list[DedupedRecord], out_dir: Path) -> int:
    """Rewrite one canonical digest per record (overwriting in place)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rec in records:
        # build a stable filename from the canonical_id
        safe = rec.canonical_id.replace(":", "_").replace("/", "_")
        target = out_dir / f"deduped_{safe}.digest.md"
        sources = ",".join(rec.source)
        body = (
            f"# {rec.canonical_id}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {rec.title}\n\n"
            f"**Sources:** {sources}\n\n"
            f"**Year:** {rec.year}\n\n"
            f"**Venue:** {rec.venue or '—'}\n\n"
            f"**DOI:** {rec.doi or '—'}\n"
            f"**arXiv:** {rec.arxiv_id or '—'}\n"
            f"**PMID:** {rec.pmid or '—'}\n"
            f"**S2 ID:** {rec.s2_id or '—'}\n\n"
            f"**Cite keys:** {' '.join(rec.cite_keys)}\n\n"
            f"**Abstract:**\n\n{rec.abstract}\n"
        )
        target.write_text(body, encoding="utf-8")
        written += 1
    return written
