"""AutoSCI — PubMed / PMC ingester (v0.9).

Uses NCBI's E-utilities API to search and fetch papers from PubMed
(36M+ citations) and PubMed Central (9M+ full-text articles).

Endpoints (no API key required, but rate-limited to 3 req/s without key):
    ESEARCH  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
    EFETCH   https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
    ESUMMARY https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi

Typical flow:
    1. esearch returns a list of PMIDs for a query
    2. efetch (or esummary) returns full records (title, abstract, authors,
       journal, MeSH terms, DOI)
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


log = logging.getLogger("autosci.pubmed")


ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


@dataclass
class PubmedPaper:
    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    doi: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    ptype: list[str] = field(default_factory=list)

    @property
    def cite_key(self) -> str:
        return f"[pubmed_{self.pmid}]"


def _extract_text(xml: str, tag: str) -> str:
    """Cheap XML tag extraction (we don't pull in lxml)."""
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S | re.I)
    if not m:
        return ""
    return re.sub(r"<[^>]+>", "", m.group(1)).strip()


def _extract_all(xml: str, tag: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S | re.I):
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if text:
            out.append(text)
    return out


def _extract_year(xml: str) -> int:
    m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 0
    return 0


def _extract_doi(xml: str) -> str:
    m = re.search(r"<ArticleId IdType=\"doi\">(.*?)</ArticleId>", xml)
    if m:
        return m.group(1).strip()
    return ""


def esearch(query: str, *, retmax: int = 50, mindate: str = "", maxdate: str = "",
            client: Optional[httpx.Client] = None, timeout: float = 20.0) -> list[str]:
    """Return a list of PMIDs matching ``query``."""
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "sort": "relevance",
    }
    if mindate:
        params["mindate"] = mindate
    if maxdate:
        params["maxdate"] = maxdate
    try:
        r = client.get(ESEARCH, params=params)
        r.raise_for_status()
        j = r.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        log.warning("pubmed esearch '%s' failed: %s", query, e)
        return []
    pmids = (j.get("esearchresult") or {}).get("idlist") or []
    time.sleep(0.4)  # be polite (3 req/s without key)
    return list(pmids)


def efetch_records(pmids: list[str], *,
                   client: Optional[httpx.Client] = None,
                   batch: int = 50) -> list[PubmedPaper]:
    """Fetch full records for a list of PMIDs (batched)."""
    if not pmids:
        return []
    client = client or httpx.Client(timeout=30, trust_env=False)
    out: list[PubmedPaper] = []
    for i in range(0, len(pmids), batch):
        chunk = pmids[i: i + batch]
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "retmode": "xml",
            "rettype": "abstract",
        }
        try:
            r = client.get(EFETCH, params=params)
            r.raise_for_status()
            xml = r.text
        except httpx.HTTPError as e:
            log.warning("pubmed efetch failed: %s", e)
            continue

        # Each article is wrapped in <PubmedArticle>...</PubmedArticle>
        for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.S):
            try:
                title = _extract_text(art, "ArticleTitle")
                abstract = _extract_text(art, "Abstract")
                if not abstract:
                    # try the structured abstract
                    parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S)
                    if parts:
                        abstract = " ".join(re.sub(r"<[^>]+>", "", p).strip() for p in parts)
                if not abstract:
                    continue
                authors = _extract_all(art, "LastName")
                # better: parse ForeName + LastName pairs
                names = re.findall(r"<LastName>(.*?)</LastName>\s*<ForeName>(.*?)</ForeName>", art, re.S)
                if names:
                    authors = [f"{fn} {ln}".strip() for ln, fn in names]
                journal = _extract_text(art, "Title") or _extract_text(art, "MedlineTA")
                year = _extract_year(art)
                doi = _extract_doi(art)
                pmid = _extract_text(art, "PMID")
                mesh = _extract_all(art, "DescriptorName")
                ptype = _extract_all(art, "PublicationType")
                if not pmid:
                    continue
                out.append(PubmedPaper(
                    pmid=pmid, title=title, abstract=abstract, authors=authors,
                    journal=journal, year=year, doi=doi, mesh_terms=mesh, ptype=ptype,
                ))
            except Exception as e:
                log.debug("bad pubmed entry: %s", e)
        time.sleep(0.4)
    return out


def search_pubmed(
    query: str, *, max_records: int = 50, mindate: str = "", maxdate: str = "",
) -> list[PubmedPaper]:
    """One-shot: search + fetch."""
    pmids = esearch(query, retmax=max_records, mindate=mindate, maxdate=maxdate)
    return efetch_records(pmids)


def write_as_digests(papers: list[PubmedPaper], out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for p in papers:
        if p.pmid in seen or not p.pmid:
            continue
        seen.add(p.pmid)
        body = (
            f"# pubmed:{p.pmid}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {p.title}\n\n"
            f"**Authors:** {', '.join(p.authors) or 'unknown'}\n\n"
            f"**Journal:** {p.journal or '—'}\n\n"
            f"**Year:** {p.year}\n\n"
            f"**DOI:** {p.doi or '—'}\n\n"
            f"**MeSH terms:** {', '.join(p.mesh_terms[:8]) or '—'}\n\n"
            f"**Publication types:** {', '.join(p.ptype[:4]) or '—'}\n\n"
            f"**Abstract:**\n\n{p.abstract}\n"
        )
        target = out_dir / f"pubmed_{p.pmid}.digest.md"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written
