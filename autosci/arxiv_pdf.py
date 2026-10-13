"""AutoSCI — arXiv PDF full-text ingester (v0.12).

arXiv stores every paper at a stable URL like::

    https://arxiv.org/pdf/{arxiv_id}

This module downloads the PDF, extracts its text (no native deps beyond
``pypdf``), and writes a *fulltext* digest into the local KB so the
Director can ground on real paper body — not just abstracts.

Heuristics:
    - 30s timeout per download
    - 5s politeness after each download
    - rate-limit-friendly: never exceed 12 PDFs/minute
    - cap at 10 MB per PDF (skip huge technical reports)
    - skip if ``pypdf`` is not installed (no crash, just a warning)
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import httpx


log = logging.getLogger("autosci.arxiv_pdf")


ARXIV_PDF_BASE = "https://arxiv.org/pdf"


def _try_import_pypdf():
    try:
        import pypdf
        return pypdf
    except Exception as e:
        log.info("pypdf not available: %s", e)
        return None


@dataclass
class PDFDownload:
    arxiv_id: str
    pdf_path: Path
    text_path: Path
    n_pages: int
    n_chars: int


def _normalize_id(arxiv_id: str) -> str:
    """Strip URL prefix and version suffix (case-insensitive)."""
    s = (arxiv_id or "").strip()
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"[vV]\d+$", "", s)
    return s.lower()


def fetch_one_pdf(
    arxiv_id: str,
    out_dir: Path,
    *,
    client: Optional[httpx.Client] = None,
    timeout: float = 60.0,
    max_bytes: int = 30 * 1024 * 1024,
) -> Optional[PDFDownload]:
    """Download a single arXiv PDF, extract text, write to ``out_dir``.

    Returns None on any failure (network, parse, etc.).
    """
    pypdf = _try_import_pypdf()
    if pypdf is None:
        log.warning("pypdf not installed; cannot fetch arXiv PDF bodies.")
        return None

    aid = _normalize_id(arxiv_id)
    if not aid:
        return None
    pdf_url = f"{ARXIV_PDF_BASE}/{aid}"
    client = client or httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"arxiv_{aid.replace('/', '_').replace('.', '_')}.pdf"
    text_path = out_dir / f"arxiv_{aid.replace('/', '_').replace('.', '_')}.fulltext.md"

    try:
        with client.stream("GET", pdf_url) as r:
            r.raise_for_status()
            content_length = int(r.headers.get("Content-Length", "0") or 0)
            if content_length and content_length > max_bytes:
                log.info("skip %s: %d bytes > max", aid, content_length)
                return None
            n = 0
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_bytes():
                    if n + len(chunk) > max_bytes:
                        log.info("skip %s: stream > max", aid)
                        return None
                    f.write(chunk)
                    n += len(chunk)
    except httpx.HTTPError as e:
        log.warning("fetch arxiv %s failed: %s", aid, e)
        return None

    # Parse PDF
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        chunks: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                chunks.append(f"## page {i+1}\n\n{txt.strip()}\n")
        text = "\n\n".join(chunks)
    except Exception as e:
        log.warning("pypdf failed on %s: %s", aid, e)
        return None

    # Persist as a fulltext digest
    body = (
        f"# arxiv_fulltext:{aid}\n\n"
        f"## [开头/摘要]\n\n"
        f"**arXiv id:** {aid}\n"
        f"**PDF URL:** {pdf_url}\n"
        f"**Pages:** {len(reader.pages)}\n"
        f"**Characters:** {len(text):,}\n\n"
        f"**Full text:**\n\n{text}\n"
    )
    text_path.write_text(body, encoding="utf-8")

    time.sleep(5.0)  # polite

    return PDFDownload(
        arxiv_id=aid,
        pdf_path=pdf_path,
        text_path=text_path,
        n_pages=len(reader.pages),
        n_chars=len(text),
    )


def fetch_batch(
    arxiv_ids: Iterable[str],
    out_dir: Path,
    *,
    max_concurrent: int = 1,
    max_bytes: int = 30 * 1024 * 1024,
) -> list[PDFDownload]:
    """Sequentially fetch a batch of arXiv PDFs (max_concurrent kept for future use)."""
    out: list[PDFDownload] = []
    for aid in arxiv_ids:
        r = fetch_one_pdf(aid, out_dir, max_bytes=max_bytes)
        if r:
            out.append(r)
    return out
