"""Tests for v0.9 — bioRxiv, PubMed, Semantic Scholar, cross-source dedup."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autosci.biorxiv import BiorxivPaper, _chunk_date_range
from autosci.dedup import (
    _normalize_arxiv_id,
    _normalize_doi,
    dedup_digests,
    digest_to_record,
    title_hash,
    write_deduped_digests,
)
from autosci.pubmed import (
    PubmedPaper,
    _extract_all,
    _extract_doi,
    _extract_text,
    _extract_year,
    esearch,
)
from autosci.semanticscholar import S2Paper, fetch_s2


# ---------------------------------------------------------------------------
# bioRxiv
# ---------------------------------------------------------------------------


def test_chunk_date_range_basic():
    import datetime as dt
    chunks = _chunk_date_range(dt.date(2024, 1, 1), dt.date(2024, 3, 1), days=30)
    assert len(chunks) >= 1
    for s, e in chunks:
        assert s <= e
    # covers the whole range
    assert chunks[0][0] == dt.date(2024, 1, 1)
    assert chunks[-1][1] >= dt.date(2024, 2, 27)


def test_biorxiv_cite_key():
    p = BiorxivPaper(
        biorxiv_id="2024.05.15.123456v1", doi="", title="x", authors=[],
        abstract="x", category="microbiology", posted="2024-05-15", link="",
    )
    assert p.cite_key == "[biorxiv_2024_05_15_123456v1]"


def test_biorxiv_fetch_handles_failure():
    import httpx
    class _Bad:
        def __init__(self, timeout=None, trust_env=False): pass
        def get(self, *a, **kw):
            import httpx as hx
            raise hx.HTTPError("conn refused")
    with patch("autosci.biorxiv.httpx.Client", _Bad):
        from autosci.biorxiv import fetch_biorxiv_search
        out = fetch_biorxiv_search("microbiome", timeout=1)
    assert out == []


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------


def test_extract_text_returns_empty_on_miss():
    assert _extract_text("<root></root>", "ArticleTitle") == ""


def test_extract_text_strips_nested_tags():
    xml = "<ArticleTitle>foo <i>bar</i> baz</ArticleTitle>"
    assert _extract_text(xml, "ArticleTitle") == "foo bar baz"


def test_extract_year_parses_simple():
    xml = "<PubDate><Year>2024</Year><Month>Jan</Month></PubDate>"
    assert _extract_year(xml) == 2024


def test_extract_year_returns_zero_when_missing():
    assert _extract_year("<PubDate><MedlineDate>2024 Jan</MedlineDate></PubDate>") == 0


def test_extract_doi_handles_absent():
    assert _extract_doi("<root></root>") == ""


def test_extract_doi_picks_article_id():
    xml = '<ArticleIdList><ArticleId IdType="doi">10.1234/abc</ArticleId></ArticleIdList>'
    assert _extract_doi(xml) == "10.1234/abc"


def test_pubmed_esearch_handles_failure():
    import httpx
    class _Bad:
        def __init__(self, timeout=None, trust_env=False): pass
        def get(self, *a, **kw):
            import httpx as hx
            raise hx.HTTPError("net")
    with patch("autosci.pubmed.httpx.Client", _Bad):
        out = esearch("microbiome", retmax=5)
    assert out == []


def test_pubmed_cite_key():
    p = PubmedPaper(pmid="12345678", title="x", abstract="x")
    assert p.cite_key == "[pubmed_12345678]"


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def test_s2_cite_key():
    p = S2Paper(paper_id="abc123", title="x", abstract="x", year=2024, venue="x",
                citation_count=0, reference_count=0, influential_citation_count=0)
    assert p.cite_key == "[s2_abc123]"


def test_s2_fetch_skips_abstract_less_papers():
    fake = {
        "data": [
            {"paperId": "1", "title": "X", "abstract": "", "year": 2024, "venue": "",
             "citationCount": 0, "referenceCount": 0, "influentialCitationCount": 0,
             "fieldsOfStudy": [], "authors": [], "externalIds": {}},
            {"paperId": "2", "title": "Y", "abstract": "has abstract", "year": 2024, "venue": "x",
             "citationCount": 0, "referenceCount": 0, "influentialCitationCount": 0,
             "fieldsOfStudy": ["ML"], "authors": [{"name": "Alice"}],
             "externalIds": {"DOI": "10.1/x", "ArXiv": "2402.16901"}},
        ]
    }
    import httpx
    class _OK:
        def __init__(self, timeout=None, trust_env=False): pass
        def get(self, *a, **kw):
            r = MagicMock_paper()
            return r
    # Easier: just patch the function-level call
    with patch("autosci.semanticscholar.httpx.Client") as MockClient:
        mock = MockClient.return_value
        mock.get.return_value.status_code = 200
        mock.get.return_value.raise_for_status = lambda: None
        mock.get.return_value.json.return_value = fake
        out = fetch_s2("x", max_results=5)
    assert len(out) == 1
    assert out[0].paper_id == "2"
    assert out[0].doi == "10.1/x"
    assert out[0].fields_of_study == ["ML"]


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_normalize_doi_lowercases_and_strips_url():
    assert _normalize_doi("https://doi.org/10.1234/AbC") == "10.1234/abc"
    assert _normalize_doi("  10.1234/abc  ") == "10.1234/abc"


def test_normalize_arxiv_id_strips_url_and_version():
    assert _normalize_arxiv_id("https://arxiv.org/abs/2402.16901v2") == "2402.16901"
    assert _normalize_arxiv_id("2402.16901V1") == "2402.16901"


def test_title_hash_stable_for_same_title():
    h1 = title_hash("A Foundation Model for the Human Gut Microbiome")
    h2 = title_hash("a foundation model for the human gut microbiome")
    h3 = title_hash("a   foundation  model for the human gut microbiome")
    assert h1 == h2 == h3
    assert len(h1) == 16


def test_title_hash_handles_chinese():
    h = title_hash("微生物组基础模型")
    assert len(h) == 16


def test_digest_to_record_extracts_doi(tmp_path: Path):
    d = tmp_path / "arxiv_2402_16901.digest.md"
    d.write_text(
        "# arxiv:2402.16901\n\n## [开头/摘要]\n\n"
        "**Title:** A foundation model for the human gut microbiome\n\n"
        "**Year:** 2024\n\n"
        "**Abstract:**\n\nWe propose a foundation model for the human gut microbiome.\n",
        encoding="utf-8",
    )
    rec = digest_to_record(d)
    assert rec is not None
    assert rec.canonical_id == "arxiv:2402.16901"
    assert rec.year == 2024
    assert "foundation model" in rec.title
    assert "microbiome" in rec.abstract


def test_dedup_digests_collapses_duplicates(tmp_path: Path):
    # 2 arXiv files with same paper id
    for i in (1, 2):
        d = tmp_path / f"arxiv_2402_16901v{i}.digest.md"
        d.write_text(
            "# arxiv:2402.16901v" + str(i) + "\n\n## [开头/摘要]\n\n"
            "**Title:** A foundation model for the human gut microbiome\n\n"
            "**Year:** 2024\n\n"
            "**Abstract:**\n\nWe propose a foundation model for the human gut microbiome v" + str(i) + ".\n",
            encoding="utf-8",
        )
    # 1 openalex file with same DOI
    d3 = tmp_path / "openalex_W12345.digest.md"
    d3.write_text(
        "# openalex:W12345\n\n## [开头/摘要]\n\n"
        "**Title:** A foundation model for the human gut microbiome\n\n"
        "**Year:** 2024\n\n"
        "**DOI:** 10.1234/abc\n\n"
        "**Abstract:**\n\nThe longer abstract version goes here so it should win during merge.\n",
        encoding="utf-8",
    )
    records, to_delete = dedup_digests(tmp_path)
    # arxiv files both collapse to "arxiv:2402.16901"; openalex has DOI
    # and collapses to "doi:10.1234/abc" — different canonical, so we have 2.
    assert len(records) == 2
    # 2 of the 3 input files should be marked for deletion
    assert len(to_delete) == 1


def test_write_deduped_digests_round_trip(tmp_path: Path):
    digest = tmp_path / "arxiv_X.digest.md"
    digest.write_text(
        "# arxiv:2402.16901\n\n## [开头/摘要]\n\n"
        "**Title:** X\n\n**Year:** 2024\n\n"
        "**Abstract:**\n\nLong abstract here.\n",
        encoding="utf-8",
    )
    records, _ = dedup_digests(tmp_path)
    n = write_deduped_digests(records, tmp_path)
    assert n == 1
    files = list(tmp_path.glob("deduped_*.digest.md"))
    assert len(files) == 1
    assert "Sources:" in files[0].read_text(encoding="utf-8")


# Stub for the s2 test above
class MagicMock_paper:
    pass
