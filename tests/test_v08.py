"""Tests for v0.8 — mass arXiv, OpenAlex, wide idea miner, run_mass."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from autosci.agents.idea_miner import (
    GAP_KEYWORDS,
    WIDE_ANGLES,
    dedupe_by_title,
    propose_ideas,
    propose_wide_ideas,
)
from autosci.kb import Doc, KnowledgeBase
from autosci.mass_ingestor import (
    DEFAULT_QUERY_BANK,
    fetch_one_query,
    run_mass,
)
from autosci.openalex import (
    _reconstruct_abstract,
    OpenAlexWork,
    fetch_openalex,
)
from autosci.schemas import IdeaSeed, Modality


# ---------------------------------------------------------------------------
# Mass ingester
# ---------------------------------------------------------------------------


def test_default_query_bank_is_substantial():
    assert len(DEFAULT_QUERY_BANK) >= 30


def test_fetch_one_query_handles_failure(monkeypatch):
    """fetch_one_query must not raise; failures return count=0."""
    # Patch httpx.Client used inside the function
    import httpx
    class _Bad:
        def __init__(self, timeout=None, trust_env=False): pass
        def get(self, *args, **kwargs):
            import httpx as hx
            raise hx.HTTPError("connection refused")
    with patch("autosci.mass_ingestor.httpx.Client", _Bad):
        q, n, entries = fetch_one_query("anything", 10, timeout=1)
    assert n == 0
    assert entries == []


def test_run_mass_aggregates_results(tmp_path: Path):
    """run_mass with mocked fetcher yields digests and stats."""
    from autosci.arxiv_ingester import ArxivEntry
    fake_entries = [
        ArxivEntry(arxiv_id=f"2601.{i:05d}", title=f"Paper {i}", summary=f"Abstract {i}",
                   authors=["Alice"], year=2026, categories=["q-bio"],
                   link=f"https://arxiv.org/abs/2601.{i:05d}")
        for i in range(5)
    ]
    with patch("autosci.mass_ingestor.fetch_one_query",
               return_value=("fake query", 5, [e.__dict__ | {"_query": "fake"} for e in fake_entries])):
        stats = run_mass(["fake query"] * 3, out_dir=tmp_path / "out", per_query=5, max_workers=2,
                         max_total_seconds=30)
    assert stats["n_unique_papers"] == 5
    assert stats["n_queries_succeeded"] == 3
    # 5 unique digests written
    assert len(list((tmp_path / "out").glob("*.digest.md"))) == 5


def test_run_mass_dedupes_across_queries(tmp_path: Path):
    from autosci.arxiv_ingester import ArxivEntry
    fake = [ArxivEntry(arxiv_id="2601.00001", title="X", summary="X", authors=[],
                       year=2026, categories=[], link="x")]
    with patch("autosci.mass_ingestor.fetch_one_query",
               return_value=("q", 1, [fake[0].__dict__ | {"_query": "q"}])):
        stats = run_mass(["q1", "q2", "q3"], out_dir=tmp_path / "out", per_query=1, max_workers=2,
                         max_total_seconds=30)
    assert stats["n_unique_papers"] == 1  # dedup


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def test_reconstruct_abstract_handles_empty():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


def test_reconstruct_abstract_orders_by_position():
    inv = {"foo": [3, 7], "bar": [1, 5]}
    assert _reconstruct_abstract(inv) == "bar foo bar foo"


def test_openalex_cite_key_strips_url():
    w = OpenAlexWork(openalex_id="https://openalex.org/W12345", doi="", title="",
                    abstract="x", year=2024, authors=[], venue="", cited_by_count=0)
    assert w.cite_key == "[openalex_W12345]"


def test_fetch_openalex_handles_failure():
    import httpx
    class _Bad:
        def __init__(self, timeout=None, trust_env=False): pass
        def get(self, *args, **kwargs):
            import httpx as hx
            raise hx.HTTPError("conn refused")
    with patch("autosci.openalex.httpx.Client", _Bad):
        out = fetch_openalex("x", max_results=5, timeout=1)
    assert out == []


# ---------------------------------------------------------------------------
# Wide IdeaMiner
# ---------------------------------------------------------------------------


def test_wide_angles_is_substantial():
    assert len(WIDE_ANGLES) >= 5


def test_propose_wide_ideas_returns_many_unique(monkeypatch):
    class _LLM:
        def with_timeout(self, name): pass
        def reset_timeout(self): pass
        def chat_json(self, *, system, user, temperature, max_tokens=None):
            return {"ideas": [
                {"id": f"x{i:02d}", "title": f"Title {i}", "one_liner": "x", "angle": "y",
                 "modalities": ["ssl"], "priority": 5, "paper_corpus_hint": ""}
                for i in range(5)
            ]}
    kb = _mini_kb()
    ideas = propose_wide_ideas(kb, _LLM(), total_n=15)
    # 3 calls × 5 ideas each = up to 15, deduped by title
    assert len(ideas) >= 5
    # each idea has the wide-idea signature
    for s in ideas:
        assert s.id
        assert s.title


def test_propose_wide_ideas_topoff(monkeypatch):
    """If angles produce < total_n, we should top up."""
    counter = {"n": 0}
    class _LLM:
        def with_timeout(self, name): pass
        def reset_timeout(self): pass
        def chat_json(self, *, system, user, temperature, max_tokens=None):
            counter["n"] += 1
            # return 2 ideas per call, with unique titles each time
            base = (counter["n"] - 1) * 2
            return {"ideas": [
                {"id": f"a{base + i}", "title": f"Title {base + i}",
                 "one_liner": "x", "angle": "y", "modalities": ["ssl"],
                 "priority": 5, "paper_corpus_hint": ""}
                for i in range(2)
            ]}
    kb = _mini_kb()
    ideas = propose_wide_ideas(kb, _LLM(), total_n=20)
    # 8 angles × 2 ideas = 16, then one top-off call (also unique) → 18
    assert len(ideas) >= 16


def _mini_kb() -> KnowledgeBase:
    docs = [Doc(doc_id=f"d{i:02d}", title=f"T{i}", year=2024, venue="x",
                sections={"开头/摘要": "graph neural network microbiome OTU " * 10}) for i in range(5)]
    for d in docs:
        d.fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in d.sections.items())
    return KnowledgeBase(docs)
