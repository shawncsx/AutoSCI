"""Offline tests for the KnowledgeBase (no LLM, no network)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from autosci.kb import KnowledgeBase, _tokenize, parse_digest, render_docs_for_llm


DIGEST_DIR = Path(r"D:\shawn\paper\_digest")


@pytest.mark.skipif(not DIGEST_DIR.exists(), reason="paper digest dir missing")
def test_ingest_all_digests():
    kb = KnowledgeBase.from_digest_dir(DIGEST_DIR)
    assert len(kb.docs) >= 50, f"unexpectedly small KB: {len(kb.docs)}"
    # Sanity: first doc has fulltext
    assert kb.docs[0].fulltext, "docs[0] missing fulltext"


def test_tokenize_handles_chinese_punct():
    toks = _tokenize("MGM: a microbiome foundation model；用图模型替换排序")
    # expect at least ascii tokens preserved
    assert "mgm" in toks
    # Chinese characters are dropped (alphanumeric regex), but the test still
    # confirms no exceptions on unicode.


@pytest.mark.skipif(not DIGEST_DIR.exists(), reason="paper digest dir missing")
def test_search_pure_bm25_top8():
    kb = KnowledgeBase.from_digest_dir(DIGEST_DIR)
    docs = kb.search("graph neural network microbiome graph model", k=8)
    assert len(docs) == 8
    # at least one of the docs should mention 'graph' as ASCII fragment
    hits = sum(1 for d in docs if "graph" in d.fulltext.lower())
    assert hits >= 1, "BM25 didn't surface any graph-related doc"


@pytest.mark.skipif(not DIGEST_DIR.exists(), reason="paper digest dir missing")
def test_render_docs_for_llm_truncates():
    kb = KnowledgeBase.from_digest_dir(DIGEST_DIR)
    docs = kb.search("transformer foundation model", k=8)
    block = render_docs_for_llm(docs, max_chars=1800, per_doc_chars=400)
    assert isinstance(block, str)
    # Hard cap MUST be honored.
    assert len(block) <= 1800


def test_parse_digest_round_trip(tmp_path: Path):
    sample = tmp_path / "sample.digest.md"
    sample.write_text(
        "# sample\n\n## [开头/摘要]\nThis is the abstract. 2025 was great. Graph models are cool.\n",
        encoding="utf-8",
    )
    d = parse_digest(sample)
    assert d.doc_id == "sample"
    assert d.year == 2025
    assert "开头/摘要" in d.sections
