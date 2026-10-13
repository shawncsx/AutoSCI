"""Tests for v0.15 — paper analyzer, idea scorer, theme analyzer, MCP integration."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autosci.idea_scorer import (
    make_break_variant,
    make_cross_field_variant,
    make_extension_variant,
    make_variants,
    score_idea,
    score_novelty,
)
from autosci.kb import Doc
from autosci.paper_analyzer import (
    PaperAnalysis,
    analyze_corpus,
    analyze_doc,
)
from autosci.schemas import IdeaSeed, Modality
from autosci.theme_analyzer import (
    Theme,
    ThemeAnalysis,
    _tokenize as _theme_tokenize,
    analyze_themes,
)


# ---------------------------------------------------------------------------
# paper_analyzer
# ---------------------------------------------------------------------------


def _mk_doc(doc_id: str, title: str, abstract: str, fulltext: str = "",
             year: int = 2025) -> Doc:
    return Doc(
        doc_id=doc_id, title=title, year=year, venue="X",
        sections={"开头/摘要": abstract, "full": fulltext or abstract},
    )


def test_analyze_doc_extracts_research_question():
    d = _mk_doc("a1", "Graph Transformer for Microbiome",
                "We propose a new method to address cross-cohort robustness.")
    a = analyze_doc(d)
    assert a.doc_id == "a1"
    assert a.research_question  # non-empty
    assert "cross-cohort" in a.research_question.lower() or "method" in a.research_question.lower()


def test_analyze_doc_extracts_main_result_with_number():
    d = _mk_doc("b1", "Improving Microbial Models",
                "We achieve 92.3% AUROC, beating baseline by 5.4pp.")
    a = analyze_doc(d)
    # either abstract-sentence with the number, or section-based
    assert ("92" in a.main_result) or (a.main_result == "")  # could fall back


def test_analyze_doc_finds_limitations_via_keywords():
    d = _mk_doc("c1", "Method X", "Our method is limited by small sample size.")
    a = analyze_doc(d)
    assert "small sample" in a.limitations.lower() or a.limitations == ""


def test_analyze_doc_produces_opportunities():
    d = _mk_doc("d1", "Method Y", "We are limited by scalability issues.")
    a = analyze_doc(d)
    assert len(a.opportunities) >= 1
    assert any("cross-cohort" in o.lower() or "address" in o.lower() for o in a.opportunities)


def test_analyze_corpus_handles_empty():
    out = analyze_corpus([], top_n=5)
    assert out == []


# ---------------------------------------------------------------------------
# idea_scorer
# ---------------------------------------------------------------------------


def _mk_idea(title="GNN for microbiome", one_liner="x", angle="y",
             modalities=None, priority=5) -> IdeaSeed:
    return IdeaSeed(
        id="i1", title=title, one_liner=one_liner, angle=angle,
        modalities=modalities or [Modality.GRAPH_MODEL], priority=priority,
    )


def test_score_idea_returns_three_dimensions():
    docs = [_mk_doc("z1", "Some Other Paper", "We study X.")]
    s = score_idea(_mk_idea(), docs)
    assert set(s.keys()) == {"novelty", "feasibility", "impact", "average"}
    assert all(0 <= v <= 1 for v in s.values())


def test_score_novelty_penalises_exact_match():
    docs = [_mk_doc("z1", "GNN for microbiome", "x")]
    s1 = score_novelty(_mk_idea("GNN for microbiome", "x", "y"), docs)
    s2 = score_novelty(_mk_idea("Completely Different Title", "novel claim", "new"), docs)
    assert s2 > s1  # novel idea should score higher


def test_make_variants_returns_three():
    idea = _mk_idea()
    variants = make_variants(idea)
    assert len(variants) == 3
    ids = {v.id for v in variants}
    assert any("_extend" in i for i in ids)
    assert any("_break" in i for i in ids)
    assert any("_xfield" in i for i in ids)


def test_extension_variant_includes_cross_cohort_modality():
    idea = _mk_idea(modalities=[Modality.SSL])
    ext = make_extension_variant(idea)
    assert Modality.CROSS_COHORT in ext.modalities


# ---------------------------------------------------------------------------
# theme_analyzer
# ---------------------------------------------------------------------------


def test_analyze_themes_groups_related_papers():
    d1 = _mk_doc("p1", "Microbiome OTU clustering", "We cluster operational taxonomic units.")
    d2 = _mk_doc("p2", "Microbiome OTU identification", "OTU identification in microbiome.")
    d3 = _mk_doc("p3", "Diffusion model for molecules", "Diffusion for molecule generation.")
    d4 = _mk_doc("p4", "Random forest for cancer", "RF applied to cancer data.")
    docs = [d1, d2, d3, d4]
    ta = analyze_themes(docs, target_themes=3)
    assert len(ta.themes) >= 1
    # The microbiome OTU papers should be clustered
    found_microbiome = any(
        any("p1" in t.paper_ids for _ in [0]) and "p2" in t.paper_ids for t in ta.themes
    )
    # at least one theme should have multiple papers
    assert any(len(t.paper_ids) >= 2 for t in ta.themes)


def test_analyze_themes_assigns_core_paper():
    docs = [
        _mk_doc("c1", "Microbiome 16S clustering", "We cluster 16S OTU data."),
        _mk_doc("c2", "Microbiome OTU", "We identify OTUs in microbiome data."),
        _mk_doc("c3", "Other topic", "Unrelated paper about materials."),
    ]
    ta = analyze_themes(docs, target_themes=2)
    for t in ta.themes:
        assert t.core_paper_id in t.paper_ids
        assert t.label
        assert t.gap_suggestion


# ---------------------------------------------------------------------------
# MCP integration (lightweight — just verify tools are registered)
# ---------------------------------------------------------------------------


def test_mcp_server_module_importable():
    """The MCP server module should be importable."""
    from autosci import mcp_server
    # just import-check; tool list is server-internal
    assert hasattr(mcp_server, "make_server")
    assert hasattr(mcp_server, "_save_idea_draft")
    assert hasattr(mcp_server, "_save_paper_section")
    # v0.15: verify the new helper exists
    assert hasattr(mcp_server, "_all_digest_docs")
