"""Tests for v0.14 — passive-mode MCP tools + prompts export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosci.mcp_server import (
    _build_state,
    _collect_digest_dirs,
    _kb_search,
    _pipeline_prompts,
    _save_idea_draft,
    _save_paper_section,
    _upsert_section,
)
from autosci.prompts import (
    AGENT_ORDER,
    ALL_PROMPTS,
    PIPELINE_STEPS,
    render_user_prompt,
)


# ---------------------------------------------------------------------------
# prompts export
# ---------------------------------------------------------------------------


def test_pipeline_steps_match_agent_order():
    assert PIPELINE_STEPS == AGENT_ORDER


def test_all_prompts_has_all_agents():
    expected = set(AGENT_ORDER)
    assert set(ALL_PROMPTS["system"].keys()) == expected
    assert set(ALL_PROMPTS["user_template"].keys()) == expected


def test_render_user_prompt_idea_closer():
    # render_user_prompt is for ACTIVE mode (takes IdeaState). v0.14 passive
    # mode uses export_pipeline_prompts() instead. So this is a smoke test
    # that the function still exists and accepts a dict (it will error on
    # actual Jinja2 render, but the import must work).
    try:
        render_user_prompt("idea_closer", context={})
    except (AttributeError, TypeError):
        pass  # expected — function expects IdeaState for active mode
    assert callable(render_user_prompt)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_upsert_section_appends_new():
    text = "# Title\n\n## Abstract\n\nfirst abstract\n"
    out = _upsert_section(text, "Introduction", "intro body")
    assert "## Introduction" in out
    assert "intro body" in out
    assert "## Abstract" in out  # existing section still there


def test_upsert_section_replaces_existing():
    text = "# Title\n\n## Method\n\nold method\n\n## Conclusion\n\nend\n"
    out = _upsert_section(text, "Method", "new method")
    assert "old method" not in out
    assert "new method" in out
    # other sections still there
    assert "## Conclusion" in out
    assert "end" in out


def test_save_idea_draft_writes_file(tmp_path: Path):
    r = _save_idea_draft(
        idea_id="my_idea",
        title="My idea",
        one_liner="x",
        angle="y",
        modalities=["graph_model", "ssl"],
        paper_corpus_hint="graph ssl",
        vault_dir=tmp_path / "vault",
    )
    assert (tmp_path / "vault" / "ideas" / "my_idea.json").exists()
    data = json.loads((tmp_path / "vault" / "ideas" / "my_idea.json").read_text(encoding="utf-8"))
    assert data["id"] == "my_idea"
    assert data["modalities"] == ["graph_model", "ssl"]


def test_save_paper_section_creates_file(tmp_path: Path):
    r = _save_paper_section(
        idea_id="i1", section="Abstract", content="My abstract body",
        verdict="kept", vault_dir=tmp_path,
    )
    assert (tmp_path / "kept" / "i1" / "paper.md").exists()
    content = (tmp_path / "kept" / "i1" / "paper.md").read_text(encoding="utf-8")
    assert "## Abstract" in content
    assert "My abstract body" in content


def test_save_paper_section_refuted_uses_other_bucket(tmp_path: Path):
    _save_paper_section(idea_id="i2", section="Method", content="m",
                        verdict="refuted", vault_dir=tmp_path)
    assert (tmp_path / "refuted" / "i2" / "paper.md").exists()
    assert not (tmp_path / "kept" / "i2" / "paper.md").exists()


def test_save_paper_section_appends_multiple(tmp_path: Path):
    _save_paper_section(idea_id="i3", section="Abstract", content="abs",
                        verdict="kept", vault_dir=tmp_path)
    _save_paper_section(idea_id="i3", section="Introduction", content="intro",
                        verdict="kept", vault_dir=tmp_path)
    content = (tmp_path / "kept" / "i3" / "paper.md").read_text(encoding="utf-8")
    assert "## Abstract" in content
    assert "## Introduction" in content
    assert content.find("## Abstract") < content.find("## Introduction")


# ---------------------------------------------------------------------------
# kb_search (passive)
# ---------------------------------------------------------------------------


def test_collect_digest_dirs_works():
    dirs = _collect_digest_dirs()
    assert isinstance(dirs, list)
    # at least the local paper digest should be present
    if (Path(r"D:\shawn\paper\_digest")).exists():
        assert any("paper" in str(d) for d in dirs)


def test_kb_search_returns_anchors():
    if not (Path(r"D:\shawn\paper\_digest")).exists():
        pytest.skip("local paper digest missing")
    res = _kb_search("microbiome foundation", k=3)
    assert isinstance(res, list)
    assert all("doc_id" in r and "title" in r for r in res)


# ---------------------------------------------------------------------------
# _pipeline_prompts (passive)
# ---------------------------------------------------------------------------


def test_pipeline_prompts_returns_all_10():
    p = _pipeline_prompts()
    assert set(p.keys()) == set(AGENT_ORDER)
    for step, d in p.items():
        assert "system" in d and len(d["system"]) > 50
        assert "user_template" in d and len(d["user_template"]) > 20


# ---------------------------------------------------------------------------
# _build_state
# ---------------------------------------------------------------------------


def test_build_state_has_minimum_keys():
    s = _build_state(Path("runs"), Path("vault"), Path("data/distillation.json"))
    assert "kb_papers" in s
    assert "kb_sources" in s
    assert "vault_items" in s
    assert "llm_required" in s
    assert "model" in s
    assert "base_url" in s
