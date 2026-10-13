"""Tests for v0.4 — citation grounder, section validator, arxiv ingester, idea templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autosci.agents.citation_grounder import (
    CitationGrounderAgent,
    extract_bracket_keys,
    extract_cite_keys,
)
from autosci.agents.section_validator import _word_count, REQUIRED_SECTIONS
from autosci.idea_template import IdeaTemplate, load_templates, propose_from_template
from autosci.kb import Doc, KnowledgeBase
from autosci.schemas import (
    IdeaSeed,
    IdeaState,
    LitSurveyItem,
    PaperDraft,
)


# ---------------------------------------------------------------------------
# Citation grounder
# ---------------------------------------------------------------------------


def _mk_state_with_paper(known_keys: list[str], used_keys: list[str]) -> IdeaState:
    state = IdeaState(idea_seed=IdeaSeed(id="x", title="t", one_liner="l", angle="a", modalities=[]))
    state.lit_survey = [
        LitSurveyItem(cite_key=f"[{k}]", title=k, year=2024, venue="x", relevance="r", takeaway="t")
        for k in known_keys
    ]
    sections: dict[str, str] = {}
    for k in used_keys:
        sections[f"intro_{k}"] = f"This section mentions [{k}] for context."
    state.paper_draft = PaperDraft(
        title="t", abstract="a", sections=sections, references=[],
    )
    return state


class _StubLLM:
    def with_timeout(self, name): self.timeout_name = name
    def reset_timeout(self): pass
    def chat_json(self, **kwargs):
        # Pretend the LLM rescued any missing key
        return {"items": [
            {"cite_key": "[RESCUED-X]", "title": "Rescued", "year": 2024, "venue": "arxiv",
             "relevance": "r", "takeaway": "t"},
        ]}


def test_extract_brackets():
    s = "Hello [Foo-2024] and [Bar_2025], and [Foo-2024] again."
    assert extract_bracket_keys(s) == ["[Bar_2025]", "[Foo-2024]"]


def test_citation_grounder_marks_known_keys_verified():
    state = _mk_state_with_paper(["Foo-2024", "Bar-2025"], ["Foo-2024", "Bar-2025"])
    agent = CitationGrounderAgent(_StubLLM())  # type: ignore[arg-type]
    rep = agent.run(state)
    assert rep["n_verified"] == 2
    assert rep["n_missing_before"] == 0


def test_citation_grounder_rescues_missing_keys():
    state = _mk_state_with_paper(["Foo-2024"], ["Foo-2024", "Missing-2024"])
    agent = CitationGrounderAgent(_StubLLM())  # type: ignore[arg-type]
    rep = agent.run(state)
    assert "[Missing-2024]" in rep["missing"]
    assert len(rep["added_keys"]) >= 1
    # missing key should now be in references
    assert "[Missing-2024]" in state.paper_draft.references
    # and in lit_survey
    assert any(it.cite_key == "[RESCUED-X]" for it in state.lit_survey)


# ---------------------------------------------------------------------------
# Section validator
# ---------------------------------------------------------------------------


def test_word_count_cjk_and_latin():
    # 4 Chinese chars + 2 Latin words (Greek letters fall outside both regexes
    # so we don't test those here).
    assert _word_count("中文测试 hello world") == 6


def test_required_sections_count():
    assert len(REQUIRED_SECTIONS) == 8


class _FakeSecLLM:
    def __init__(self) -> None:
        self.timeout_name = None
    def with_timeout(self, name): self.timeout_name = name
    def reset_timeout(self): pass
    def chat_json(self, *, system, user, temperature, max_tokens=None):
        # v0.4 batched: produce per-section dict based on which sections appear
        # in the user prompt. Sections with "PASS_OK" in body get quality=4.
        out: dict[str, dict] = {}
        # Detect sections from the user prompt marker `## <name>`
        import re
        for s in re.findall(r"^## (\w+)\s", user, flags=re.M):
            # Find the body chunk
            m = re.search(rf"## {s}\n(.*?)(?=\n## |\Z)", user, flags=re.S)
            body = m.group(1) if m else ""
            if "PASS_OK" in body:
                out[s] = {"covers_hypothesis": True, "cites_papers": True, "quality": 4, "suggestion": "ok"}
            elif "(empty)" in body or not body.strip():
                out[s] = {"covers_hypothesis": False, "cites_papers": False, "quality": 1, "suggestion": "missing"}
            else:
                out[s] = {"covers_hypothesis": False, "cites_papers": False, "quality": 1, "suggestion": "rewrite"}
        return out


def test_section_validator_marks_missing():
    state = IdeaState(idea_seed=IdeaSeed(id="x", title="t", one_liner="l", angle="a", modalities=[]))
    state.paper_draft = PaperDraft(
        title="t", abstract="a",
        sections={"introduction": "PASS_OK hello world"},  # only intro has content
        references=[],
    )
    from autosci.agents.section_validator import SectionValidatorAgent
    agent = SectionValidatorAgent(_FakeSecLLM())  # type: ignore[arg-type]
    rep = agent.run(state)
    # 1 section passed (intro with PASS_OK), 7 missing → all fail
    assert rep["n_passed"] == 1
    assert rep["n_failed"] == 7
    assert rep["needs_revision"] is True
    assert rep["details"]["introduction"]["verdict"] == "ok"
    assert rep["details"]["method"]["verdict"] == "missing"


# ---------------------------------------------------------------------------
# Idea template
# ---------------------------------------------------------------------------


def test_load_templates(tmp_path: Path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({
        "templates": [
            {"id": "x", "name": "X", "method_anchor": "GAT", "modalities": ["graph_model"]},
        ]
    }), encoding="utf-8")
    out = load_templates(f)
    assert len(out) == 1
    assert out[0].id == "x"
    assert "GAT" in out[0].render_prompt_block()


def test_propose_from_template_with_fake_llm():
    class _TmplLLM:
        def with_timeout(self, name): pass
        def reset_timeout(self): pass
        def chat_json(self, *, system, user, temperature, max_tokens=None):
            return {"ideas": [
                {"id": "x01", "title": "GAT for CRC", "one_liner": "x", "angle": "y",
                 "modalities": ["graph_model", "threshold_sensitivity"], "priority": 8,
                 "paper_corpus_hint": "graph crc"},
            ]}

    tpl = IdeaTemplate(id="t1", name="t1", method_anchor="GAT", modalities=["graph_model"])
    kb = _tiny_kb()
    out = propose_from_template(_TmplLLM(), kb, tpl, n=1)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0].title == "GAT for CRC"
    # If LLM didn't supply modalities, we fall back to template's
    assert "graph_model" in [m.value for m in out[0].modalities]


def _tiny_kb() -> KnowledgeBase:
    docs = [Doc(doc_id=f"d{i:02d}", title=f"T{i}", year=2024, venue="x",
                sections={"开头/摘要": "graph neural network microbiome " * 10}) for i in range(5)]
    for d in docs:
        d.fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in d.sections.items())
    return KnowledgeBase(docs)
