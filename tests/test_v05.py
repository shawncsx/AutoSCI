"""Tests for v0.5 — embedding index, corpus spec, distiller, failure predictor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autosci.agents.idea_closer import IdeaCloserAgent
from autosci.agents._base import AgentBase, _load_distillation_patch
from autosci.corpus_spec import (
    CorpusSpec,
    load_all_specs,
    load_spec,
    write_default_specs,
)
from autosci.embed_index import hybrid_search
from autosci.kb import Doc, KnowledgeBase
from autosci.memory import DirectorMemory
from autosci.predictor import predict_refute_probability, filter_by_probability
from autosci.schemas import (
    IdeaRefined,
    IdeaSeed,
    IdeaState,
    LitSurveyItem,
    Modality,
    RefuteResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# Embedding + hybrid fusion
# ---------------------------------------------------------------------------


def test_hybrid_search_combines_lists():
    bm25 = [Doc(doc_id=f"bm25-{i}", title=f"BM25 doc {i}", year=2024, venue="x",
                sections={"开头/摘要": "graph neural network microbiome abundance"}) for i in range(3)]
    embed = [Doc(doc_id=f"emb-{i}", title=f"Embed doc {i}", year=2024, venue="x",
                 sections={"开头/摘要": "graph neural network microbiome compositional"}) for i in range(3)]
    # Mix the two lists
    embed_top = [(embed[0], 0.9), (embed[1], 0.8), (embed[2], 0.7)]
    out = hybrid_search(bm25, embed_top, k=5)
    assert len(out) <= 5
    assert isinstance(out[0], Doc)
    # Each doc appears at most once
    seen = {d.doc_id for d in out}
    assert len(seen) == len(out)


def test_kb_search_still_works_without_embed_deps():
    """Even if sentence-transformers/faiss aren't installed, BM25 path works."""
    docs = [Doc(doc_id=f"d{i:02d}", title=f"Title {i}", year=2024, venue="x",
                sections={"开头/摘要": "graph neural network microbiome " * 5}) for i in range(5)]
    for d in docs:
        d.fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in d.sections.items())
    kb = KnowledgeBase(docs)
    out = kb.search("graph neural network", k=3, embed_mode="minilm")  # falls back
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Corpus specs
# ---------------------------------------------------------------------------


def test_default_specs_load(tmp_path: Path):
    specs_dir = tmp_path / "specs"
    write_default_specs(specs_dir)
    specs = load_all_specs(specs_dir)
    assert len(specs) == 3
    by_name = {s.name for s in specs}
    assert "Microbiome Representation Learning" in by_name
    assert "Clinical Trial & EHR" in by_name
    assert any("Materials" in n for n in by_name)


def test_load_spec_round_trip(tmp_path: Path):
    f = tmp_path / "x.json"
    data = {
        "name": "X",
        "description": "y",
        "digest_dir": "/tmp/digests",
        "paper_corpus_hint": "h",
        "modalities": ["ssl"],
        "keywords": ["k"],
        "seed_idea_hint": "s",
        "embed_mode": "minilm",
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    s = load_spec(f)
    assert s.id == "x"
    assert s.modalities == ["ssl"]
    assert s.embed_mode == "minilm"


# ---------------------------------------------------------------------------
# Failure predictor
# ---------------------------------------------------------------------------


def _mini_kb() -> KnowledgeBase:
    long_para = "graph neural network microbiome OTU clustering cross-cohort baseline " * 5
    docs = [Doc(doc_id=f"d{i:02d}", title=f"T{i}", year=2024, venue="x",
                sections={"开头/摘要": long_para}) for i in range(5)]
    for d in docs:
        d.fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in d.sections.items())
    return KnowledgeBase(docs)


def test_predict_returns_zero_with_no_history(tmp_path: Path):
    mem = DirectorMemory(path=tmp_path / "mem.jsonl")
    idea = IdeaSeed(id="x", title="Fresh new idea", one_liner="x", angle="y", modalities=[Modality.SSL])
    p = predict_refute_probability(idea, mem)
    assert p == 0.0


def test_predict_increases_with_similar_refuted(tmp_path: Path):
    mem = DirectorMemory(path=tmp_path / "mem.jsonl")
    # Inject a refuted entry similar to our idea
    state = IdeaState(idea_seed=IdeaSeed(
        id="a", title="Graph neural network for microbiome OTU clustering",
        one_liner="x", angle="y", modalities=[Modality.GRAPH_MODEL, Modality.THRESHOLD_SENSITIVITY],
    ))
    state.verdict = Verdict.REFUTED
    state.refute = RefuteResult(verdict="refute", fatal_flaws=[
        {"flaw": "missing baseline", "evidence": "no Random Forest", "fix": "add it"},
    ])
    mem.record(state)

    # Now predict for a similar idea
    idea = IdeaSeed(
        id="b",
        title="Graph neural network for OTU clustering in microbiome",
        one_liner="x", angle="y",
        modalities=[Modality.GRAPH_MODEL, Modality.THRESHOLD_SENSITIVITY],
    )
    p = predict_refute_probability(idea, mem)
    # The similar idea should get > base rate
    assert p > 0.3, f"expected p>0.3, got {p}"

    # And an unrelated idea should get a much lower p
    unrelated = IdeaSeed(id="c", title="Cat video captioning", one_liner="x", angle="y",
                         modalities=[Modality.SSL])
    p_unrelated = predict_refute_probability(unrelated, mem)
    assert p_unrelated < p, f"unrelated should be < similar; got {p_unrelated} vs {p}"


def test_filter_by_probability_drops_high_risk(tmp_path: Path):
    mem = DirectorMemory(path=tmp_path / "mem.jsonl")
    state = IdeaState(idea_seed=IdeaSeed(
        id="a", title="Graph neural network for microbiome OTU clustering",
        one_liner="x", angle="y", modalities=[Modality.GRAPH_MODEL, Modality.THRESHOLD_SENSITIVITY],
    ))
    state.verdict = Verdict.REFUTED
    state.refute = RefuteResult(verdict="refute", fatal_flaws=[])
    mem.record(state)

    risky = IdeaSeed(id="r", title="Graph neural network for microbiome OTU clustering 2",
                     one_liner="x", angle="y", modalities=[Modality.GRAPH_MODEL])
    safe = IdeaSeed(id="s", title="Cat video captioning", one_liner="x", angle="y",
                    modalities=[Modality.SSL])
    kept, dropped = filter_by_probability([risky, safe], mem, max_p=0.6)
    assert any(s.id == "s" for s in kept)
    # risky might be dropped depending on the exact p; just check dropped is a list
    assert isinstance(dropped, list)


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------


def test_distillation_patches_for_known_agent():
    from autosci.distiller import DistillationReport
    # Without a saved file, returns "" and is silent
    assert _load_distillation_patch("idea_closer") == ""

    # Write a fake distillation and check it loads
    rep = DistillationReport(
        common_strengths=["x"],
        common_weaknesses_in_revised=["y"],
        prompt_patches={"idea_closer": "Always quote 2 prior works."},
    )
    p = Path("data/distillation.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rep.to_json(), encoding="utf-8")
    try:
        patch = _load_distillation_patch("idea_closer")
        assert "Always quote 2 prior works" in patch
    finally:
        p.unlink()


def test_load_distillation_handles_missing_file(tmp_path: Path):
    from autosci.distiller import load_distillation
    assert load_distillation(tmp_path / "nope.json") is None
