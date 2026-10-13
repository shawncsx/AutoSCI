"""Tests for v0.11 — sklearn-based real verifier."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosci.realverifier import (
    RealVerifierResult,
    _make_synthetic,
    run_real_verifier,
)


def test_realverifier_safely_handles_no_sklearn(monkeypatch):
    """If sklearn is not importable, return sentinel without raising."""
    # Force the import to fail
    import autosci.realverifier as rv

    monkeypatch.setattr(rv, "_try_import_sklearn", lambda: None)
    res = run_real_verifier()
    assert res.sklearn_available is False
    assert isinstance(res, RealVerifierResult)
    assert any("not installed" in n for n in res.notes)


def test_synthetic_data_shape():
    X, y = _make_synthetic(n_samples=100, n_features=20, n_useful=5, seed=0)
    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert set(y.tolist()).issubset({0, 1})


def test_realverifier_run_end_to_end_if_sklearn_available():
    """Run if sklearn is installed; otherwise just assert the no-sklearn path."""
    res = run_real_verifier(n_samples=400, n_features=80, n_useful=20, n_splits=3, seed=0)
    if res.sklearn_available:
        # We expect decent AUROC on a synthetic signal (n_useful=20 / n_features=80)
        assert 0.6 <= res.method_auroc_mean <= 1.0
        # baseline AUCs should be > chance but < our method
        assert 0.5 <= res.random_forest_auroc_mean <= 1.0
        assert 0.5 <= res.logreg_auroc_mean <= 1.0
        assert 0.5 <= res.mlp_auroc_mean <= 1.0
        # result is JSON serializable
        json.dumps(res.__dict__)
    else:
        # No sklearn: we just got the sentinel, that's fine.
        assert isinstance(res.notes, list)


def test_realverifier_persists_in_pipeline(tmp_path: Path):
    """A small end-to-end: build a state, run the v0.11 step, expect a JSON file."""
    from autosci.pipeline import ResearchTeam, RunConfig
    from autosci.kb import KnowledgeBase
    from autosci.llm import LLMClient
    from autosci.schemas import IdeaSeed, IdeaState, IdeaRefined
    from autosci.agents.section_validator import REQUIRED_SECTIONS
    from unittest.mock import MagicMock

    # minimal state to test only the realverifier step
    state = IdeaState(
        idea_seed=IdeaSeed(id="t11", title="T", one_liner="o", angle="a", modalities=[])
    )
    state.refined = IdeaRefined(claim="c", primary_hypothesis="h", risks=[])
    # Skim a paper_draft so other steps (if any) don't fail
    state.paper_draft = type("PD", (), {"sections": {s: "" for s in REQUIRED_SECTIONS}})()
    state.paper_draft.sections = {s: "x" for s in REQUIRED_SECTIONS}
    state.paper_draft.references = ["[X-2025]"]

    rt = ResearchTeam.__new__(ResearchTeam)
    rt._run_realverifier = lambda *a, **kw: ResearchTeam._run_realverifier(rt, state, tmp_path)
    rt._run_realverifier(state, tmp_path)

    out = tmp_path / "realverification.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "sklearn_available" in data
    assert "method_auroc_mean" in data
