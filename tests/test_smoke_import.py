"""Lightweight offline tests covering import + schemas round-trip.

These do NOT make any network call. They give us a quick sanity check.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from autosci.schemas import (
    IdeaSeed,
    IdeaRefined,
    IdeaState,
    Modality,
    Verdict,
    MethodSpec,
    ExpSpec,
    PaperDraft,
    CodeSketch,
    Critique,
    RefuteResult,
    LitSurveyItem,
    DatasetSpec,
    BaselineSpec,
    AblationSpec,
    CritiqueLineItem,
)


def test_imports():
    assert IdeaSeed.__name__ == "IdeaSeed"


def test_idea_seed_round_trip():
    seed = IdeaSeed(
        id="x",
        title="X vs Y",
        one_liner="X is better than Y at Z",
        angle="because we add W",
        modalities=[Modality.GRAPH_MODEL],
        priority=7,
    )
    payload = json.loads(seed.model_dump_json())
    assert payload["title"] == "X vs Y"
    assert payload["modalities"] == ["graph_model"]


def test_idea_state_full_round_trip():
    state = IdeaState(idea_seed=IdeaSeed(id="z", title="z", one_liner="z", angle="z"))
    state.refined = IdeaRefined(claim="c", primary_hypothesis="h")
    state.lit_survey = [
        LitSurveyItem(cite_key="[X-2026]", title="X", year=2026, relevance="r", takeaway="t")
    ]
    state.method_spec = MethodSpec(
        architecture="arch",
        algorithm="algo",
        training_objective="obj",
        novelty_bullets=["n1"],
        ablations_planned=["a1"],
    )
    state.exp_spec = ExpSpec(
        datasets=[DatasetSpec(name="d1", n_samples="100", phenotype="p1", use="train")],
        baselines=[BaselineSpec(name="b1", type="existing", notes="n")],
        ablations=[AblationSpec(name="a1", toggled="t", rationale="r")],
        metrics=["AUROC"],
        significance_test="Wilcoxon",
        cross_cohort_protocol="leave-one-cohort-out",
        expected_outcome="+5pp",
    )
    state.paper_draft = PaperDraft(
        title="t", abstract="a", keywords=["k"], sections={"intro": "i"}, references=["[X-2026]"]
    )
    state.review = Critique(
        summary="s",
        line_items=[CritiqueLineItem(section="intro", anchor="abc", severity="minor", suggestion="s")],
    )
    state.refute = RefuteResult(verdict="keep", fatal_flaws=[], revise_suggestions=[])
    state.verdict = Verdict.KEPT
    state.history_log.append("ok")
    j = state.model_dump_json()
    state2 = IdeaState.model_validate_json(j)
    assert state2.verdict == Verdict.KEPT
    assert state2.method_spec is not None
    assert state2.method_spec.novelty_bullets == ["n1"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
