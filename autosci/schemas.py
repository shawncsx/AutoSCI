"""AutoSCI — Pydantic schemas for the entire pipeline.

All state, all inter-agent IO, all step outputs use these schemas so the
pipeline is fully type-checked end to end. JSON serialization is done
with `model_dump_json(ensure_ascii=False, indent=2)`.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. The idea (seed + agent refinements)
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """High-level research direction. Lets us group similar ideas."""
    GRAPH_MODEL = "graph_model"
    SSL = "ssl"
    COMPOSITIONAL = "compositional"
    CROSS_COHORT = "cross_cohort"
    THRESHOLD_SENSITIVITY = "threshold_sensitivity"
    MULTIMODAL_FUSION = "multimodal_fusion"


class IdeaSeed(BaseModel):
    """A research problem we want AutoSCI to attempt."""

    id: str = Field(..., description="Short slug, e.g. 'graph_vs_rank_transformer'")
    title: str
    one_liner: str = Field(..., description="One-sentence motivation")
    angle: str = Field(..., description="Innovation angle, one paragraph")
    modalities: list[Modality] = Field(default_factory=list)
    priority: int = 5
    paper_corpus_hint: str = Field(
        default="",
        description="Free-text hint to bias KB search (terms, datasets, methods).",
    )


class IdeaRefined(BaseModel):
    """IdeaCloser output: precise claim + testable hypotheses."""

    claim: str = Field(..., description="One-paragraph crisp claim.")
    primary_hypothesis: str
    secondary_hypotheses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Literature survey
# ---------------------------------------------------------------------------


class LitSurveyItem(BaseModel):
    cite_key: str
    title: str
    year: int = 0
    venue: str = ""
    relevance: str = Field(..., description="Why this work is relevant, 1-2 sentences")
    takeaway: str = Field(..., description="What we borrow / contradict / extend")


# ---------------------------------------------------------------------------
# 3. Method design
# ---------------------------------------------------------------------------


class MethodSpec(BaseModel):
    architecture: str = Field(..., description="One paragraph describing the model.")
    algorithm: str = Field(..., description="Algorithm pseudocode in plain English / simple Python.")
    training_objective: str = Field(..., description="Self-supervised / supervised loss description.")
    novelty_bullets: list[str] = Field(..., min_length=1)
    ablations_planned: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. Experimental design
# ---------------------------------------------------------------------------


class DatasetSpec(BaseModel):
    name: str
    n_samples: str = "unknown"
    phenotype: str = ""
    use: str = "train+eval"


class BaselineSpec(BaseModel):
    name: str
    type: str = "existing"
    notes: str = ""


class AblationSpec(BaseModel):
    name: str
    toggled: str
    rationale: str = ""


class ExpSpec(BaseModel):
    datasets: list[DatasetSpec]
    baselines: list[BaselineSpec]
    ablations: list[AblationSpec]
    metrics: list[str]
    significance_test: str
    cross_cohort_protocol: str
    expected_outcome: str


# ---------------------------------------------------------------------------
# 5. Paper
# ---------------------------------------------------------------------------


class PaperDraft(BaseModel):
    title: str
    abstract: str
    keywords: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(
        default_factory=dict,
        description="Ordered sections. Auto-ordered by writer.",
    )
    references: list[str] = Field(default_factory=list, description="Cite keys, e.g. [MGM2-2026]")


# ---------------------------------------------------------------------------
# 6. Code skeleton
# ---------------------------------------------------------------------------


class CodeSketchFile(BaseModel):
    path: str
    content: str


class CodeSketch(BaseModel):
    files: list[CodeSketchFile]
    run_instructions: str


# ---------------------------------------------------------------------------
# 7. Review / Refute
# ---------------------------------------------------------------------------


class CritiqueLineItem(BaseModel):
    section: str
    anchor: str = Field(..., description="Quote or paragraph index we refer to.")
    severity: str = Field(..., description="minor | major | blocking")
    suggestion: str


class Critique(BaseModel):
    summary: str
    line_items: list[CritiqueLineItem]
    severity_counts: dict[str, int] = Field(default_factory=dict)


class RefuteResult(BaseModel):
    verdict: str = Field(..., description="keep | revise | refute")
    fatal_flaws: list[dict[str, str]] = Field(default_factory=list)
    revise_suggestions: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# 8. The full state of one research project
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    PENDING = "pending"
    KEPT = "kept"
    REVISED = "revised"
    REFUTED = "refuted"


class IdeaState(BaseModel):
    """Everything we know about an in-progress (or finished) project."""

    idea_seed: IdeaSeed
    refined: Optional[IdeaRefined] = None
    lit_survey: list[LitSurveyItem] = Field(default_factory=list)
    method_spec: Optional[MethodSpec] = None
    exp_spec: Optional[ExpSpec] = None
    paper_draft: Optional[PaperDraft] = None
    code_sketch: Optional[CodeSketch] = None
    review: Optional[Critique] = None
    refute: Optional[RefuteResult] = None
    verdict: Verdict = Verdict.PENDING
    paper_path: Optional[str] = None
    code_paths: list[str] = Field(default_factory=list)
    history_log: list[str] = Field(default_factory=list)

    # bookkeeping
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# 9. Convenience for the agents
# ---------------------------------------------------------------------------


class AgentError(BaseModel):
    """Records why an agent step failed (e.g., LLM returned bad JSON)."""

    agent: str
    error: str
    payload: Optional[dict[str, Any]] = None
    at: datetime = Field(default_factory=datetime.now)


__all__ = [
    "Modality",
    "IdeaSeed",
    "IdeaRefined",
    "LitSurveyItem",
    "MethodSpec",
    "DatasetSpec",
    "BaselineSpec",
    "AblationSpec",
    "ExpSpec",
    "PaperDraft",
    "CodeSketch",
    "CodeSketchFile",
    "Critique",
    "CritiqueLineItem",
    "RefuteResult",
    "IdeaState",
    "Verdict",
    "AgentError",
]
