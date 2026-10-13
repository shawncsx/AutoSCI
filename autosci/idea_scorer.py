"""AutoSCI — Idea scoring + variant generation (v0.15).

For any IdeaSeed, compute a 3-dimension score:

    - novelty      (0-1): how new vs the local corpus
    - feasibility  (0-1): how plausible to implement with current data + methods
    - impact       (0-1): how much the field would benefit

Each dimension is computed via:
    1. Cheap lexical heuristics on the idea's own text
    2. A lookup against existing corpus titles for the novelty axis

Plus, generate **3 variants** of the idea:
    - extension     (build on the idea's own work)
    - break         (challenge one of its assumptions)
    - cross-field   (apply its method to a different field)

All deterministic, no LLM call. Fast (sub-second per idea).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from autosci.kb import Doc
from autosci.schemas import IdeaSeed, Modality


log = logging.getLogger("autosci.idea_scorer")


def _tokenize(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[\s\W_]+", " ", s, flags=re.UNICODE)
    return {t for t in s.split() if len(t) >= 4}


def _corpus_titles(docs: list[Doc]) -> list[str]:
    return [d.title for d in docs if d.title]


def _corpus_titles_set(docs: list[Doc]) -> set[str]:
    return {t.lower() for t in _corpus_titles(docs)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_novelty(idea: IdeaSeed, docs: list[Doc]) -> float:
    """Heuristic: low if idea's main nouns already appear in corpus titles."""
    idea_tokens = _tokenize(idea.title + " " + (idea.one_liner or "") + " " + (idea.angle or ""))
    if not idea_tokens:
        return 0.5
    # 1. exact-title-substring hit
    title_lower = " ".join(t.lower() for t in _corpus_titles(docs))
    title_set = _corpus_titles_set(docs)
    title_l = title_lower
    novelty = 1.0
    # full-substring hit → 0.1
    if any(t in title_l for t in idea_tokens if len(t) > 5):
        novelty -= 0.3
    # strong token overlap → reduce
    common = sum(1 for t in idea_tokens if t in title_l)
    novelty -= min(0.5, common / max(1, len(idea_tokens)))
    novelty -= len(title_set.intersection({idea.title.lower()})) * 0.5
    return round(max(0.0, min(1.0, novelty)), 3)


def score_feasibility(idea: IdeaSeed) -> float:
    """Heuristic: high if modalities align with common bio ML; penalise vague 'all'."""
    score = 0.6
    mod = set(idea.modalities or [])
    well_supported = {Modality.GRAPH_MODEL, Modality.SSL, Modality.COMPOSITIONAL,
                      Modality.CROSS_COHORT, Modality.MULTIMODAL_FUSION}
    if mod & well_supported:
        score += 0.2
    if Modality.THRESHOLD_SENSITIVITY in mod:
        score += 0.1
    if not mod:
        score -= 0.2
    if idea.priority and idea.priority >= 8:
        score += 0.05
    # Penalise giant corpus_hint (likely unfocused)
    hint_words = (idea.paper_corpus_hint or "").split()
    if len(hint_words) > 8:
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 3)


def score_impact(idea: IdeaSeed, docs: list[Doc]) -> float:
    """Heuristic: high if a corpus modality group is underserved."""
    mod = set(idea.modalities or [])
    # if the corpus has many docs in same modality, the new idea's marginal
    # impact is lower (the area is already crowded)
    counts = Counter()
    for d in docs:
        text = (d.fulltext or d.title or "").lower()
        if "graph" in text or "gat" in text or "gnn" in text:
            counts["graph"] += 1
        if "ssl" in text or "self-supervised" in text or "pretrain" in text:
            counts["ssl"] += 1
        if "compositional" in text or "dirichlet" in text or "clr" in text:
            counts["compositional"] += 1
        if "leave-one-cohort" in text or "cross-cohort" in text:
            counts["cross_cohort"] += 1
        if "threshold" in text and ("97" in text or "99" in text):
            counts["threshold"] += 1
    impact = 0.5
    for m in mod:
        impact += 0.15 if counts.get(m.value, 0) < 5 else 0.0  # under-served
    if idea.priority and idea.priority >= 8:
        impact += 0.1
    return round(max(0.0, min(1.0, impact)), 3)


def score_idea(idea: IdeaSeed, docs: list[Doc]) -> dict:
    return {
        "novelty":     score_novelty(idea, docs),
        "feasibility": score_feasibility(idea),
        "impact":      score_impact(idea, docs),
        "average":     round((
            score_novelty(idea, docs) + score_feasibility(idea) + score_impact(idea, docs)
        ) / 3, 3),
    }


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def make_extension_variant(idea: IdeaSeed) -> IdeaSeed:
    """Same direction, just one step further. Add scope / scale / modality."""
    new_mod = list(idea.modalities or [])
    if Modality.CROSS_COHORT not in new_mod:
        new_mod.append(Modality.CROSS_COHORT)
    return IdeaSeed(
        id=f"{idea.id}_extend",
        title=f"Scale-up of {idea.title[:60]}: cross-cohort evaluation",
        one_liner=(idea.one_liner or "") + " across 5+ cohorts",
        angle=f"This work extends the paper's evaluation to N additional cohorts. "
              f"While {idea.title[:40]!r} validated on a single cohort, the field needs "
              f"cross-cohort evidence before clinical adoption.",
        modalities=new_mod,
        priority=idea.priority,
        paper_corpus_hint=(idea.paper_corpus_hint or "") + " cross-cohort",
    )


def make_break_variant(idea: IdeaSeed) -> IdeaSeed:
    """Challenge a strong assumption. Look for the word 'always' / 'invariant' etc."""
    neg = ""
    if idea.angle and "permutation" in idea.angle.lower():
        neg = "We do not assume permutation-invariance. The ordering of taxa in real samples carries biological information (e.g. phylogeny, ecological succession). "
    elif idea.angle and "self-supervised" in idea.angle.lower():
        neg = "We do not assume self-supervised pre-training transfers. The pretext task may distort representation in ways that hurt downstream tasks. "
    elif idea.angle and "graph" in idea.angle.lower():
        neg = "We do not assume the chosen graph construction is the right one. Multiple graph structures are biologically plausible; we'll ablate. "
    else:
        neg = "We do not assume the paper's framing is correct. The headline claim may be a methodological artefact. "
    return IdeaSeed(
        id=f"{idea.id}_break",
        title=f"Negative result: {idea.title[:50]} fails on [chosen stress test]",
        one_liner=(idea.one_liner or "") + " — but here's where it breaks",
        angle=neg + " This work runs a controlled stress test designed to falsify the paper's headline claim, and reports where the claim survives and where it doesn't.",
        modalities=list(idea.modalities or []) + [Modality.CROSS_COHORT],
        priority=max(1, (idea.priority or 5) - 1),
        paper_corpus_hint=(idea.paper_corpus_hint or "") + " negative result falsification",
    )


def make_cross_field_variant(idea: IdeaSeed) -> IdeaSeed:
    """Apply the method to a different field."""
    if any(k in (idea.angle or "").lower() for k in ("microbi", "metagen", "16s", "otu")):
        cross = "materials science / MOFs / catalysis"
    elif "patient" in (idea.angle or "").lower() or "clinical" in (idea.angle or "").lower():
        cross = "metabolomics / proteomics"
    else:
        cross = "NLP / recommender systems"
    return IdeaSeed(
        id=f"{idea.id}_xfield",
        title=f"{idea.title[:50]} applied to {cross}",
        one_liner=f"Reuse the paper's method, but on {cross} data",
        angle=f"This work ports the method in {idea.title[:30]!r} to {cross}, "
              f"where the data is fundamentally different. We expect new failure modes "
              f"and new wins, both informative for the original method's design.",
        modalities=list(idea.modalities or []),
        priority=idea.priority,
        paper_corpus_hint=f"{cross} domain transfer",
    )


def make_variants(idea: IdeaSeed) -> list[IdeaSeed]:
    return [make_extension_variant(idea),
            make_break_variant(idea),
            make_cross_field_variant(idea)]
