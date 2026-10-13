"""AutoSCI — FailurePredictor (v0.5).

Given a candidate IdeaSeed and the DirectorMemory of past attempts,
estimate the probability of the idea being refuted.

Simple v0.5 heuristic:
    p_refute = 0.0
    for each refuted entry in memory:
        + title_similarity * 0.3
        + modality_overlap * 0.4
        + shared_refuted_keyword_count * 0.2
    clamp [0, 0.9]
    if memory has < 3 refuted entries, fall back to base rate
        (refuted / total) with a 0.5 floor.

The user / Director can decide what threshold to use as a gate
(default 0.6 — refuse to start anything > 0.6 probability).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from autosci.memory import DirectorMemory
from autosci.schemas import IdeaSeed


log = logging.getLogger("autosci.predictor")


def _tokens(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[\s\W_]+", " ", s, flags=re.UNICODE)
    return {t for t in s.split() if len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def predict_refute_probability(idea: IdeaSeed, memory: DirectorMemory) -> float:
    """Return a probability in [0, 0.9]."""
    refuted = [e for e in memory.entries if e.verdict == "refuted"]
    if not refuted:
        return 0.0
    base_rate = min(0.5, sum(1 for e in memory.entries if e.verdict == "refuted") / max(1, len(memory.entries)))

    idea_tokens = _tokens(idea.title + " " + (idea.one_liner or "") + " " + (idea.angle or ""))
    idea_modalities = set(idea.modalities or [])

    score = 0.0
    for e in refuted:
        # 1. title similarity 0..0.3
        entry_tokens = _tokens(e.title)
        sim = _jaccard(idea_tokens, entry_tokens)
        score += sim * 0.3
        # 2. modality overlap 0..0.4
        if idea_modalities and set(e.modalities):
            mod_overlap = len(idea_modalities & set(e.modalities)) / len(idea_modalities | set(e.modalities))
            score += mod_overlap * 0.4
        # 3. shared refuted keywords 0..0.2 (cap at 5 shared)
        shared = sum(1 for kw in e.refuted_keywords if kw in idea_tokens)
        score += min(0.2, shared * 0.04)

    # Normalize across refuted count
    avg = score / max(1, len(refuted))
    # Blend with base rate
    p = 0.4 * base_rate + 0.6 * min(0.9, avg)
    return round(p, 4)


def filter_by_probability(
    ideas: list[IdeaSeed],
    memory: DirectorMemory,
    max_p: float = 0.6,
) -> tuple[list[IdeaSeed], list[tuple[IdeaSeed, float]]]:
    """Drop ideas whose refute-probability exceeds ``max_p``."""
    keep: list[IdeaSeed] = []
    dropped: list[tuple[IdeaSeed, float]] = []
    for idea in ideas:
        p = predict_refute_probability(idea, memory)
        if p > max_p:
            dropped.append((idea, p))
        else:
            keep.append(idea)
    if dropped:
        log.info(
            "FailurePredictor dropped %d idea(s) with p_refute > %.2f",
            len(dropped), max_p,
        )
        for idea, p in dropped:
            log.info("  - %s (p=%.2f): %s", idea.id, p, idea.title[:60])
    return keep, dropped
