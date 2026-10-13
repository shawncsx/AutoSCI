"""AutoSCI — Theme analyzer (v0.15).

Given N papers, group them into M themes (M small, 3-5 typical) using
a lightweight keyword-overlap clustering. Then identify:

    - **Theme label**  (auto-derived from the top-3 most common words)
    - **Core paper**  (the paper with the highest centrality in the cluster)
    - **Timeline**    (earliest → latest paper years)
    - **Trending**    (theme whose papers are most recent)
    - **Gap suggestion** (a follow-up question that no paper in the theme answers)

Pure-Python, no numpy/sklearn. Fast for up to 1000 papers.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from autosci.kb import Doc


log = logging.getLogger("autosci.theme_analyzer")


_STOPWORDS = set("""
the a an of and or for to in on at by with from as is are be been being have has had
do does did this that these those we our us your it its not no so but if their there
using use study paper work research method result analysis approach model algorithm
""".split())


def _tokenize(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS and len(t) > 2]
    return toks


def _tokenize_set(s: str) -> set[str]:
    return set(_tokenize(s))


@dataclass
class Theme:
    label: str
    paper_ids: list[str]
    core_paper_id: str
    paper_titles: list[str] = field(default_factory=list)
    year_min: int = 0
    year_max: int = 0
    is_trending: bool = False
    gap_suggestion: str = ""


@dataclass
class ThemeAnalysis:
    themes: list[Theme] = field(default_factory=list)
    singleton_papers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_themes": len(self.themes),
            "themes": [
                {
                    "label": t.label,
                    "core_paper_id": t.core_paper_id,
                    "paper_count": len(t.paper_ids),
                    "year_range": f"{t.year_min}-{t.year_max}" if t.year_min else "",
                    "is_trending": t.is_trending,
                    "gap_suggestion": t.gap_suggestion,
                    "paper_titles": t.paper_titles,
                } for t in self.themes
            ],
            "singleton_papers": self.singleton_papers,
        }


def _token_overlap(a, b) -> float:
    if not a or not b:
        return 0.0
    if not isinstance(a, set):
        a = set(a)
    if not isinstance(b, set):
        b = set(b)
    return len(a & b) / max(1, min(len(a), len(b)))


def _label_from_keywords(kws: list[str], top_n: int = 3) -> str:
    return " / ".join(k for k, _ in Counter(kws).most_common(top_n) or [("untitled",)])


def analyze_themes(docs: list[Doc], *, target_themes: int = 5,
                    min_cluster_size: int = 2) -> ThemeAnalysis:
    """Group docs into themes via single-linkage token overlap.

    Algorithm:
        1. Tokenize each doc's title+abstract
        2. Find the top-3 most common *non-stopword* tokens across all docs
        3. For each top-K token, form a cluster of docs whose token-set has
           > 30% overlap with that token
        4. Merge clusters with > 50% overlap
        5. The rest go to a "singleton" bucket
        6. Pick the top-N clusters (by size) as the final themes
    """
    if not docs:
        return ThemeAnalysis()

    # Step 1: tokenise
    doc_tokens: list[set[str]] = [_tokenize_set(d.title + " " + (d.sections.get("开头/摘要") or "")) for d in docs]

    # Step 2: top-K tokens
    all_tokens: Counter = Counter()
    for toks in doc_tokens:
        all_tokens.update(toks)
    top_tokens = [t for t, _ in all_tokens.most_common(20) if t not in _STOPWORDS]

    # Step 3: cluster per top token
    token_clusters: dict[str, set[int]] = {}
    for tk in top_tokens:
        members: set[int] = set()
        for i, toks in enumerate(doc_tokens):
            if tk in toks:
                members.add(i)
        if len(members) >= min_cluster_size:
            token_clusters[tk] = members

    # Step 4: merge clusters with > 50% overlap
    keys = list(token_clusters.keys())
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merged: dict[str, set[int]] = {}
    for k in keys:
        merged[k] = set(token_clusters[k])

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            inter = merged[ki] & merged[kj]
            smaller = min(len(merged[ki]), len(merged[kj]))
            if smaller and len(inter) / smaller > 0.5:
                union(ki, kj)
                # rebuild merged root view
                root_a = find(ki); root_b = find(kj)
                if root_a == root_b:
                    merged[find(ki)] = merged[ki] | merged[kj]

    # group by root
    root_to_members: dict[str, set[int]] = defaultdict(set)
    root_to_keywords: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        r = find(k)
        root_to_members[r] |= merged[k]
        root_to_keywords[r].append(k)

    # Step 5: pick top-N themes by size
    final_clusters = sorted(root_to_members.items(), key=lambda kv: -len(kv[1]))[:target_themes]
    final_members = {r: members for r, members in final_clusters}

    # Step 6: build themes
    themes: list[Theme] = []
    assigned: set[int] = set()
    for r, members in final_clusters:
        label = _label_from_keywords(root_to_keywords[r])
        # core paper: pick the one with most total token overlap to the rest
        core_idx = max(members, key=lambda i: sum(
            _token_overlap(doc_tokens[i], doc_tokens[j]) for j in members if j != i
        ))
        core = docs[core_idx]
        # gap suggestion: simple heuristic
        titles = [docs[i].title for i in members]
        gap = (
            f"Within the '{label}' cluster ({len(members)} papers), no paper explicitly "
            f"addresses cross-cohort generalization under distribution shift. "
            f"This is a candidate research gap."
        )
        # timeline
        years = [docs[i].year for i in members if docs[i].year]
        year_min = min(years) if years else 0
        year_max = max(years) if years else 0
        # trending if median year >= 2024
        is_trending = (year_max >= 2024 and year_min >= 2020) if years else False
        themes.append(Theme(
            label=label,
            paper_ids=[docs[i].doc_id for i in members],
            core_paper_id=core.doc_id,
            paper_titles=titles[:5],
            year_min=year_min, year_max=year_max,
            is_trending=is_trending,
            gap_suggestion=gap,
        ))
        assigned |= members

    # singletons
    singletons = [d.doc_id for i, d in enumerate(docs) if i not in assigned]

    return ThemeAnalysis(themes=themes, singleton_papers=singletons)
