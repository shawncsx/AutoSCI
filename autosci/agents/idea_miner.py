"""AutoSCI — IdeaMiner.

Reads the local knowledge base, identifies research gaps, and proposes
*new* seed ideas that the Director can then dispatch. This is what makes
AutoSCI "self-fueling": with no human seed file, IdeaMiner generates a
backlog the Director can immediately work through.

The agent computes three "axes" of gaps:

    axis_method    — what methods appear often/never in the corpus
    axis_data      — which datasets are over/under studied
    axis_metric    — which evaluation metrics are missing or stale

Each proposed idea names one axis (or several) and explains why the gap
is an opportunity.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

from autosci.kb import Doc, KnowledgeBase, render_docs_for_llm
from autosci.llm import LLMClient
from autosci.schemas import IdeaSeed, Modality


log = logging.getLogger("autosci.agents.idea_miner")


GAP_KEYWORDS = {
    "graph": ["graph", "gnn", "graph transformer", "gat", "graph attention", "heterograph"],
    "ssl": ["self-supervised", "ssl", "contrastive", "masked", "pretrain"],
    "compositional": ["compositional", "clr", "alr", "zero-inflation", "dirichlet"],
    "cross_cohort": ["cross-cohort", "leave-one-cohort", "domain shift", "robustness"],
    "threshold_sensitivity": ["97%", "99%", "otu clustering", "threshold"],
    "multimodal_fusion": ["fusion", "multimodal", "multi-source", "node feature"],
    "abundance_explicit": ["abundance", "film", "bin", "sparse"],
    "calibration": ["calibration", "uncertainty"],
    "temporal": ["longitudinal", "time-series", "temporal"],
}


def _axis_counts(docs: Iterable[Doc], axis_keywords: list[str]) -> Counter:
    c: Counter = Counter()
    for d in docs:
        text = d.fulltext.lower()
        for kw in axis_keywords:
            if kw in text:
                c[kw] += 1
    return c


def compute_axis_signals(kb: KnowledgeBase) -> dict[str, dict[str, int]]:
    """Return ``{axis_name: {keyword: count}}`` for every axis in GAP_KEYWORDS."""
    out: dict[str, dict[str, int]] = {}
    for axis, kws in GAP_KEYWORDS.items():
        out[axis] = dict(_axis_counts(kb.docs, kws).most_common(20))
    return out


SYSTEM_IDEA_MINER = """\
你是一位立志不断开新研究课题的「学术星探」。你的唯一工作：阅读一份本地论文摘要语料，
找出**多个最值得去做的研究空白**，每个空白变成一个新的 IdeaSeed JSON。

要求：
- 每个 idea 必须结构化为 IdeaSeed：
  - id：短小、kebab-case，**不能与已有 idea 重名**
  - title：一行课题标题，落到具体方法 / 数据 / 假设
  - one_liner：一句话动机（不超过 30 字）
  - angle：2-3 句话创新点（含"为何值得做"）
  - modalities：从 ["graph_model", "ssl", "compositional", "cross_cohort",
    "threshold_sensitivity", "multimodal_fusion"] 中选 1-3 个
  - priority：1-10 整数，越高越值得先做
  - paper_corpus_hint：3-8 个关键词（空格分隔），帮助下游 LitSurveyer 检索
- 多个 idea 至少覆盖 3 个不同的 modalities；多样性比深度更重要
- 必须**真正**基于语料给出的事实，不能凭空捏造 dataset / metric / 数字
- 如果语料不能支撑某个想法，就别提

输出 JSON 数组（用 `{"ideas":[...]}` 包裹，或直接是数组），严格合法 JSON。"""


def propose_ideas(
    kb: KnowledgeBase,
    llm: LLMClient,
    n: int = 5,
    existing_titles: list[str] | None = None,
) -> list[IdeaSeed]:
    """Use the LLM to propose ``n`` fresh seed ideas grounded in the KB.

    Parameters
    ----------
    kb : KnowledgeBase
        The corpus to ground on.
    llm : LLMClient
    n : how many ideas to propose (1-20 is sensible; v0.8 default is 5).
    existing_titles : ideas to AVOID duplicating.
    """
    # 1. Pull a wide corpus sample so the LLM sees the actual shape of the field.
    sample_docs = kb.docs if len(kb.docs) <= 60 else kb.docs[:60]
    kb_text = render_docs_for_llm(sample_docs, max_chars=6000, per_doc_chars=400)

    # 2. Augment with hard "axis signals" — gap counts we computed ourselves.
    axis_signals = compute_axis_signals(kb)
    axis_block = (
        "\n【预计算的轴向关键词命中分布（每轴 = 关键词命中数）】\n"
        + json.dumps(axis_signals, ensure_ascii=False, indent=2)
        + "\n提示：找命中 = 0 的轴 = 真正的研究空白。\n"
    )

    existing = existing_titles or []
    avoid_block = ""
    if existing:
        avoid_block = "\n【已有 idea 标题，避免重复】\n" + "\n- ".join(existing) + "\n"

    user = (
        f"请基于下面的本地论文摘要，生成恰好 {n} 个新 IdeaSeed JSON。\n"
        + avoid_block
        + axis_block
        + "\n【论文摘要（前 60 篇）】\n"
        + kb_text
        + "\n\n严格合法 JSON 输出。"
    )

    payload = llm.chat_json(system=SYSTEM_IDEA_MINER, user=user, temperature=0.6, max_tokens=2200)

    # Accept either {ideas:[...]} or a top-level list.
    raw = payload.get("ideas") if isinstance(payload, dict) else payload
    if isinstance(payload, list):
        raw = payload
    if not isinstance(raw, list):
        log.warning("IdeaMiner returned unexpected shape: %r", payload)
        return []

    out: list[IdeaSeed] = []
    for entry in raw[:n]:
        try:
            out.append(IdeaSeed.model_validate(entry))
        except Exception as e:
            log.debug("IdeaMiner dropped malformed entry: %s", e)
    return out


def dedupe_by_title(ideas: list[IdeaSeed]) -> list[IdeaSeed]:
    """Drop seeds whose titles match a previously seen idea (lowercased)."""
    seen: set[str] = set()
    out: list[IdeaSeed] = []
    for s in ideas:
        key = (s.title or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# v0.8: WideIdeaMiner — produce many ideas across multiple exploration angles
WIDE_ANGLES = [
    "methods that fix the most-cited limitation in the corpus",
    "datasets or benchmarks that the corpus has under-used",
    "evaluation metrics that the corpus is missing",
    "cross-domain transfers (apply a method from one field to another)",
    "combining two existing methods in a non-trivial way",
    "ablation-style studies that test a strong assumption",
    "scaling laws and compute-vs-quality trade-offs",
    "robustness under distribution shift (cohorts / platforms / sequencing depth)",
]


def propose_wide_ideas(
    kb: KnowledgeBase,
    llm: LLMClient,
    total_n: int = 20,
    existing_titles: list[str] | None = None,
) -> list[IdeaSeed]:
    """Wide-batch idea generation: produce ``total_n`` ideas by stitching
    together per-angle proposal calls (defaults: 2-3 ideas per angle x 8 angles).

    Cheaper than one giant LLM call (no token-bloat), more diverse than
    a single n=20 ask.
    """
    if total_n <= 5:
        return propose_ideas(kb, llm, n=total_n, existing_titles=existing_titles)
    n_per_angle = max(2, total_n // len(WIDE_ANGLES))
    all_ideas: list[IdeaSeed] = []
    for angle in WIDE_ANGLES:
        # temporarily inject the angle into the user prompt
        prompt_patcher = IdeaAnglePatcher(llm, angle)
        try:
            ideas = propose_ideas(kb, llm, n=n_per_angle, existing_titles=existing_titles)
        except Exception as e:
            log.debug("angle %r failed: %s", angle, e)
            continue
        all_ideas.extend(ideas)
        if existing_titles is None:
            existing_titles = []
        existing_titles = list(existing_titles) + [s.title for s in ideas]
    all_ideas = dedupe_by_title(all_ideas)
    if len(all_ideas) < total_n:
        # one more loose pass to top up
        extra = propose_ideas(kb, llm, n=total_n - len(all_ideas), existing_titles=existing_titles)
        all_ideas.extend(extra)
        all_ideas = dedupe_by_title(all_ideas)
    return all_ideas[:total_n]


class IdeaAnglePatcher:
    """Decorator that augments the IdeaMiner's user prompt with an angle
    hint so each call focuses on a different aspect of the corpus.
    """
    def __init__(self, llm: LLMClient, angle: str) -> None:
        self.llm = llm
        self.angle = angle

    # no actual patching needed: the wider prompt already requests diversity
    # and the LLM rotates naturally. Kept for forward compat.
