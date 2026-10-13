"""AutoSCI — MetaPromptBuilder.

Sometimes an agent's default prompt is the wrong tool for the job. This
module lets an agent *generate a fresh prompt on demand* based on:

- the current state fragment (what the agent has so far)
- the agent's failure history (if any)
- a short *strategy* tag (e.g. "be more skeptical", "ground in figures",
  "use a different angle", "summarise-and-reset")

The output is a string prompt override that the agent can plug back into
its ``build_messages()`` flow.

This is the "self-prompting" layer the user asked for — small but
powerful, because OpenAI's own system instructions for the math pipeline
likely contained hundreds of similar dynamic overlays that we can't see.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from autosci.llm import LLMClient


log = logging.getLogger("autosci.meta")


STRATEGIES = {
    "be_more_skeptical": "Be skeptical of every claim. Demand evidence and weaken any assumption not supported by the KB.",
    "use_a_different_angle": "Generate the answer from a different methodological angle than the previous attempt.",
    "ground_in_figures": "Cite specific numbers, table cells, or figure captions from the KB where possible.",
    "summary_then_proceed": "Produce a 3-sentence summary of what you already know, then proceed.",
    "cut_to_the_bone": "Strip every unnecessary word. ≤ 120 words total. Plain JSON only.",
    "acknowledge_uncertainty": "State your confidence (high / medium / low) inline; do not bluff.",
}


SYSTEM = """\
你是 AutoSCI 的元提示词构造器（meta-prompt builder）。
给你「一个 agent 的任务 + 当前状态片段 + 策略标签」，你要返回**一段 ≥ 100 字的中文 prompt**，
让原 agent 在下一轮用这个 prompt 替换自己默认的 user-prompt。

约束：
- 必须明确「新 prompt 要解决什么失败模式」以及「新 prompt 应该给出哪些额外约束」
- 不要重复 system prompt 里已经写好的硬规则
- 段落清晰，可直接被 LLM 消费
- 仅输出 prompt 文本本身，无 JSON / 无 Markdown 包裹
"""


@dataclass
class Override:
    strategy: str
    new_user_prompt: str
    reason: str = ""


def build_override(
    llm: LLMClient,
    *,
    agent_name: str,
    state_json: str,
    failure_hint: str,
    strategy: str,
) -> Optional[str]:
    """Return a new user-prompt override, or None if strategy is unknown."""
    strategy_line = STRATEGIES.get(strategy)
    if not strategy_line:
        log.warning("Unknown strategy '%s'", strategy)
        return None

    user = (
        f"Agent：{agent_name}\n"
        f"当前状态片段（截断 JSON）：\n```\n{state_json[:3500]}\n```\n"
        f"已知失败：{failure_hint}\n"
        f"策略：{strategy}（{strategy_line}）\n\n"
        "请写出下一轮要使用的 user prompt。"
    )
    payload = llm.chat_json(
        system=SYSTEM,
        user=user + "\n\n严格合法 JSON：{\"new_user_prompt\":\"...\",\"reason\":\"...\"}",
        temperature=0.4,
        max_tokens=1200,
    )
    new = (payload.get("new_user_prompt") or "").strip()
    if not new or len(new) < 40:
        log.warning("MetaPromptBuilder returned too-short override; skipping")
        return None
    return new


def suggest_strategy(failure_hint: str) -> str:
    """Trivial rule-based fallback when the meta call is too costly."""
    h = failure_hint.lower()
    if "重复" in h or "duplicate" in h or "duplicat" in h:
        return "be_more_skeptical"
    if "长" in h or "truncate" in h or "exceed" in h:
        return "cut_to_the_bone"
    if "relevance" in h or "无相关" in h or "不相关" in h:
        return "use_a_different_angle"
    if "citation" in h or "引用" in h:
        return "ground_in_figures"
    return "summary_then_proceed"
