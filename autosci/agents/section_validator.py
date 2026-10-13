"""AutoSCI — SectionValidator (v0.4 new agent).

Runs after PaperWriter + CitationGrounder. For each section we call a
focused LLM to evaluate:

    - word_count          (we compute, not LLM)
    - covers_hypothesis   (does it explicitly address the primary claim?)
    - cites_papers        (does it cite ≥ 1 local paper when appropriate?)
    - quality             (1-5 self-rating; "would I let a master's student submit?")
    - suggestion          (one specific improvement, or "ok")

The output is a SectionValidationReport (per-section) plus an overall
verdict label. Director uses this to demote papers to REVISED if
multiple sections fail.
"""
from __future__ import annotations

import logging
from typing import Optional

from autosci.agents._base import AgentBase
from autosci.schemas import IdeaState, PaperDraft


log = logging.getLogger("autosci.agents.section_validator")


REQUIRED_SECTIONS = [
    "introduction", "related_work", "method", "experiments",
    "results_template", "discussion", "limitations", "conclusion",
]


SECTION_VALIDATION_SYSTEM = """\
你是 AutoSCI 的 section 评审员。给你「论文的某一段 IMRaD 段落 + 全文 claim」，评估这一段是否合格。

要输出严格 JSON：
{
  "covers_hypothesis": <true|false>,    // 是否覆盖 claim
  "cites_papers": <true|false>,         // 是否引用了本地 KB 论文
  "quality": <1-5 整数, 5=硕士能提交, 1=空话>,
  "suggestion": "<一句话具体修改建议，或 ok>"
}
"""


def _per_section_user(section_name: str, body: str, claim: str) -> str:
    return (
        f"【Section】{section_name}\n"
        f"【全文主诉】{claim}\n\n"
        f"【段落正文】\n{body[:3500]}\n\n"
        "请输出 JSON。"
    )


def _word_count(text: str) -> int:
    # crude word count: split on whitespace for English; treat CJK chars as 1 word each
    import re
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    rest = len(re.findall(r"[A-Za-z0-9_]+", text or ""))
    return cjk + rest


def _coerce(payload, section: str) -> dict:
    if not isinstance(payload, dict):
        return {"covers_hypothesis": False, "cites_papers": False, "quality": 1,
                "suggestion": f"LLM returned non-dict for section={section}"}
    return {
        "covers_hypothesis": bool(payload.get("covers_hypothesis", False)),
        "cites_papers": bool(payload.get("cites_papers", False)),
        "quality": int(payload.get("quality", 1)),
        "suggestion": str(payload.get("suggestion", ""))[:500],
    }


class SectionValidatorAgent(AgentBase):
    name = "section_validator"
    output_schema = type(...)  # returns dict

    def run(self, state: IdeaState) -> dict:  # type: ignore[override]
        draft = state.paper_draft
        if not draft or not draft.sections:
            log.warning("[section_validator] no paper_draft; skipping")
            return {"ok": False, "n_passed": 0, "n_failed": 0, "details": {}, "needs_revision": False}

        claim = (state.refined.claim if state.refined else state.idea_seed.one_liner) or ""
        per_section: dict[str, dict] = {}

        # Pre-fill missing sections
        for section in REQUIRED_SECTIONS:
            body = (draft.sections or {}).get(section, "")
            if not body:
                per_section[section] = {
                    "covers_hypothesis": False, "cites_papers": False,
                    "quality": 1, "suggestion": "空 section：缺失或完全空白",
                    "word_count": 0, "verdict": "missing",
                }

        # Batch-evaluate present sections in a SINGLE LLM call (was 8 separate calls)
        sections_to_eval = [s for s in REQUIRED_SECTIONS if s not in per_section]
        if sections_to_eval:
            try:
                payload = self._llm_batch_eval(sections_to_eval, draft.sections or {}, claim)
            except Exception as e:
                log.warning("[section_validator] batch LLM failed: %s", e)
                payload = {}

            for section in sections_to_eval:
                body = (draft.sections or {}).get(section, "")
                p = payload.get(section) or {}
                entry = _coerce(p, section)
                entry["word_count"] = _word_count(body)
                entry["verdict"] = "ok" if entry["quality"] >= 3 else "weak"
                per_section[section] = entry

        n_passed = sum(1 for v in per_section.values() if v.get("verdict") == "ok")
        n_failed = len(REQUIRED_SECTIONS) - n_passed
        needs_revision = n_failed >= 2 or any(
            v.get("quality", 1) == 1 for v in per_section.values()
        )

        summary = {
            "ok": True,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "needs_revision": needs_revision,
            "details": per_section,
        }
        log.info(
            "[section_validator] %d/%d sections pass; needs_revision=%s",
            n_passed, len(REQUIRED_SECTIONS), needs_revision,
        )
        state.history_log.append(
            f"[section_validator] passed={n_passed} failed={n_failed} needs_revision={needs_revision}"
        )
        return summary

    # ------------------------------------------------------------------
    def _llm_batch_eval(
        self,
        sections: list[str],
        bodies: dict[str, str],
        claim: str,
    ) -> dict[str, dict]:
        """One LLM call that returns per-section judgments.

        Output schema (JSON):
            {
              "introduction": {"covers_hypothesis": bool, "cites_papers": bool,
                               "quality": 1-5, "suggestion": "..."},
              "related_work": {...},
              ...
            }
        """
        system = (
            "你是 AutoSCI 的 IMRaD 评审员。给你一个 IMRaD 论文的多个 section 段落 + 全文主诉，"
            "请对**每个** section 独立评分。评分维度：\n"
            "  - covers_hypothesis: 是否覆盖 claim\n"
            "  - cites_papers: 是否引用了本地 KB 论文（出现 [cite_key]）\n"
            "  - quality: 1-5 整数，5=硕士可提交，1=空话\n"
            "  - suggestion: 一句话具体修改建议，或 'ok'\n"
            "严格合法 JSON 输出，键名固定为传入的 section 名。\n"
            "如果某 section 内容为空（你看不见内容），把 covers_hypothesis=false, quality=1, suggestion='缺失或完全空白'。"
        )
        # Build the user prompt with all sections
        chunks: list[str] = []
        for s in sections:
            body = (bodies or {}).get(s, "")
            chunks.append(f"## {s}\n{body[:1500] if body else '(empty)'}\n")
        user = (
            f"【全文主诉】{claim}\n\n"
            + "\n".join(chunks)
            + f"\n请按 system 要求输出 JSON，键名必须为：{', '.join(sections)}。"
        )

        if hasattr(self.llm, "with_timeout"):
            self.llm.with_timeout(self.name)
        try:
            payload = self.llm.chat_json(
                system=system, user=user, temperature=0.2, max_tokens=2000,
            )
        finally:
            if hasattr(self.llm, "reset_timeout"):
                self.llm.reset_timeout()

        if not isinstance(payload, dict):
            return {}
        # Strip unknown keys (so caller only sees the sections we asked about)
        return {k: v for k, v in payload.items() if k in set(sections) and isinstance(v, dict)}
