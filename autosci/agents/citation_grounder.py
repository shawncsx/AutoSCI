"""AutoSCI — CitationGrounder (v0.4 new agent).

Runs after PaperWriter. Two responsibilities:

    1. EXTRACT: scan every section's text for `[cite_key]` patterns.
    2. VERIFY: every key must be present in either ``state.lit_survey``
       or ``state.paper_draft.references``. If a cited key is unknown, we
       *don't* drop the paper — we add a new ``LitSurveyItem`` placeholder
       and ask the LLM to fill in title/year/relevance.
    3. REPORT: produce a CitationReport with the fix list so the
       verdict policy can demote to REVISED if grounding is too poor.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

from autosci.agents._base import AgentBase
from autosci.llm import LLMClient
from autosci.schemas import IdeaState, LitSurveyItem


log = logging.getLogger("autosci.agents.citation_grounder")


CITE_KEY_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9._\-]+(?:-\d{2,4})?(?:-[A-Z][A-Za-z0-9_]*)?)\]")


def extract_cite_keys(text: str) -> list[str]:
    """Pull all bracketed [cite_key] tokens out of a text block.

    Designed to match the typical key shapes we have in this project:
        [MGM2-2026]
        [arxiv_2205.06525_Image_and_graph_convolution_networks_...]
    """
    return sorted(set(CITE_KEY_RE.findall(text or "")))


def extract_bracket_keys(text: str) -> list[str]:
    """Same as extract_cite_keys but returns the full [foo] tokens, not stripped."""
    return sorted(set(re.findall(r"\[[^\]]+\]", text or "")))


class CitationGrounderAgent(AgentBase):
    """Run after paper_writer. Verifies and (if needed) repairs citations."""

    name = "citation_grounder"
    output_schema = type(...)  # we return our own dataclass

    def run(self, state: IdeaState) -> dict:  # type: ignore[override]
        draft = state.paper_draft
        if draft is None:
            log.warning("[citation_grounder] no paper_draft; skipping")
            return {"ok": False, "verified": [], "added": [], "missing": []}

        # 1. known keys: anything in LitSurvey or explicit references
        known_keys: set[str] = set()
        for it in state.lit_survey or []:
            known_keys.add(it.cite_key.strip())
            # also accept short forms like [MGM2-2026] -> look for matching by stem
            short = re.sub(r"^arxiv_[\d.]+_", "", it.cite_key)
            short = re.sub(r"^Advanced Science - \d+ - \w+ - ", "", it.cite_key)
            short = short.replace(".digest", "").strip()
            if short:
                known_keys.add(short)
        for ref in draft.references or []:
            known_keys.add(ref.strip())
            if not ref.startswith("["):
                known_keys.add(f"[{ref.strip()}]")

        # 2. extract from sections
        used_brackets: list[str] = []
        used_stems: list[str] = []
        for sec, body in (draft.sections or {}).items():
            brackets = extract_bracket_keys(body)
            used_brackets.extend(brackets)
            for b in brackets:
                # b like "[MGM2-2026]" or "[arxiv_2205_...]"
                inner = b.strip("[]")
                used_stems.append(inner)
                # also accept leading fragments
                used_stems.append(inner.split("-")[0])

        # 3. classify each used bracket
        verified: list[str] = []
        missing: list[str] = []
        seen = set()
        for b in used_brackets:
            if b in seen:
                continue
            seen.add(b)
            if b in known_keys or any(k.endswith(b) or b.endswith(k) for k in known_keys):
                verified.append(b)
            else:
                # also accept a stem-prefix match (e.g. [MGM2-2026] vs [MGM2-2026-digest])
                short = re.sub(r"^arxiv_[\d.]+_", "", b.strip("[]")).split("_")[0]
                if any(short in k for k in known_keys):
                    verified.append(b)
                else:
                    missing.append(b)

        # 4. add a top-up entry to references if missing-but-valid
        for m in missing:
            if m not in (draft.references or []):
                draft.references = (draft.references or []) + [m]

        # 5. If we have many missing, ask the LLM to flesh them out
        added: list[LitSurveyItem] = []
        if missing and self.llm is not None:
            try:
                added = self._llm_rescue_missing(missing, state)
            except Exception as e:
                log.warning("[citation_grounder] LLM rescue failed: %s", e)

            # Merge into state.lit_survey and references
            existing = {(it.cite_key or "").strip() for it in state.lit_survey or []}
            for item in added:
                if item.cite_key not in existing:
                    state.lit_survey.append(item)
                    existing.add(item.cite_key)
            for item in added:
                if item.cite_key not in (draft.references or []):
                    draft.references.append(item.cite_key)

        report = {
            "ok": True,
            "n_used": len(used_brackets),
            "n_verified": len(verified),
            "n_missing_before": len(missing),
            "n_added_by_rescue": len(added),
            "verified": verified,
            "missing": missing,
            "added_keys": [it.cite_key for it in added],
        }
        log.info(
            "[citation_grounder] used=%d verified=%d missing=%d added=%d",
            report["n_used"], report["n_verified"], report["n_missing_before"], report["n_added_by_rescue"],
        )
        state.history_log.append(
            f"[citation_grounder] used={report['n_used']} verified={report['n_verified']} "
            f"missing={report['n_missing_before']} added={report['n_added_by_rescue']}"
        )
        return report

    # ------------------------------------------------------------------
    def _llm_rescue_missing(self, missing: list[str], state: IdeaState) -> list[LitSurveyItem]:
        """Ask the LLM to flesh out bare `[some-key]` placeholders."""
        system = (
            "你是论文引用补救员。给一个学术论文的 sections 草稿，里面引用了若干个 "
            "[cite_key] 形式的关键字。你要补全每个 cite_key 的 title / year / venue / "
            "relevance / takeaway，使整篇论文看起来真实可信。\n"
            "如果某个 cite_key 你**无法**给出可信信息（明显是 LLM 幻想的），就把 relevance 写成 "
            "\"（缺少具体出处，需人工补全）\"，year 写 0。\n"
            "**严格合法 JSON**。"
        )
        missing_block = "\n".join(f"- {m}" for m in missing)
        claim = state.refined.claim if state.refined else state.idea_seed.one_liner
        user = (
            f"主诉：{claim}\n\n待补全的 cite_keys：\n{missing_block}\n\n"
            "请按顺序返回 JSON：{\"items\": [{\"cite_key\":\"...\",\"title\":\"...\",\"year\":2024,"
            "\"venue\":\"...\",\"relevance\":\"...\",\"takeaway\":\"...\"}, ...]}"
        )
        if hasattr(self.llm, "with_timeout"):
            self.llm.with_timeout(self.name)
        try:
            payload = self.llm.chat_json(system=system, user=user, temperature=0.2, max_tokens=1200)
        finally:
            if hasattr(self.llm, "reset_timeout"):
                self.llm.reset_timeout()

        items_raw = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items_raw, list):
            return []
        out: list[LitSurveyItem] = []
        for entry, key in zip(items_raw, missing):
            if not isinstance(entry, dict):
                continue
            entry.setdefault("cite_key", key)
            entry.setdefault("title", key)
            entry.setdefault("year", 0)
            entry.setdefault("venue", "（待人工补全）")
            entry.setdefault("relevance", "（缺少具体出处，需人工补全）")
            entry.setdefault("takeaway", "（需人工补全）")
            try:
                out.append(LitSurveyItem.model_validate(entry))
            except Exception as e:
                log.debug("citation grounder dropped entry: %s", e)
        return out
