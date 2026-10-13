"""AutoSCI — LitSurveyer: picks k relevant papers from KB."""
from __future__ import annotations

import logging
from typing import Optional

from autosci.agents._base import AgentBase
from autosci.kb import Doc, KnowledgeBase, make_llm_reranker, render_docs_for_llm
from autosci.llm import LLMClient
from autosci.schemas import IdeaState, LitSurveyItem


log = logging.getLogger("autosci.agents.lit_surveyer")


class LitSurveyerAgent(AgentBase):
    name = "lit_surveyer"
    output_schema = type(...)  # see run() — we return list[LitSurveyItem]

    def __init__(self, llm: LLMClient, kb: KnowledgeBase) -> None:
        super().__init__(llm)
        self.kb = kb

    # LitSurveyer's output schema is a *list* of items inside a wrapper.
    # We override output coercion to produce that list.

    def run(self, state: IdeaState, k: int = 8) -> list[LitSurveyItem]:  # type: ignore[override]
        # 1) Pick top candidate docs (BM25 + LLM rerank).
        reranker = make_llm_reranker(self.llm, k=k)
        claim_text = (state.refined.claim if state.refined else state.idea_seed.one_liner) + " " + state.idea_seed.paper_corpus_hint
        docs: list[Doc] = self.kb.search(claim_text, k=k, candidate_pool=30, rerank_with=reranker)

        # 2) Render for prompt
        kb_text = render_docs_for_llm(docs, max_chars=4500, per_doc_chars=400)

        # 3) Ask the agent to emit structured items
        if hasattr(self.llm, "with_timeout"):
            self.llm.with_timeout(self.name)
        try:
            payload = self.llm.chat_json(
                system=self._system_for(k),
                user=self._user_for(state, k, kb_text),
                temperature=0.2,
            )
        finally:
            if hasattr(self.llm, "reset_timeout"):
                self.llm.reset_timeout()

        # 4) Coerce into list[LitSurveyItem]. We accept either {items:[...]}
        #    or a flat list. Validate each item individually.
        items_raw = payload.get("items") if isinstance(payload, dict) else None
        if items_raw is None and isinstance(payload, list):
            items_raw = payload
        if items_raw is None:
            log.warning("LitSurveyer returned unexpected shape: %r", payload)
            items_raw = []

        validated: list[LitSurveyItem] = []
        for entry in items_raw:
            try:
                validated.append(LitSurveyItem.model_validate(entry))
            except Exception as e:
                log.debug("Skipped malformed LitSurveyItem: %s", e)

        # 5) Always inject the KB docs as fallback if model returned too few,
        #    so the pipeline is robust if LLM under-produces.
        if len(validated) < min(4, k):
            for d in docs:
                if len(validated) >= k:
                    break
                # Only inject ones we haven't already covered.
                if any(it.title == d.title for it in validated):
                    continue
                validated.append(
                    LitSurveyItem(
                        cite_key=d.cite_key,
                        title=d.title,
                        year=d.year,
                        venue=d.venue,
                        relevance="从本地 KB BM25 兜底引入（LLM 抽取条目不足时补救）",
                        takeaway="借鉴其方法 / 数据 / 评估协议作为对比基线之一",
                    )
                )

        state.lit_survey = validated
        state.history_log.append(f"[lit_surveyer] picked {len(validated)} papers")
        return validated[:k]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _system_for(self, k: int) -> str:
        from autosci.prompts import SYSTEM_LIT_SURVEYER
        return SYSTEM_LIT_SURVEYER.replace("{{ k }}", str(k))

    def _user_for(self, state: IdeaState, k: int, kb_text: str) -> str:
        from autosci.prompts import user_lit_surveyer
        return user_lit_surveyer(state, k=k, kb_text=kb_text)
