"""AutoSCI — Refuter."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import IdeaState, RefuteResult


class RefuterAgent(AgentBase):
    name = "refuter"
    output_schema = RefuteResult

    def run(self, state: IdeaState) -> RefuteResult:  # type: ignore[override]
        out: RefuteResult = super().run(state, temperature=0.1, max_tokens=1500)
        state.refute = out
        state.history_log.append(f"[refuter] verdict={out.verdict}, fatal={len(out.fatal_flaws)}")
        return out
