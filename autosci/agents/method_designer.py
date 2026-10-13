"""AutoSCI — MethodDesigner."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import IdeaState, MethodSpec


class MethodDesignerAgent(AgentBase):
    name = "method_designer"
    output_schema = MethodSpec

    def run(self, state: IdeaState) -> MethodSpec:  # type: ignore[override]
        out: MethodSpec = super().run(state, temperature=0.3, max_tokens=1600)
        state.method_spec = out
        state.history_log.append(f"[method_designer] novelty={len(out.novelty_bullets)} items")
        return out
