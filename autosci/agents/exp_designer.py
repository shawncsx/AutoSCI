"""AutoSCI — ExpDesigner."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import ExpSpec, IdeaState


class ExpDesignerAgent(AgentBase):
    name = "exp_designer"
    output_schema = ExpSpec

    def run(self, state: IdeaState) -> ExpSpec:  # type: ignore[override]
        out: ExpSpec = super().run(state, temperature=0.3, max_tokens=1500)
        state.exp_spec = out
        state.history_log.append(f"[exp_designer] {len(out.datasets)} datasets / {len(out.baselines)} baselines")
        return out
