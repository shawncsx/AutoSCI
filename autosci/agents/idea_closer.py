"""AutoSCI — IdeaCloser: tightens a research idea into a falsifiable claim."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import IdeaRefined, IdeaState


class IdeaCloserAgent(AgentBase):
    name = "idea_closer"
    output_schema = IdeaRefined

    def run(self, state: IdeaState) -> IdeaRefined:  # type: ignore[override]
        out: IdeaRefined = super().run(state, temperature=0.4)
        # Persist onto state for downstream agents.
        state.refined = out
        state.history_log.append(f"[idea_closer] claim={out.claim[:60]}…")
        return out
