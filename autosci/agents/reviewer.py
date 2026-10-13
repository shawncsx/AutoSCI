"""AutoSCI — Reviewer."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import Critique, IdeaState


class ReviewerAgent(AgentBase):
    name = "reviewer"
    output_schema = Critique

    def run(self, state: IdeaState) -> Critique:  # type: ignore[override]
        out: Critique = super().run(state, temperature=0.2, max_tokens=2000)

        # Compute severity counts if missing.
        if not out.severity_counts:
            cnt = {"minor": 0, "major": 0, "blocking": 0}
            for li in out.line_items or []:
                key = (li.severity or "minor").lower()
                if key not in cnt:
                    key = "minor"
                cnt[key] += 1
            out.severity_counts = cnt

        state.review = out
        state.history_log.append(f"[reviewer] {len(out.line_items)} items, blocking={out.severity_counts.get('blocking', 0)}")
        return out
