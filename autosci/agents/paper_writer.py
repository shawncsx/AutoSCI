"""AutoSCI — PaperWriter."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import IdeaState, PaperDraft


REQUIRED_SECTIONS = [
    "introduction", "related_work", "method", "experiments",
    "results_template", "discussion", "limitations", "conclusion",
]


class PaperWriterAgent(AgentBase):
    name = "paper_writer"
    output_schema = PaperDraft

    def run(self, state: IdeaState) -> PaperDraft:  # type: ignore[override]
        out: PaperDraft = super().run(state, temperature=0.4, max_tokens=4000)

        # Make sure every required section is present; missing → empty string.
        sections = dict(out.sections or {})
        for k in REQUIRED_SECTIONS:
            sections.setdefault(k, "")
        out.sections = sections

        # If references are sparse, top-up from lit_survey cite_keys.
        if len(out.references or []) < 8 and state.lit_survey:
            existing = set(out.references or [])
            for it in state.lit_survey:
                if it.cite_key not in existing:
                    out.references.append(it.cite_key)
                    existing.add(it.cite_key)
                    if len(out.references) >= 8:
                        break

        state.paper_draft = out
        state.history_log.append(f"[paper_writer] {len(out.sections)} sections, {len(out.references)} refs")
        return out
