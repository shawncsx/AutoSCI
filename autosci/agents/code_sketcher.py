"""AutoSCI — CodeSketcher."""
from __future__ import annotations

from autosci.agents._base import AgentBase
from autosci.schemas import CodeSketch, IdeaState


class CodeSketcherAgent(AgentBase):
    name = "code_sketcher"
    output_schema = CodeSketch

    def run(self, state: IdeaState) -> CodeSketch:  # type: ignore[override]
        out: CodeSketch = super().run(state, temperature=0.3, max_tokens=2500)
        state.code_sketch = out
        state.history_log.append(f"[code_sketcher] {len(out.files)} files")
        return out
