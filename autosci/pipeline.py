"""AutoSCI — ResearchTeam: orchestrates 8 sub-agents per idea.

Each ``run_one`` invocation walks one IdeaSeed through:

    1. IdeaCloser       -> IdeaRefined
    2. LitSurveyer      -> list[LitSurveyItem]
    3. MethodDesigner   -> MethodSpec
    4. ExpDesigner      -> ExpSpec
    5. PaperWriter      -> PaperDraft
    6. CodeSketcher     -> CodeSketch
    7. Reviewer         -> Critique
    8. Refuter          -> RefuteResult (verdict: keep | revise | refute)

Each step writes two artifacts under ``runs/<idea.id>/``:

    state.json                  full state after the step
    steps/<seq>_<agent>.md     the agent's structured output rendered as markdown
    paper.md                    final paper rendered as a single markdown doc
    code/                       python files written from CodeSketch

The whole pipeline is idempotent: re-running skips steps whose state is
already populated.
"""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path

from autosci.agents import (
    CodeSketcherAgent,
    ExpDesignerAgent,
    IdeaCloserAgent,
    LitSurveyerAgent,
    MethodDesignerAgent,
    PaperWriterAgent,
    RefuterAgent,
    ReviewerAgent,
    CitationGrounderAgent,
    SectionValidatorAgent,
)
from autosci.kb import KnowledgeBase
from autosci.llm import LLMClient
from autosci.schemas import IdeaSeed, IdeaState, Verdict


log = logging.getLogger("autosci.pipeline")


@dataclass
class RunConfig:
    """Tuneable knobs for one pipeline run."""
    output_root: Path
    kb_topk: int = 8
    lit_pool: int = 30
    write_paper_md: bool = True


class ResearchTeam:
    """Drives a single project through the 8-agent pipeline."""

    def __init__(self, llm: LLMClient, kb: KnowledgeBase, cfg: RunConfig | None = None) -> None:
        self.llm = llm
        self.kb = kb
        self.cfg = cfg or RunConfig(output_root=Path("runs"))

        self.idea_closer = IdeaCloserAgent(llm)
        self.lit_surveyer = LitSurveyerAgent(llm, kb)
        self.method_designer = MethodDesignerAgent(llm)
        self.exp_designer = ExpDesignerAgent(llm)
        self.paper_writer = PaperWriterAgent(llm)
        self.code_sketcher = CodeSketcherAgent(llm)
        self.citation_grounder = CitationGrounderAgent(llm)
        self.section_validator = SectionValidatorAgent(llm)
        self.reviewer = ReviewerAgent(llm)
        self.refuter = RefuterAgent(llm)

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------

    def run_one(self, seed: IdeaSeed, state: IdeaState | None = None) -> IdeaState:
        state = state or IdeaState(idea_seed=seed)
        run_dir = self.cfg.output_root / seed.id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._run_step("idea_closer", run_dir, state, lambda: self.idea_closer.run(state))
        self._run_step("lit_surveyer", run_dir, state, lambda: self.lit_surveyer.run(state, k=self.cfg.kb_topk))
        self._run_step("method_designer", run_dir, state, lambda: self.method_designer.run(state))
        self._run_step("exp_designer", run_dir, state, lambda: self.exp_designer.run(state))
        self._run_step("paper_writer", run_dir, state, lambda: self.paper_writer.run(state))
        # v0.4: Citation grounder runs *after* PaperWriter, before Reviewer
        self._run_step("citation_grounder", run_dir, state, lambda: self.citation_grounder.run(state))
        # v0.4: Section validator — independent LLM rates each IMRaD section.
        self._run_step("section_validator", run_dir, state, lambda: self.section_validator.run(state))

        if state.code_sketch is None:
            self._run_step("code_sketcher", run_dir, state, lambda: self.code_sketcher.run(state))
        if state.code_sketch is not None:
            self._write_code_files(run_dir, state)

        self._run_step("reviewer", run_dir, state, lambda: self.reviewer.run(state))
        self._run_step("refuter", run_dir, state, lambda: self.refuter.run(state))
        # v0.7: pseudo-experiment verifier (numpy only)
        self._run_step("verifier", run_dir, state, lambda: self._run_verifier(state, run_dir))
        # v0.11: real experiment verifier (sklearn baselines if available)
        self._run_step("realverifier", run_dir, state, lambda: self._run_realverifier(state, run_dir))

        # Decide verdict
        self._set_verdict(state)

        # Persist final state + paper.md
        self._write_state(state, run_dir)
        if self.cfg.write_paper_md:
            self._write_paper_md(state, run_dir)

        return state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_step(self, name: str, run_dir: Path, state: IdeaState, fn) -> None:
        log.info("[%s] %s ...", state.idea_seed.id, name)
        try:
            fn()
        except Exception as e:
            log.error("[%s] %s failed: %s\n%s", state.idea_seed.id, name, e, traceback.format_exc())
            state.history_log.append(f"[{name}] FAILED: {e}")
            raise

    def _set_verdict(self, state: IdeaState) -> None:
        """Decide `kept` / `revised` / `refuted` from Refuter + Reviewer + SectionValidator + Verifier.

        Policy (v0.7):
            1. Refuter verdict='refute' + at least 1 fatal_flaw → REFUTED
            2. Refuter verdict='refute' but no fatal flaws → REVISED (still useful)
            3. Reviewer with >= 2 blocking items → REVISED (no matter what refuter said)
            4. SectionValidator needs_revision=True → at most REVISED
            5. Verifier (v0.7) — claim target exists but unattained → at most REVISED
            6. Refuter verdict='keep' AND no fatal flaws AND no other vetoes:
                - Reviewer blocking == 0  → KEPT
                - Reviewer blocking == 1  → REVISED
                - Reviewer blocking >= 2 → REVISED
            7. Refuter verdict='revise' → REVISED
            8. Anything unrecognised → REVISED (visible)
        """
        from autosci.agents.section_validator import REQUIRED_SECTIONS  # local
        refute = state.refute
        review = state.review
        if refute is None:
            state.verdict = Verdict.PENDING
            return
        v = (refute.verdict or "").lower().strip()
        n_fatal = len(refute.fatal_flaws or [])
        n_block = (review.severity_counts.get("blocking", 0) if review else 0)

        # SectionValidator signal
        sv_needs_revision = False
        for h in reversed(state.history_log):
            if h.startswith("[section_validator]"):
                sv_needs_revision = "needs_revision=True" in h
                break

        # v0.7 Verifier signal
        v_unverified = False
        for h in reversed(state.history_log):
            if h.startswith("[verifier]"):
                v_unverified = "verified=False" in h
                break

        # 1+2
        if v == "refute":
            state.verdict = Verdict.REFUTED if n_fatal > 0 else Verdict.REVISED
            return

        # 3
        if n_block >= 2:
            state.verdict = Verdict.REVISED
            return

        # 4
        if sv_needs_revision and v == "keep" and n_fatal == 0 and n_block <= 1:
            state.verdict = Verdict.REVISED
            return

        # 5 v0.7 — verifier vetoes KEPT
        if v_unverified and v == "keep" and n_fatal == 0 and n_block <= 1:
            state.verdict = Verdict.REVISED
            return

        # 6
        if v == "keep" and n_fatal == 0:
            state.verdict = Verdict.KEPT if n_block == 0 else Verdict.REVISED
            return

        # 7
        if v == "revise":
            state.verdict = Verdict.REVISED
            return

        # 8
        state.verdict = Verdict.REVISED

    def _write_state(self, state: IdeaState, run_dir: Path) -> None:
        (run_dir / "state.json").write_text(
            state.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_paper_md(self, state: IdeaState, run_dir: Path) -> None:
        if not state.paper_draft:
            return
        d = state.paper_draft
        lines = [f"# {d.title}", "", "## Abstract", "", d.abstract, ""]
        if d.keywords:
            lines += ["**Keywords:** " + ", ".join(d.keywords), ""]
        lines += ["## Table of Contents", ""]
        for k in d.sections.keys():
            lines.append(f"- [{k}](#{k.replace('_','-')})")
        lines.append("")
        section_order = [
            "introduction", "related_work", "method", "experiments",
            "results_template", "discussion", "limitations", "conclusion",
        ]
        for k in section_order:
            content = d.sections.get(k, "")
            if content:
                lines += [f"## {k.replace('_', ' ').title()}", "", content, ""]
        if d.references:
            lines += ["## References", ""]
            for i, ref in enumerate(d.references, 1):
                lines.append(f"{i}. {ref}")
            lines.append("")

        if state.refute is not None:
            lines += [
                "## Internal Audit",
                "",
                f"**Refuter verdict:** {state.refute.verdict}",
                "",
                f"**Fatal flaws:** {len(state.refute.fatal_flaws)}",
                "",
            ]
            for f in state.refute.fatal_flaws or []:
                lines.append(f"- {f.get('flaw','?')}  // evidence: {f.get('evidence','?')}")
            lines.append("")

        if state.review is not None:
            lines.append(f"**Reviewer verdict:** {state.review.summary}")
            counts = state.review.severity_counts
            lines.append(
                f"   severity counts: {counts.get('minor', 0)} minor / "
                f"{counts.get('major', 0)} major / {counts.get('blocking', 0)} blocking"
            )
            lines.append("")

        md = "\n".join(lines)
        path = run_dir / "paper.md"
        path.write_text(md, encoding="utf-8")
        state.paper_path = str(path)

    def _write_code_files(self, run_dir: Path, state: IdeaState) -> None:
        if not state.code_sketch:
            return
        code_dir = run_dir / "code"
        code_dir.mkdir(exist_ok=True)
        paths: list[str] = []
        for f in state.code_sketch.files:
            target = code_dir / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")
            paths.append(str(target))
        (code_dir / "README.md").write_text(
            "# AutoSCI code sketch\n\n"
            + (state.code_sketch.run_instructions or "")
            + "\n",
            encoding="utf-8",
        )
        state.code_paths = paths

    # ------------------------------------------------------------------
    # v0.7 — pseudo-experiment verifier
    # ------------------------------------------------------------------
    def _run_verifier(self, state: IdeaState, run_dir: Path) -> None:
        try:
            from autosci.verifier import verify_and_persist
            verify_and_persist(state, out_dir=run_dir.parent)
        except Exception as e:
            log.warning("verifier failed for %s: %s", state.idea_seed.id, e)

    # ------------------------------------------------------------------
    # v0.11 — sklearn-based real experiment verifier
    # ------------------------------------------------------------------
    def _run_realverifier(self, state: IdeaState, run_dir: Path) -> None:
        try:
            from autosci.realverifier import run_real_verifier
            res = run_real_verifier()
            import json
            target = run_dir / "realverification.json"
            target.write_text(json.dumps(res.__dict__, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            note = (
                f"[realverifier] sklearn_available={res.sklearn_available} "
                f"method_auroc={res.method_auroc_mean:.3f}±{res.method_auroc_std:.3f} "
                f"best_baseline_delta={res.delta_vs_best_baseline_mean:+.3f}"
            )
            state.history_log.append(note)
        except Exception as e:
            log.warning("realverifier failed for %s: %s", state.idea_seed.id, e)
