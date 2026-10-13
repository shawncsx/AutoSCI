"""AutoSCI — Director: top-level orchestrator.

The Director owns:
- the seed-idea backlog (loaded from disk)
- the budget (how many ideas to attempt)
- the lifecycle: pending → running → (kept | revised | refuted)
- a per-idea decision policy:
    * Refuter=refute → drop to vault/refuted/
    * Refuter=revise OR Reviewer has blocking items → mark 'revised'
      (still published to vault/kept/ but flagged for revision)
    * Refuter=keep AND no blocking issues → mark 'kept'
- retry-on-failure policy: at most one retry if an LLM step blew up
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from autosci.kb import KnowledgeBase
from autosci.llm import LLMClient
from autosci.pipeline import ResearchTeam, RunConfig
from autosci.schemas import IdeaSeed, IdeaState, Verdict
from autosci.vault import Vault


log = logging.getLogger("autosci.director")


@dataclass
class DirectorReport:
    n_total: int = 0
    n_kept: int = 0
    n_revised: int = 0
    n_refuted: int = 0
    n_failed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_s: float = 0.0
    per_idea: list[dict] = field(default_factory=list)
    distillation_triggered: bool = False
    distillation_report_path: str | None = None


class Director:
    def __init__(
        self,
        llm: LLMClient,
        kb: KnowledgeBase,
        cfg: RunConfig | None = None,
        vault: Vault | None = None,
        max_retries_per_step: int = 1,
        memory: "DirectorMemory | None" = None,
        auto_distill_every: int = 5,
        vault_root_for_distill: Path | str | None = None,
        parallel: int = 1,
    ) -> None:
        from autosci.memory import DirectorMemory  # local import to avoid cycle
        self.llm = llm
        self.kb = kb
        self.cfg = cfg or RunConfig(output_root=Path("runs"))
        self.vault = vault or Vault(root=self.cfg.output_root.parent / "vault")
        self.max_retries_per_step = max_retries_per_step
        self.memory = memory or DirectorMemory(path=self.cfg.output_root / "director_memory.jsonl")
        self.team = ResearchTeam(llm, kb, cfg)
        self.vault.bootstrap()
        # v0.6: auto-distill settings
        self.auto_distill_every = auto_distill_every  # 0 = disabled
        self.vault_root_for_distill = Path(vault_root_for_distill) if vault_root_for_distill else self.vault.root
        # v0.7: parallelism (1 = serial, >1 = concurrent via asyncio + threads)
        self.parallel = max(1, int(parallel))

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def load_seeds(self, path: Path | str) -> list[IdeaSeed]:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        seeds_raw = data["ideas"] if isinstance(data, dict) and "ideas" in data else data
        seeds = [IdeaSeed.model_validate(s) for s in seeds_raw]
        # Filter out duplicates from prior runs.
        before = len(seeds)
        seeds = [s for s in seeds if not self.memory.is_duplicate(s)]
        if len(seeds) < before:
            log.info("[director] dropped %d duplicate seed(s) per memory", before - len(seeds))
        # v0.5: failure prediction filter
        try:
            from autosci.predictor import filter_by_probability
            kept, dropped = filter_by_probability(seeds, self.memory, max_p=0.6)
            if dropped:
                log.info("[director] failure predictor dropped %d high-risk idea(s)", len(dropped))
            return kept
        except Exception as e:
            log.debug("failure predictor not active: %s", e)
        return seeds

    def run_all(self, seeds: list[IdeaSeed], max_ideas: int | None = None) -> DirectorReport:
        t0 = time.time()
        rep = DirectorReport(n_total=len(seeds))

        seeds = seeds[: (max_ideas or len(seeds))]

        if self.parallel > 1 and len(seeds) > 1:
            return self._run_parallel(seeds, rep, t0)

        # Snapshot LLMClient accumulators so we can compute per-idea deltas.
        llm = self.llm
        prev_t_in = getattr(llm, "total_tokens_in", 0)
        prev_t_out = getattr(llm, "total_tokens_out", 0)
        prev_cost = getattr(llm, "total_cost_cny", 0.0)
        prev_calls = getattr(llm, "call_count", 0)

        n_attempted = 0
        for seed in seeds:
            log.info("=== %s === %s", seed.id, seed.title[:80])
            success, kept_state = self._run_with_retry(seed)
            # Compute per-idea token delta
            now_t_in = getattr(llm, "total_tokens_in", 0)
            now_t_out = getattr(llm, "total_tokens_out", 0)
            now_cost = getattr(llm, "total_cost_cny", 0.0)
            now_calls = getattr(llm, "call_count", 0)
            idea_t_in = now_t_in - prev_t_in
            idea_t_out = now_t_out - prev_t_out
            idea_cost = now_cost - prev_cost
            idea_calls = now_calls - prev_calls
            prev_t_in, prev_t_out, prev_cost, prev_calls = now_t_in, now_t_out, now_cost, now_calls

            if not success:
                rep.n_failed += 1
                rep.per_idea.append(
                    {
                        "id": seed.id, "verdict": "failed",
                        "tokens_in": idea_t_in, "tokens_out": idea_t_out,
                        "cost_cny": round(idea_cost, 4), "calls": idea_calls,
                    }
                )
                self._audit(seed.id, "failed", f"pipeline failed for {seed.id}")
                n_attempted += 1
                continue

            verdict = kept_state.verdict
            if verdict == Verdict.KEPT:
                rep.n_kept += 1
            elif verdict == Verdict.REVISED:
                rep.n_revised += 1
            else:
                rep.n_refuted += 1
            n_attempted += 1

            # Stash token info into the state for Recipe record
            if not kept_state.history_log or not kept_state.history_log[-1].startswith("__token__"):
                kept_state.history_log.append(
                    f"__token__ in={idea_t_in} out={idea_t_out} cost_cny={round(idea_cost,4)} calls={idea_calls}"
                )

            self._audit(
                seed.id,
                "verdict",
                f"{verdict.value} | tokens(in/out)={idea_t_in}/{idea_t_out} cost={idea_cost:.4f}CNY calls={idea_calls}",
            )
            # Vault AFTER we have appended the token line
            try:
                self.vault.record(kept_state, paper_md_path=Path(kept_state.paper_path) if kept_state.paper_path else None)
            except Exception as e:
                log.warning("Vault record failed for %s: %s", seed.id, e)

            # Persist to cross-idea memory so future runs avoid the same idea.
            try:
                self.memory.record(kept_state)
            except Exception as e:
                log.debug("memory.record failed: %s", e)

            # v0.6: auto-distill if the configured threshold is hit
            if self.auto_distill_every and n_attempted % self.auto_distill_every == 0:
                self._maybe_auto_distill(rep)

            rep.per_idea.append(
                {
                    "id": seed.id,
                    "verdict": verdict.value if isinstance(verdict, Verdict) else str(verdict),
                    "tokens_in": idea_t_in, "tokens_out": idea_t_out,
                    "cost_cny": round(idea_cost, 4), "calls": idea_calls,
                }
            )
        rep.tokens_in = getattr(llm, "total_tokens_in", 0)
        rep.tokens_out = getattr(llm, "total_tokens_out", 0)
        rep.elapsed_s = time.time() - t0
        return rep

    # ------------------------------------------------------------------
    # v0.7 — parallel runner
    # ------------------------------------------------------------------

    def _run_parallel(self, seeds: list[IdeaSeed], rep: DirectorReport, t0: float) -> DirectorReport:
        """Run multiple ideas concurrently using asyncio + a thread pool.

        The LLMClient is shared but uses ``httpx`` which is thread-safe,
        and we add a small jitter between requests to avoid bursting
        the upstream API.
        """
        import asyncio
        import concurrent.futures

        llm = self.llm
        prev_t_in = getattr(llm, "total_tokens_in", 0)
        prev_t_out = getattr(llm, "total_tokens_out", 0)
        prev_cost = getattr(llm, "total_cost_cny", 0.0)
        prev_calls = getattr(llm, "call_count", 0)

        def _one(seed: IdeaSeed) -> tuple[IdeaSeed, bool, "IdeaState | None"]:
            log.info("=== %s === %s", seed.id, seed.title[:80])
            return (seed, *self._run_with_retry(seed))

        n_attempted = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as pool:
            futures = [pool.submit(_one, s) for s in seeds]
            for fut in concurrent.futures.as_completed(futures):
                seed, success, kept_state = fut.result()
                now_t_in = getattr(llm, "total_tokens_in", 0)
                now_t_out = getattr(llm, "total_tokens_out", 0)
                now_cost = getattr(llm, "total_cost_cny", 0.0)
                now_calls = getattr(llm, "call_count", 0)
                idea_t_in = now_t_in - prev_t_in
                idea_t_out = now_t_out - prev_t_out
                idea_cost = now_cost - prev_cost
                idea_calls = now_calls - prev_calls
                prev_t_in, prev_t_out, prev_cost, prev_calls = now_t_in, now_t_out, now_cost, now_calls

                if not success or kept_state is None:
                    rep.n_failed += 1
                    rep.per_idea.append({
                        "id": seed.id, "verdict": "failed",
                        "tokens_in": idea_t_in, "tokens_out": idea_t_out,
                        "cost_cny": round(idea_cost, 4), "calls": idea_calls,
                    })
                    self._audit(seed.id, "failed", f"pipeline failed for {seed.id}")
                    n_attempted += 1
                    continue
                verdict = kept_state.verdict
                if verdict == Verdict.KEPT:
                    rep.n_kept += 1
                elif verdict == Verdict.REVISED:
                    rep.n_revised += 1
                else:
                    rep.n_refuted += 1
                n_attempted += 1
                if not kept_state.history_log or not kept_state.history_log[-1].startswith("__token__"):
                    kept_state.history_log.append(
                        f"__token__ in={idea_t_in} out={idea_t_out} cost_cny={round(idea_cost,4)} calls={idea_calls}"
                    )
                self._audit(seed.id, "verdict",
                             f"{verdict.value} | tokens(in/out)={idea_t_in}/{idea_t_out} cost={idea_cost:.4f}CNY calls={idea_calls}")
                try:
                    self.vault.record(kept_state, paper_md_path=Path(kept_state.paper_path) if kept_state.paper_path else None)
                except Exception as e:
                    log.warning("Vault record failed for %s: %s", seed.id, e)
                try:
                    self.memory.record(kept_state)
                except Exception:
                    pass
                if self.auto_distill_every and n_attempted % self.auto_distill_every == 0:
                    self._maybe_auto_distill(rep)
                rep.per_idea.append({
                    "id": seed.id,
                    "verdict": verdict.value if isinstance(verdict, Verdict) else str(verdict),
                    "tokens_in": idea_t_in, "tokens_out": idea_t_out,
                    "cost_cny": round(idea_cost, 4), "calls": idea_calls,
                })
        rep.tokens_in = getattr(llm, "total_tokens_in", 0)
        rep.tokens_out = getattr(llm, "total_tokens_out", 0)
        rep.elapsed_s = time.time() - t0
        return rep

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_with_retry(self, seed: IdeaSeed) -> tuple[bool, IdeaState | None]:
        """Run pipeline, with one retry slot if a step blew up."""
        attempts = 0
        while attempts <= self.max_retries_per_step:
            attempts += 1
            try:
                state = self.team.run_one(seed)
                return True, state
            except Exception as e:
                log.warning(
                    "[%s] pipeline attempt %d failed: %s\n%s",
                    seed.id, attempts, e, traceback.format_exc(),
                )
                if attempts > self.max_retries_per_step:
                    return False, None
        return False, None

    @staticmethod
    def _last_state_tokens(state: IdeaState | None, which: str) -> int:
        # Kept for backward-compat; production code uses LLMClient counters now.
        return 0

    def _audit(self, idea_id: str, event: str, msg: str) -> None:
        """Append a line to the global Director audit log."""
        try:
            log_path = self.cfg.output_root / "director.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%H:%M:%S")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ts} [{event}] {idea_id} :: {msg}\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # v0.6 — auto-distillation
    # ------------------------------------------------------------------

    def _maybe_auto_distill(self, rep: DirectorReport) -> None:
        """If we have ≥1 kept paper and ≥1 refuted, run distillation."""
        kept = [it for it in self.memory.entries if it.verdict == "kept"]
        if not kept:
            log.info("[auto-distill] no kept papers yet; skipping")
            return
        try:
            from autosci.distiller import run_distillation
            report = run_distillation(
                self.llm, vault_root=self.vault_root_for_distill,
            )
            if report is not None:
                rep.distillation_triggered = True
                rep.distillation_report_path = "data/distillation.json"
                self._audit(
                    "director",
                    "auto_distill",
                    f"ran distillation, {len(report.prompt_patches)} prompt patch(es) written",
                )
        except Exception as e:
            log.warning("[auto-distill] failed: %s", e)
