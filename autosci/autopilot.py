"""AutoSCI — Autopilot: continuous, self-fueling run.

When launched as ``autopilot`` mode, AutoSCI:

    1. Loads seeds from a file (if given) and clears the in-flight backlog
    2. Runs each idea through the pipeline + Director verdicts
    3. After exhausting seeds, calls IdeaMiner to propose N more ideas
    4. Repeats indefinitely or until a token/iteration budget is hit
    5. Optionally pulls GitHub repos on each "resupply" cycle

This is what makes AutoSCI truly "user-starts-it-then-walks-away".
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from autosci.director import Director, DirectorReport
from autosci.github_ingestor import RepoSpec, ingest_repos
from autosci.agents.idea_miner import compute_axis_signals, dedupe_by_title, propose_ideas
from autosci.kb import KnowledgeBase
from autosci.llm import LLMClient
from autosci.pipeline import RunConfig
from autosci.schemas import IdeaSeed, IdeaState, Verdict
from autosci.vault import Vault


log = logging.getLogger("autosci.autopilot")


@dataclass
class AutopilotConfig:
    seed_path: Path | None = None
    max_seeds_per_cycle: int = 5
    max_cycles: int | None = None  # None = forever
    mine_extra_per_cycle: int = 3
    resupply_kb_each_cycle: bool = True
    github_repos: list[RepoSpec] = field(default_factory=list)
    github_digest_dir: Path = Path("data/github_digests")
    sleep_between_ideas_s: float = 0.0
    keep_alive_path: Path = Path("runs/autopilot_state.json")


@dataclass
class AutopilotReport:
    cycles: int = 0
    ideas_attempted: int = 0
    ideas_kept: int = 0
    ideas_revised: int = 0
    ideas_refuted: int = 0
    ideas_mined: int = 0
    elapsed_s: float = 0.0
    last_kept_ids: list[str] = field(default_factory=list)
    last_refuted_ids: list[str] = field(default_factory=list)


class Autopilot:
    """Continuous runner: seeds + Director + IdeaMiner loop."""

    def __init__(
        self,
        llm: LLMClient,
        kb: KnowledgeBase,
        director: Director,
        cfg: AutopilotConfig,
    ) -> None:
        self.llm = llm
        self.kb = kb
        self.director = director
        self.cfg = cfg
        self.cfg.keep_alive_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self) -> AutopilotReport:
        t0 = time.time()
        rep = AutopilotReport()
        cycle = 0

        # Pre-load seeds if any
        seed_pool = self._initial_seeds()
        seen_titles: set[str] = {(s.title or "").strip().lower() for s in seed_pool}

        while True:
            cycle += 1
            if self.cfg.max_cycles is not None and cycle > self.cfg.max_cycles:
                break

            log.info("==== AutoSCI autopilot cycle %d (pool=%d) ====", cycle, len(seed_pool))

            # Optionally refresh KB from GitHub each cycle
            if self.cfg.resupply_kb_each_cycle and self.cfg.github_repos:
                self._resupply_kb()

            # Pick the next batch
            batch = seed_pool[: self.cfg.max_seeds_per_cycle]
            seed_pool = seed_pool[self.cfg.max_seeds_per_cycle:]

            if not batch:
                log.info("pool exhausted; mining %d new ideas", self.cfg.mine_extra_per_cycle)
                new_ideas = propose_ideas(
                    self.kb,
                    self.llm,
                    n=self.cfg.mine_extra_per_cycle,
                    existing_titles=list(seen_titles),
                )
                new_ideas = dedupe_by_title(new_ideas)
                batch = new_ideas
                rep.ideas_mined += len(new_ideas)
                for s in batch:
                    seen_titles.add((s.title or "").strip().lower())

            # Run Director on the batch (treat each as its own "cycle" report)
            sub = self.director.run_all(batch)
            rep.ideas_attempted += sub.n_total
            rep.ideas_kept += sub.n_kept
            rep.ideas_revised += sub.n_revised
            rep.ideas_refuted += sub.n_refuted
            for row in sub.per_idea:
                if row["verdict"] in ("kept", "revised"):
                    rep.last_kept_ids.append(row["id"])
                elif row["verdict"] == "refuted":
                    rep.last_refuted_ids.append(row["id"])

            self._save_state(rep)
            if self.cfg.sleep_between_ideas_s > 0:
                time.sleep(self.cfg.sleep_between_ideas_s)

            # If batch was empty (nothing to mine), gracefully stop
            if not batch:
                log.warning("No ideas available — autopilot stopping to avoid hot-loop.")
                break

        rep.cycles = cycle
        rep.elapsed_s = time.time() - t0
        return rep

    # ------------------------------------------------------------------
    def _initial_seeds(self) -> list[IdeaSeed]:
        if self.cfg.seed_path and self.cfg.seed_path.exists():
            return self.director.load_seeds(self.cfg.seed_path)
        return []

    def _resupply_kb(self) -> None:
        try:
            files = ingest_repos(self.cfg.github_digest_dir, repos=self.cfg.github_repos)
            if files:
                new_kb = KnowledgeBase.from_digest_dir(self.cfg.github_digest_dir)
                # Preserve existing docs + add new docs.
                existing = {d.doc_id for d in self.kb.docs}
                new_docs = [d for d in new_kb.docs if d.doc_id not in existing]
                self.kb.docs.extend(new_docs)
                self.kb._bm25 = None  # force reindex
                log.info("resupplied KB with %d new github docs", len(new_docs))
        except Exception as e:
            log.warning("KB resupply failed: %s", e)

    def _save_state(self, rep: AutopilotReport) -> None:
        try:
            self.cfg.keep_alive_path.write_text(
                json.dumps(
                    {
                        "cycles": rep.cycles,
                        "ideas_attempted": rep.ideas_attempted,
                        "ideas_kept": rep.ideas_kept,
                        "ideas_revised": rep.ideas_revised,
                        "ideas_refuted": rep.ideas_refuted,
                        "ideas_mined": rep.ideas_mined,
                        "elapsed_s": round(rep.elapsed_s, 1),
                        "last_kept_ids": rep.last_kept_ids,
                        "last_refuted_ids": rep.last_refuted_ids,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Failed to save autopilot state: %s", e)
