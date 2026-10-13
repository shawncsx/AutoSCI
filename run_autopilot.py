"""AutoSCI — autopilot entry point: zero-touch run.

Usage::

    # Continuous, refills via IdeaMiner after the seeds run out.
    python run_autopilot.py

    # Limited: only 1 cycle, no IdeaMiner, no GitHub.
    python run_autopilot.py --cycles 1 --no-mine --no-resupply
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/seed_ideas.json")
    parser.add_argument("--digest-dir", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--vault-dir", default="vault")
    parser.add_argument("--github-digest-dir", default="data/github_digests")
    parser.add_argument("--max-ideas-per-cycle", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--mine", type=int, default=3, help="ideas to mine per exhausted cycle")
    parser.add_argument("--no-mine", action="store_true", help="disable IdeaMiner resupply")
    parser.add_argument("--no-resupply", action="store_true", help="skip GitHub KB resupply each cycle")
    parser.add_argument("--kb-topk", type=int, default=8)
    args = parser.parse_args(argv)

    _setup_logging()
    log = logging.getLogger("autosci.autopilot.main")

    from autosci.autopilot import Autopilot, AutopilotConfig
    from autosci.github_ingestor import DEFAULT_REPOS
    from autosci.director import Director
    from autosci.kb import KnowledgeBase
    from autosci.llm import LLMClient
    from autosci.pipeline import RunConfig
    from autosci.vault import Vault

    digest_dir = Path(args.digest_dir) if args.digest_dir else (
        Path(r"D:\shawn\paper\_digest")
    )
    log.info("Digest dir: %s", digest_dir)
    log.info("Runs dir:   %s", args.runs_dir)
    log.info("Vault dir:  %s", args.vault_dir)

    llm = LLMClient()
    if not llm.api_key:
        log.error("AGNES_API_KEY missing")
        return 1

    kb = KnowledgeBase.from_digest_dir(digest_dir)
    log.info("KB: %d docs", len(kb.docs))

    cfg = RunConfig(output_root=Path(args.runs_dir), kb_topk=args.kb_topk)
    vault = Vault(root=Path(args.vault_dir))
    director = Director(llm=llm, kb=kb, cfg=cfg, vault=vault)

    auto_cfg = AutopilotConfig(
        seed_path=Path(args.seeds),
        max_seeds_per_cycle=args.max_ideas_per_cycle,
        max_cycles=args.max_cycles,
        mine_extra_per_cycle=(0 if args.no_mine else args.mine),
        resupply_kb_each_cycle=(not args.no_resupply),
        github_repos=DEFAULT_REPOS if not args.no_resupply else [],
        github_digest_dir=Path(args.github_digest_dir),
    )
    autopilot = Autopilot(llm=llm, kb=kb, director=director, cfg=auto_cfg)

    log.info("Starting autopilot. Stop it with Ctrl-C or by killing the process.")
    rep = autopilot.run()

    log.info("=== Autopilot done ===")
    log.info(
        "cycles=%d attempted=%d kept=%d revised=%d refuted=%d mined=%d  (%.1fs)",
        rep.cycles, rep.ideas_attempted,
        rep.ideas_kept, rep.ideas_revised, rep.ideas_refuted, rep.ideas_mined,
        rep.elapsed_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
