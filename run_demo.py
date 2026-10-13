"""AutoSCI — top-level demo entry point.

Usage::

    # 1. Make sure your .env exists (copy from .env.example).
    cp .env.example .env

    # 2. Default: 5 seed ideas, write to ./vault
    python run_demo.py

    # 3. Limit + custom output
    python run_demo.py --max-ideas 2 --vault-dir ./my_vault --runs-dir ./my_runs
"""
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="AutoSCI demo runner")
    parser.add_argument("--seeds", default="data/seed_ideas.json")
    parser.add_argument("--digest-dir", default=None,
                        help="directory with *.digest.md; default = $AUTOSCI_PAPER_ROOT/_digest")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--vault-dir", default="vault")
    parser.add_argument("--max-ideas", type=int, default=None)
    parser.add_argument("--kb-topk", type=int, default=8)
    args = parser.parse_args(argv)

    _setup_logging()
    log = logging.getLogger("autosci.main")

    from autosci.director import Director
    from autosci.kb import KnowledgeBase
    from autosci.llm import LLMClient
    from autosci.pipeline import RunConfig
    from autosci.vault import Vault

    digest_dir = Path(args.digest_dir) if args.digest_dir else (
        Path(__file__).parent / ".." / ".." / "paper" / "_digest"
    )
    # fallback to env / default
    if not digest_dir.exists():
        env_root = Path(
            __import__("os").environ.get(
                "AUTOSCI_PAPER_ROOT",
                r"D:\shawn\paper",
            )
        )
        digest_dir = env_root / "_digest"

    log.info("Digest dir: %s", digest_dir)
    log.info("Runs dir:   %s", args.runs_dir)
    log.info("Vault dir:  %s", args.vault_dir)

    llm = LLMClient()
    if not llm.api_key:
        log.error("AGNES_API_KEY missing — set it in .env")
        return 1

    kb = KnowledgeBase.from_digest_dir(digest_dir)
    cfg = RunConfig(output_root=Path(args.runs_dir), kb_topk=args.kb_topk)
    vault = Vault(root=Path(args.vault_dir))

    director = Director(llm=llm, kb=kb, cfg=cfg, vault=vault)
    seeds = director.load_seeds(Path(args.seeds))
    log.info("Loaded %d seed ideas.", len(seeds))

    t0 = time.time()
    report = director.run_all(seeds, max_ideas=args.max_ideas)
    elapsed = time.time() - t0

    summary = {
        "elapsed_s": round(elapsed, 1),
        "n_total": report.n_total,
        "n_kept": report.n_kept,
        "n_revised": report.n_revised,
        "n_refuted": report.n_refuted,
        "n_failed": report.n_failed,
        "per_idea": report.per_idea,
    }
    out = Path(args.runs_dir) / "director_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("=== Director done in %.1fs ===", elapsed)
    log.info("kept=%d revised=%d refuted=%d failed=%d",
             report.n_kept, report.n_revised, report.n_refuted, report.n_failed)
    log.info("Summary written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
