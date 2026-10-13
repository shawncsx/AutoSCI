"""AutoSCI — Claude-Code-style dashboard (v0.10).

Reads the same on-disk state as the v0.6 dashboard (vault/manifest.json,
runs/director.log, director_summary.json, data/distillation.json) and
renders it using rich.

Usage::

    python run_dashboard.py                  # one-shot
    python run_dashboard.py --watch 5        # refresh every 5s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from autosci.rich_console import (
    banner,
    corpus_table,
    section,
    status_panel,
    success,
    verdict_table,
    welcome,
    help_table,
    console,
)


def _read_json(p: Path) -> dict | list | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _read_text(p: Path, max_chars: int = 4000) -> str:
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def _build_snapshot(runs: Path, vault: Path, distill: Path) -> dict:
    snapshot: dict = {}
    manifest = _read_json(vault / "manifest.json") or {"items": []}
    items = manifest.get("items") or []
    snapshot["items"] = items
    verdict_counts = Counter(it.get("verdict", "?") for it in items)
    snapshot["counts"] = verdict_counts
    snapshot["last_entry"] = items[-1] if items else None
    snapshot["summary"] = _read_json(runs / "director_summary.json") or {}
    snapshot["director_log"] = _read_text(runs / "director.log")
    snapshot["distillation"] = _read_json(distill) or {}
    return snapshot


def render_snapshot(snapshot: dict, runs: Path, vault: Path) -> None:
    """Render the rich dashboard."""
    console.clear()
    banner()

    items = snapshot.get("items") or []
    summary = snapshot.get("summary") or {}
    verdict_counts = snapshot.get("counts") or Counter()
    by_source = _corpus_by_source(runs, vault)
    kb_total = sum(by_source.values())

    welcome(
        kb_size=kb_total,
        vault_kept=verdict_counts.get("kept", 0),
        vault_revised=verdict_counts.get("revised", 0),
        vault_refuted=verdict_counts.get("refuted", 0),
        pending_ideas=0,
        model="(see .env AGNES_MODEL)",
        next_cmd="Type /help for commands, or run /demo.",
    )

    if by_source:
        section("Knowledge base")
        console.print(corpus_table(by_source))

    if summary:
        section("Latest run")
        console.print(status_panel({
            "n_total": summary.get("n_total"),
            "n_kept": summary.get("n_kept"),
            "n_revised": summary.get("n_revised"),
            "n_refuted": summary.get("n_refuted"),
            "tokens_in": summary.get("tokens_in", 0),
            "tokens_out": summary.get("tokens_out", 0),
            "cost_cny": summary.get("tokens_in", 0) * 0.002 / 1000 + summary.get("tokens_out", 0) * 0.006 / 1000,
            "elapsed_s": summary.get("elapsed_s"),
        }))

    # Recent vault entries
    if items:
        section(f"Recent vault entries (last {min(5, len(items))})")
        rows = []
        for it in items[-5:]:
            rows.append({
                "id": it.get("id", ""),
                "title": it.get("title", ""),
                "verdict": it.get("verdict", "?"),
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_cny": 0,
                "calls": "",
            })
        console.print(verdict_table(rows))

    # Director log
    log_text = snapshot.get("director_log", "")
    if log_text.strip():
        section("Director log (last 6)")
        for line in log_text.strip().splitlines()[-6:]:
            console.print(f"  [grey58]{line}[/]")

    # Distillation
    d = snapshot.get("distillation") or {}
    if d:
        section("Distillation")
        cs = d.get("common_strengths") or []
        cw = d.get("common_weaknesses_in_revised") or []
        pp = d.get("prompt_patches") or {}
        console.print(f"  strengths={len(cs)}  weaknesses={len(cw)}  patches={len(pp)}")
        for k, v in list(pp.items())[:3]:
            console.print(f"  [cyan]{k}[/]: [grey58]{v[:80]}[/]")
    else:
        section("Distillation")
        console.print("  [grey58](no distillation.json yet — will be created after first [kept] idea)[/]")

    console.print()
    section("Commands")
    console.print(help_table())


def _corpus_by_source(runs: Path, vault: Path) -> dict[str, int]:
    """Count digest files by source under data/* and the local paper dir."""
    counts: dict[str, int] = {}
    # Local paper
    local = Path(r"D:\shawn\paper\_digest")
    if local.exists():
        counts["Local (D:/shawn/paper/_digest)"] = len(list(local.glob("*.digest.md")))
    for d in (
        Path("data/mass_corpus"),
        Path("data/openalex_corpus"),
        Path("data/biorxiv_corpus"),
        Path("data/pubmed_corpus"),
        Path("data/semanticscholar_corpus"),
    ):
        if d.exists():
            n = len(list(d.glob("*.digest.md")))
            if n:
                counts[d.name] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoSCI dashboard")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--vault-dir", default="vault")
    parser.add_argument("--distill", default="data/distillation.json")
    parser.add_argument("--watch", type=int, default=0,
                        help="refresh every N seconds (0 = one-shot)")
    args = parser.parse_args(argv)

    runs = Path(args.runs_dir)
    vault = Path(args.vault_dir)
    distill = Path(args.distill)
    try:
        while True:
            snapshot = _build_snapshot(runs, vault, distill)
            render_snapshot(snapshot, runs, vault)
            if args.watch <= 0:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
