"""AutoSCI — CLI dashboard (v0.6).

One-shot, non-blocking report on the current state of an AutoSCI run.

Shows (in plain text, no external deps):
    - active run
    - kept / revised / refuted / failed counts (from vault manifest)
    - cumulative token spend + cost
    - recent director.log entries
    - whether a distillation patch is loaded

Usage:
    python run_dashboard.py
    python run_dashboard.py --watch 5    # refresh every 5s
    python run_dashboard.py --runs-dir runs --vault-dir vault
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


def _read_text(p: Path, max_chars: int = 4000) -> str:
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def _read_json(p: Path) -> dict | list | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _format_size(num: int) -> str:
    for unit in ("", "K", "M", "B"):
        if num < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.1f}B"


def render_snapshot(runs_dir: Path, vault_dir: Path, distill_path: Path) -> str:
    out: list[str] = []
    out.append("=" * 64)
    out.append(f"  AutoSCI Dashboard  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    out.append("=" * 64)

    # 1. Manifest
    manifest = _read_json(vault_dir / "manifest.json") or {"items": []}
    items = manifest.get("items") or []
    verdict_counts = Counter(it.get("verdict", "?") for it in items)
    out.append("")
    out.append("## Vault")
    out.append(f"  total vaulted : {len(items)}")
    for v in ("kept", "revised", "refuted", "pending"):
        out.append(f"  {v:10s} : {verdict_counts.get(v, 0)}")
    if items:
        last = items[-1]
        out.append(f"  last entry     : {last.get('id')} [{last.get('verdict')}] at {last.get('updated_at')}")

    # 2. Latest run summary
    summary = _read_json(runs_dir / "director_summary.json") or {}
    if summary:
        out.append("")
        out.append("## Last run summary")
        out.append(f"  elapsed       : {summary.get('elapsed_s', 0):.1f}s")
        out.append(f"  tokens(in)    : {_format_size(summary.get('n_total', 0) or 0)}  total: {summary.get('tokens_in', 0):,}")
        out.append(f"  tokens(out)   : {summary.get('tokens_out', 0):,}")
        n_kept = summary.get('n_kept', 0)
        n_rev = summary.get('n_revised', 0)
        n_ref = summary.get('n_refuted', 0)
        n_fail = summary.get('n_failed', 0)
        n_tot = max(1, summary.get('n_total', 1))
        out.append(f"  kept={n_kept}  revised={n_rev}  refuted={n_ref}  failed={n_fail}  (n_total={n_tot})")
        if summary.get("distillation_triggered"):
            out.append(f"  distillation  : triggered (→ {summary.get('distillation_report_path', 'data/distillation.json')})")

    # 3. Director log tail
    log = _read_text(runs_dir / "director.log", max_chars=3000)
    if log.strip():
        lines = log.strip().splitlines()[-8:]
        out.append("")
        out.append("## Director log (last 8)")
        for ln in lines:
            out.append("  " + ln)

    # 4. Distillation
    out.append("")
    out.append("## Distillation")
    d = _read_json(distill_path) or {}
    if not d:
        out.append("  (no distillation.json yet)")
    else:
        cs = d.get("common_strengths") or []
        cw = d.get("common_weaknesses_in_revised") or []
        pp = d.get("prompt_patches") or {}
        out.append(f"  strengths : {len(cs)} | weaknesses : {len(cw)} | patches : {len(pp)}")
        if pp:
            out.append("  patches (first 3):")
            for k, v in list(pp.items())[:3]:
                out.append(f"    {k}: {v[:80]}")

    # 5. Recent vault entries
    if items:
        out.append("")
        out.append("## Recent vault entries (last 5)")
        for it in items[-5:]:
            title = (it.get("title") or it.get("id") or "?")[:60]
            out.append(f"  [{it.get('verdict'):8s}] {it.get('id')[:24]:24s}  {title}")

    out.append("")
    out.append("=" * 64)
    return "\n".join(out)


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
            if args.watch > 0:
                # simple clear-the-screen
                sys.stdout.write("\x1b[2J\x1b[H")
            print(render_snapshot(runs, vault, distill))
            if args.watch <= 0:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
