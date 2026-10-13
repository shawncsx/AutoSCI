"""AutoSCI — Claude-Code-style interactive REPL (v0.10).

Drop the user into a prompt like::

    ❯ /help
    ❯ /run --ideas 5
    ❯ /status
    ❯ /mass --per-query 30

Slash-commands cover the same actions as the standalone runners, plus
inspect the on-disk state.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from autosci.rich_console import (
    HELP_COMMANDS,
    banner,
    console,
    corpus_table,
    error,
    help_table,
    section,
    status,
    status_panel,
    success,
    tool_call,
    tool_result,
    verdict_table,
    welcome,
)


COMMANDS = {
    "help", "status", "ideas", "vault", "run", "mass", "demo", "exit", "quit",
}


def _read_json(p: Path) -> dict | list | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def cmd_help(_args: list[str], ctx: dict) -> None:
    section("Commands")
    console.print(help_table())
    console.print()
    console.print("[grey58]Examples:[/]")
    console.print("  /run --ideas 3 --parallel 2")
    console.print("  /mass --per-query 50 --ideas 20")
    console.print("  /status")


def cmd_status(_args: list[str], ctx: dict) -> None:
    section("AutoSCI status")
    summary = _read_json(Path(ctx["runs_dir"]) / "director_summary.json") or {}
    manifest = _read_json(Path(ctx["vault_dir"]) / "manifest.json") or {"items": []}
    items = manifest.get("items") or []

    snap: dict = {}
    if "n_total" in summary:
        snap["n_total"] = summary.get("n_total") or len(items)
    snap["n_kept"] = summary.get("n_kept", 0)
    snap["n_revised"] = summary.get("n_revised", 0)
    snap["n_refuted"] = summary.get("n_refuted", 0)
    snap["tokens_in"] = summary.get("tokens_in", 0) or 0
    snap["tokens_out"] = summary.get("tokens_out", 0) or 0
    snap["cost_cny"] = (snap["tokens_in"] * 0.002 + snap["tokens_out"] * 0.006) / 1000
    if "elapsed_s" in summary:
        snap["elapsed_s"] = summary.get("elapsed_s", 0) or 0
    console.print(status_panel(snap))
    # KB
    by_source: dict[str, int] = {}
    local = Path(r"D:\shawn\paper\_digest")
    if local.exists():
        by_source["local"] = len(list(local.glob("*.digest.md")))
    for d in (Path("data/mass_corpus"), Path("data/openalex_corpus"),
              Path("data/biorxiv_corpus"), Path("data/pubmed_corpus"),
              Path("data/semanticscholar_corpus")):
        if d.exists():
            n = len(list(d.glob("*.digest.md")))
            if n:
                by_source[d.name] = n
    if by_source:
        section("Knowledge base")
        console.print(corpus_table(by_source))


def cmd_ideas(args: list[str], ctx: dict) -> None:
    section("Idea backlog")
    seeds_path = Path(ctx["seeds_path"])
    if not seeds_path.exists():
        error(f"Seed file not found: {seeds_path}")
        return
    data = json.loads(seeds_path.read_text(encoding="utf-8"))
    seeds = data.get("ideas") if isinstance(data, dict) and "ideas" in data else data
    from autosci.rich_console import ideas_tree
    tree = ideas_tree([
        {"id": s.get("id", ""), "title": s.get("title", ""),
         "modalities": s.get("modalities", []), "priority": s.get("priority", 5)}
        for s in seeds
    ])
    console.print(tree)


def cmd_vault(_args: list[str], ctx: dict) -> None:
    section("Vault")
    manifest = _read_json(Path(ctx["vault_dir"]) / "manifest.json") or {"items": []}
    items = manifest.get("items") or []
    if not items:
        console.print("  [grey58](empty vault)[/]")
        return
    rows = []
    for it in items:
        rows.append({
            "id": it.get("id", ""),
            "title": it.get("title", ""),
            "verdict": it.get("verdict", "?"),
            "tokens_in": 0, "tokens_out": 0, "cost_cny": 0, "calls": "",
        })
    console.print(verdict_table(rows))


def _spawn_subprocess(argv: list[str]) -> int:
    """Run an external command in-place so the user sees its live output."""
    import subprocess
    console.print(f"  [grey58]$ {' '.join(argv)}[/]")
    p = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)
    p.wait()
    return p.returncode


def cmd_run(args: list[str], ctx: dict) -> int:
    argv = [sys.executable, "run_demo.py",
            "--seeds", ctx["seeds_path"],
            "--runs-dir", ctx["runs_dir"],
            "--vault-dir", ctx["vault_dir"]]
    # pass through user flags (--ideas → --max-ideas)
    for a in args:
        if a == "--ideas":
            continue
        if a.startswith("--ideas="):
            argv += ["--max-ideas", a.split("=", 1)[1]]
            continue
        if a == "--parallel":
            continue
        if a.startswith("--parallel="):
            argv += ["--max-ideas", a.split("=", 1)[1]]
            continue
        argv.append(a)
    # honour a leading --ideas 3 etc
    if "--ideas" in args:
        i = args.index("--ideas")
        if i + 1 < len(args):
            argv += ["--max-ideas", args[i + 1]]
    return _spawn_subprocess(argv)


def cmd_mass(args: list[str], ctx: dict) -> int:
    argv = [sys.executable, "run_mass.py",
            "--runs-dir", "runs_mass",
            "--vault-dir", "vault_mass"] + list(args)
    return _spawn_subprocess(argv)


def cmd_demo(_args: list[str], ctx: dict) -> int:
    return _spawn_subprocess(
        [sys.executable, "run_demo.py",
         "--seeds", ctx["seeds_path"],
         "--max-ideas", "5",
         "--runs-dir", "runs",
         "--vault-dir", "vault"]
    )


def cmd_exit(_args: list[str], ctx: dict) -> None:
    raise SystemExit(0)


HANDLERS = {
    "help": cmd_help,
    "status": cmd_status,
    "ideas": cmd_ideas,
    "vault": cmd_vault,
    "run": cmd_run,
    "mass": cmd_mass,
    "demo": cmd_demo,
    "exit": cmd_exit,
    "quit": cmd_exit,
}


def _read_prompt() -> str:
    """Read a single line from stdin, echoing '❯ ' manually."""
    try:
        console.print(" ", end="")
        sys.stdout.flush()
        return input("❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoSCI interactive CLI")
    parser.add_argument("--seeds", default="data/seed_ideas.json")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--vault-dir", default="vault")
    parser.add_argument("--no-banner", action="store_true", help="skip welcome banner")
    args = parser.parse_args(argv)

    ctx = {
        "seeds_path": args.seeds,
        "runs_dir": args.runs_dir,
        "vault_dir": args.vault_dir,
    }

    if not args.no_banner:
        banner()
        # quick welcome with current snapshot
        manifest = _read_json(Path(args.vault_dir) / "manifest.json") or {"items": []}
        items = manifest.get("items") or []
        from collections import Counter
        vc = Counter(it.get("verdict", "?") for it in items)
        # count local KB
        local = Path(r"D:\shawn\paper\_digest")
        kb = len(list(local.glob("*.digest.md"))) if local.exists() else 0
        for d in (Path("data/mass_corpus"), Path("data/openalex_corpus"),
                  Path("data/biorxiv_corpus"), Path("data/pubmed_corpus"),
                  Path("data/semanticscholar_corpus")):
            if d.exists():
                kb += len(list(d.glob("*.digest.md")))
        welcome(
            kb_size=kb,
            vault_kept=vc.get("kept", 0),
            vault_revised=vc.get("revised", 0),
            vault_refuted=vc.get("refuted", 0),
            pending_ideas=0,
            model="(see .env AGNES_MODEL)",
        )
        console.print("[grey58]Tip: type /help to see commands.[/]")
        console.print()

    # Read-only commands: no spinner, just print.
    READ_ONLY = {"help", "status", "ideas", "vault", "exit", "quit"}
    while True:
        try:
            line = _read_prompt()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            error(f"Unknown input: {line!r}  (try /help)")
            continue
        parts = shlex.split(line[1:])
        if not parts:
            continue
        cmd, cargs = parts[0], parts[1:]
        if cmd not in HANDLERS:
            error(f"Unknown command: /{cmd}  (try /help)")
            continue
        try:
            if cmd in READ_ONLY:
                ret = HANDLERS[cmd](cargs, ctx)
            else:
                with status(f"Running /{cmd}…"):
                    ret = HANDLERS[cmd](cargs, ctx)
                if ret is None or ret == 0:
                    success(f"/{cmd} done")
                else:
                    console.print(f"  [red]/{cmd} returned {ret}[/]")
        except SystemExit:
            console.print()
            success("Goodbye.")
            break
        except KeyboardInterrupt:
            console.print()
            console.print("  [grey58](interrupted — back to prompt)[/]")
        except Exception as e:
            error(f"/{cmd} failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
