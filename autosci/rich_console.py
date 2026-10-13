"""AutoSCI — Claude-Code-style rich output (v0.10).

This module is the single source of truth for "what AutoSCI looks like
in the terminal". It exposes small, focused helpers that produce the
exact visual elements users expect from a modern AI CLI:

    - banner / welcome panel
    - tool_call + tool_result (the "● Read foo.py ⎿ ..." pattern)
    - status spinner (the "⠋ Running..." pattern)
    - markdown rendering (sections, lists, code, headers)
    - syntax-highlighted code
    - verdict tables (kept / revised / refuted with colour)
    - live status bar (idea + tokens + cost, updates in-place)
    - ideas tree (auto-mined ideas grouped by modality)
    - error / warning / success boxes

All of these can be composed; no helper assumes the others.

Why rich?  It's the Python equivalent of React + Ink + Chalk that
Claude Code itself uses (their stack is TypeScript + Bun + React + Ink
+ Chalk). rich gives us the same primitives in pure Python: panels,
syntax highlighting, spinners, progress bars, live updates, trees.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.box import HEAVY, ROUNDED, SIMPLE
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.spinner import Spinner
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


# ---------------------------------------------------------------------------
# Module-level singleton console (force_terminal respects NO_COLOR / dumb tty)
# ---------------------------------------------------------------------------

_console = Console(
    force_terminal=True,
    legacy_windows=False,
    soft_wrap=False,
    highlight=False,
    theme=None,
)


# ---------------------------------------------------------------------------
# Brand colours (warm gray + sky blue + emerald + amber + rose)
# ---------------------------------------------------------------------------

C_PRIMARY = "bright_cyan"
C_ACCENT = "deep_sky_blue1"
C_DIM = "grey58"
C_OK = "bright_green"
C_WARN = "yellow1"
C_BAD = "bright_red"
C_HINT = "magenta"


VERDICT_STYLES: dict[str, str] = {
    "kept": "bold white on bright_green",
    "revised": "bold black on yellow1",
    "refuted": "bold white on bright_red",
    "pending": "bold white on grey50",
    "failed": "bold white on bright_red",
}


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

LOGO_LINES = [
    "    ___        __  ____     ___   _____ _____      ",
    "   /   |  ____/  |/  /_   /   | / ___// ____/      ",
    "  / /| | / __  /|_/ /   / /| | \\__ \\/ /           ",
    " / ___ |/ /_/ /  / /   / ___ |___/ / /___         ",
    "/_/  |_/\\__,_/  /_/   /_/  |_/____/\\____/  v0.10",
]


def banner(
    version: str = "0.10",
    title: str = "AutoSCI — Distributed Research-Agent Pipeline",
    subtitle: str = "Reproducing OpenAI's math-project workflow, retargeted at microbiome AI.",
) -> None:
    """Print the AutoSCI ASCII-art banner, version, and tagline."""
    c = _console
    c.print()
    c.print(
        Panel(
            Group(
                Text("\n".join(LOGO_LINES), style=f"bold {C_ACCENT}"),
                Padding(
                    Text(subtitle, style=C_DIM, justify="center"),
                    (0, 0, 0, 0),
                ),
            ),
            title=f"[{C_PRIMARY}]AutoSCI v{version}[/]",
            subtitle=f"[{C_DIM}]v{version} · MIT · Microbiome AI[/]",
            border_style=C_PRIMARY,
            box=HEAVY,
            padding=(0, 2),
        )
    )
    c.print()


def welcome(
    *,
    kb_size: int = 0,
    vault_kept: int = 0,
    vault_revised: int = 0,
    vault_refuted: int = 0,
    pending_ideas: int = 0,
    model: str = "",
    next_cmd: str = "Type /help for commands.",
) -> None:
    """Welcome card shown on startup."""
    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column(justify="left", style=C_DIM, width=14)
    table.add_column(justify="left")
    table.add_row("KB papers", f"[bold {C_PRIMARY}]{kb_size:,}[/]")
    table.add_row("Kept", f"[{C_OK}]{vault_kept}[/]")
    table.add_row("Revised", f"[{C_WARN}]{vault_revised}[/]")
    table.add_row("Refuted", f"[{C_BAD}]{vault_refuted}[/]")
    table.add_row("Pending ideas", f"[{C_HINT}]{pending_ideas}[/]")
    if model:
        table.add_row("Model", f"[{C_DIM}]{model}[/]")
    panel = Panel(
        table,
        title=f"[bold {C_PRIMARY}] Welcome to AutoSCI [/]",
        border_style=C_ACCENT,
        box=ROUNDED,
        padding=(0, 1),
    )
    _console.print(panel)
    _console.print(f"  [{C_DIM}]{next_cmd}[/]")
    _console.print()


# ---------------------------------------------------------------------------
# Tool-call (Claude Code's "● Read foo.py" pattern)
# ---------------------------------------------------------------------------

def tool_call(name: str, summary: str = "") -> None:
    """Print a tool-call marker:  ● IdeaCloser.run  graph model for OTU clustering

    Matches Claude Code's visual style.
    """
    body = Text()
    body.append("● ", style=f"bold {C_PRIMARY}")
    body.append(name, style="bold")
    if summary:
        body.append("  ", style=C_DIM)
        body.append(summary, style=C_DIM)
    _console.print(body)


def tool_result(text: str, *, max_lines: int = 12, indent: int = 3) -> None:
    """Print a tool result with the ⎿ marker and indent (Claude Code style)."""
    lines = text.splitlines() or [""]
    if len(lines) > max_lines:
        shown = lines[:max_lines]
        more = f"  … ({len(lines) - max_lines} more lines)"
        shown.append(more)
    else:
        shown = lines
    first = True
    for ln in shown:
        if first:
            _console.print(
                Text("  ⎿ ", style=f"bold {C_ACCENT}")
                + Text(ln, style=C_DIM)
            )
            first = False
        else:
            _console.print(
                Text("    ", style=C_DIM) + Text(ln, style=C_DIM)
            )


# ---------------------------------------------------------------------------
# Spinner / status (the "⠋ Running…" pattern)
# ---------------------------------------------------------------------------

@contextmanager
def status(message: str, *, spinner: str = "dots"):
    """Block-scoped spinner; usage:

        with status("Running 5 ideas through the pipeline"):
            director.run_all(...)

    Equivalent to `with console.status(...) as s: s.update(...)`.
    """
    with _console.status(f"[{C_PRIMARY}]{message}[/]", spinner=spinner) as st:
        yield st


@contextmanager
def progress(total: int, *, label: str = "Working"):
    """Multi-step progress bar (rich.progress)."""
    prog = Progress(
        SpinnerColumn(style=C_PRIMARY),
        TextColumn(f"[{C_PRIMARY}]{{task.description}}[/]"),
        BarColumn(bar_width=None, complete_style=C_OK, finished_style=C_OK),
        TextColumn("[{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    )
    with prog:
        task = prog.add_task(label, total=total)
        yield prog, task


# ---------------------------------------------------------------------------
# Markdown / code rendering
# ---------------------------------------------------------------------------

def markdown(text: str) -> None:
    _console.print(Markdown(text))


def code(text: str, language: str = "python") -> None:
    _console.print(Syntax(text, language, theme="monokai", line_numbers=False, word_wrap=True))


def section(title: str) -> None:
    _console.rule(f"[{C_PRIMARY}]{title}[/]")


def separator() -> None:
    _console.print(Rule(style=C_DIM))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def verdict_table(rows: list[dict[str, str]]) -> Table:
    """A nicely formatted verdict-summary table.

    Each row is a dict with: id, title, verdict, tokens, cost, calls.
    """
    t = Table(
        title=None,
        box=SIMPLE,
        show_lines=False,
        header_style=f"bold {C_PRIMARY}",
        expand=True,
    )
    t.add_column("ID", style=C_DIM, no_wrap=True)
    t.add_column("Title", overflow="ellipsis", max_width=42)
    t.add_column("Verdict", justify="center", width=10)
    t.add_column("Tokens", justify="right", style=C_DIM)
    t.add_column("Cost", justify="right", style=C_DIM)
    t.add_column("Calls", justify="right", style=C_DIM)
    for r in rows:
        v = (r.get("verdict") or "").lower()
        v_text = Text(v, style=VERDICT_STYLES.get(v, ""))
        t.add_row(
            r.get("id", "")[:24],
            r.get("title", "")[:42],
            v_text,
            f"{int(r.get('tokens_in', 0)):,} / {int(r.get('tokens_out', 0)):,}",
            f"¥{r.get('cost_cny', 0):.4f}",
            str(r.get("calls", "")),
        )
    return t


def corpus_table(by_source: dict[str, int]) -> Table:
    t = Table(box=SIMPLE, header_style=f"bold {C_PRIMARY}", expand=True)
    t.add_column("Source", style=C_PRIMARY)
    t.add_column("Papers", justify="right", style="bold")
    for src, n in by_source.items():
        t.add_row(src, f"{n:,}")
    return t


# ---------------------------------------------------------------------------
# Trees (for ideas by modality)
# ---------------------------------------------------------------------------

def ideas_tree(ideas: list[dict]) -> Tree:
    """Group ideas by modality in a tree."""
    root = Tree(f"[bold {C_PRIMARY}]Idea backlog ({len(ideas)})[/]")
    groups: dict[str, list[dict]] = {}
    for i in ideas:
        for m in i.get("modalities") or ["(uncategorised)"]:
            groups.setdefault(m, []).append(i)
    for modality, items in sorted(groups.items()):
        branch = root.add(f"[{C_HINT}]{modality}[/]  ([{C_DIM}]{len(items)}[/])")
        for i in items[:8]:
            prio = i.get("priority", "")
            branch.add(
                f"[{C_OK}]{prio}[/]  {i.get('id', '?')[:24]}  "
                f"[{C_DIM}]{i.get('title', '')[:60]}[/]"
            )
        if len(items) > 8:
            branch.add(f"[{C_DIM}]… {len(items) - 8} more[/]")
    return root


# ---------------------------------------------------------------------------
# Status bar (live, updates in place)
# ---------------------------------------------------------------------------

class StatusBar:
    """A small live status line that updates in place.

    Usage::

        bar = StatusBar()
        bar.start()
        bar.update(idea="01_graph_…", verdict="revised", tokens="25K/13K", cost="0.16 CNY")
        ...
        bar.stop()
    """

    def __init__(self) -> None:
        self._live: Optional[Live] = None
        self._state: dict[str, str] = {
            "idea": "—",
            "verdict": "—",
            "tokens": "—",
            "cost": "—",
        }

    def start(self) -> None:
        self._live = Live(self._render(), refresh_per_second=4, console=_console, transient=False)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self, **kwargs: str) -> None:
        self._state.update(kwargs)
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Text:
        s = self._state
        t = Text()
        t.append(" ❯ ", style=f"bold {C_PRIMARY}")
        t.append("idea=", style=C_DIM)
        t.append(s.get("idea", "—")[:30], style="bold")
        t.append("  verdict=", style=C_DIM)
        v = s.get("verdict", "—")
        t.append(v, style=VERDICT_STYLES.get(v.lower(), ""))
        t.append("  tokens=", style=C_DIM)
        t.append(s.get("tokens", "—"), style=C_DIM)
        t.append("  cost=", style=C_DIM)
        t.append(s.get("cost", "—"), style=C_DIM)
        return t


# ---------------------------------------------------------------------------
# Status panel (read-only one-shot, used by `dashboard`)
# ---------------------------------------------------------------------------

def status_panel(snapshot: dict) -> Panel:
    """Render a Panel from a snapshot dict (used by dashboard)."""
    t = Table.grid(expand=True, padding=(0, 2))
    t.add_column(justify="left", style=C_DIM, width=14)
    t.add_column(justify="left")
    if snapshot.get("n_total") is not None:
        t.add_row("ideas", f"[bold]{snapshot.get('n_total', 0)}[/]")
    if snapshot.get("n_kept") is not None:
        t.add_row("kept", f"[{C_OK}]{snapshot.get('n_kept', 0)}[/]")
    if snapshot.get("n_revised") is not None:
        t.add_row("revised", f"[{C_WARN}]{snapshot.get('n_revised', 0)}[/]")
    if snapshot.get("n_refuted") is not None:
        t.add_row("refuted", f"[{C_BAD}]{snapshot.get('n_refuted', 0)}[/]")
    if snapshot.get("tokens_in") is not None:
        t.add_row("tokens(in/out)", f"{snapshot.get('tokens_in', 0):,} / {snapshot.get('tokens_out', 0):,}")
    if snapshot.get("cost_cny") is not None:
        t.add_row("cost", f"¥{snapshot.get('cost_cny', 0):.4f}")
    if snapshot.get("elapsed_s") is not None:
        t.add_row("elapsed", f"{snapshot.get('elapsed_s'):.1f}s")
    return Panel(t, border_style=C_ACCENT, box=ROUNDED, padding=(0, 1))


# ---------------------------------------------------------------------------
# Errors / warnings / success boxes
# ---------------------------------------------------------------------------

def error(message: str) -> None:
    _console.print(Panel(
        Text(message, style="white on " + C_BAD),
        title=f"[bold {C_BAD}]✗ Error[/]",
        border_style=C_BAD,
        box=ROUNDED,
    ))


def warning(message: str) -> None:
    _console.print(Panel(
        Text(message, style="black on " + C_WARN),
        title=f"[bold {C_WARN}]! Warning[/]",
        border_style=C_WARN,
        box=ROUNDED,
    ))


def success(message: str) -> None:
    _console.print(Panel(
        Text(message, style="black on " + C_OK),
        title=f"[bold {C_OK}]✓ Success[/]",
        border_style=C_OK,
        box=ROUNDED,
    ))


# ---------------------------------------------------------------------------
# Help / commands table
# ---------------------------------------------------------------------------

HELP_COMMANDS: list[tuple[str, str]] = [
    ("/help",  "Show this help."),
    ("/status", "Show the current state of AutoSCI (KB / vault / pending)."),
    ("/ideas", "List the current idea backlog, grouped by modality."),
    ("/vault", "List the vault contents (kept / revised / refuted)."),
    ("/run",   "Run Director over the current seed file (or a single idea id)."),
    ("/mass",  "Mass production: arXiv + OpenAlex + bioRxiv + PubMed + S2 + Director."),
    ("/demo",  "Run the v0.1 demo on 5 default seed ideas."),
    ("/exit",  "Exit AutoSCI."),
]


def help_table() -> Table:
    t = Table(box=SIMPLE, header_style=f"bold {C_PRIMARY}", expand=True, title="Commands")
    t.add_column("Command", style=f"bold {C_PRIMARY}", width=10)
    t.add_column("Description", style=C_DIM)
    for cmd, desc in HELP_COMMANDS:
        t.add_row(cmd, desc)
    return t


# ---------------------------------------------------------------------------
# Exposed console for ad-hoc use
# ---------------------------------------------------------------------------

console = _console
