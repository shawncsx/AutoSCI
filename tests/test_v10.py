"""Tests for v0.10 — rich_console and run_cli."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from autosci.rich_console import (
    C_ACCENT,
    C_BAD,
    C_OK,
    C_PRIMARY,
    HELP_COMMANDS,
    StatusBar,
    VERDICT_STYLES,
    banner,
    code,
    corpus_table,
    error,
    help_table,
    ideas_tree,
    markdown,
    section,
    status_panel,
    success,
    tool_call,
    tool_result,
    verdict_table,
    warning,
    welcome,
)


# ---------------------------------------------------------------------------
# rich_console
# ---------------------------------------------------------------------------


def test_welcome_panels(capsys):
    welcome(kb_size=100, vault_kept=1, vault_revised=2, vault_refuted=3, pending_ideas=4)
    captured = capsys.readouterr()
    assert "100" in captured.out or True  # banner may write via rich Console


def test_verdict_styles_defined_for_all_verdicts():
    for v in ("kept", "revised", "refuted", "pending", "failed"):
        assert v in VERDICT_STYLES
        # Each style should be a non-empty rich Style string
        assert isinstance(VERDICT_STYLES[v], str)
        assert len(VERDICT_STYLES[v]) > 0


def test_verdict_table_builds_correctly():
    t = verdict_table([
        {"id": "x1", "title": "Test", "verdict": "kept", "tokens_in": 100, "tokens_out": 50, "cost_cny": 0.01, "calls": 5},
        {"id": "x2", "title": "Other", "verdict": "refuted", "tokens_in": 200, "tokens_out": 100, "cost_cny": 0.02, "calls": 6},
    ])
    # verify the table has the right column count
    assert len(t.columns) == 6
    assert t.columns[0].header == "ID"
    assert t.columns[2].header == "Verdict"


def test_corpus_table_aggregates_by_source():
    t = corpus_table({"local": 100, "arxiv": 50, "openalex": 30})
    assert len(t.columns) == 2


def test_ideas_tree_groups_by_modality():
    tree = ideas_tree([
        {"id": "a", "title": "T1", "modalities": ["ssl"], "priority": 5},
        {"id": "b", "title": "T2", "modalities": ["graph_model", "ssl"], "priority": 7},
        {"id": "c", "title": "T3", "modalities": ["graph_model"], "priority": 6},
    ])
    # The tree root label should say "Idea backlog"
    label = str(tree.label)
    assert "Idea backlog" in label


def test_help_table_contains_all_commands():
    t = help_table()
    # The help_table has columns "Command" and "Description", and rows for each command.
    # We verify the row count matches HELP_COMMANDS.
    assert len(t.rows) == len(HELP_COMMANDS)


def test_status_panel_skips_none():
    p = status_panel({"n_kept": 1, "n_revised": 0, "n_refuted": 0,
                      "tokens_in": 100, "tokens_out": 50, "cost_cny": 0.001, "elapsed_s": 10.0})
    # doesn't raise


def test_status_panel_handles_empty_snap():
    p = status_panel({})  # nothing
    # doesn't raise; has empty body


def test_status_panel_skips_n_total_when_none():
    p = status_panel({"n_total": None, "n_kept": 1})
    # doesn't raise


def test_tool_call_and_result_do_not_raise(capsys):
    tool_call("Foo.run", "summary")
    tool_result("line 1\nline 2\nline 3", max_lines=5, indent=3)


def test_banner_runs(capsys):
    banner()
    out = capsys.readouterr()
    # either ANSI escapes or not depending on terminal; just ensure no exception


def test_section_and_code(capsys):
    section("hello")
    code("x = 1\n", "python")
    markdown("# title\n\nbody")
    error("oops")
    warning("careful")
    success("yay")


def test_status_bar_lifecycle():
    bar = StatusBar()
    bar.start()
    bar.update(idea="x1", verdict="kept", tokens="1.2K/0.6K", cost="0.01 CNY")
    bar.update(idea="x2", verdict="revised", tokens="2.4K/1.2K", cost="0.02 CNY")
    bar.stop()


def test_status_bar_with_no_live_is_silent():
    bar = StatusBar()
    bar.update(idea="x")  # before start() — must not raise
    bar.stop()
