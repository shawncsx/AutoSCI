"""Tests for v0.6 — auto-distillation, CLI dashboard, HTML report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosci.cli_dashboard import render_snapshot
from autosci.html_report import render_paper_html, render_paper_to_file, _md_to_html
from autosci.director import DirectorReport


# ---------------------------------------------------------------------------
# Auto-distillation trigger
# ---------------------------------------------------------------------------


def test_director_report_tracks_distillation_state():
    rep = DirectorReport(n_total=3)
    assert rep.distillation_triggered is False
    rep.distillation_triggered = True
    rep.distillation_report_path = "data/distillation.json"
    d = rep.__dict__
    assert d["distillation_triggered"] is True
    assert d["distillation_report_path"] == "data/distillation.json"


def test_director_run_marks_summary(tmp_path: Path):
    """When Director.run_all is invoked, the director_summary.json should
    serialize the new distillation_triggered flag.
    """
    rep = DirectorReport(n_total=2, n_kept=1, n_revised=1, n_refuted=0, n_failed=0)
    rep.distillation_triggered = True
    rep.distillation_report_path = "data/distillation.json"
    out = tmp_path / "summary.json"
    out.write_text(json.dumps({
        "n_total": rep.n_total,
        "n_kept": rep.n_kept,
        "n_revised": rep.n_revised,
        "n_refuted": rep.n_refuted,
        "n_failed": rep.n_failed,
        "distillation_triggered": rep.distillation_triggered,
        "distillation_report_path": rep.distillation_report_path,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["distillation_triggered"] is True
    assert data["distillation_report_path"] == "data/distillation.json"


# ---------------------------------------------------------------------------
# CLI dashboard rendering
# ---------------------------------------------------------------------------


def test_dashboard_snapshot_runs_without_error(tmp_path: Path):
    # Create dummy manifest
    (tmp_path / "manifest.json").write_text(
        json.dumps({"items": [
            {"id": "x01", "verdict": "kept", "title": "Test", "updated_at": "2026-10-13T08:00:00", "modality": ["ssl"]},
            {"id": "x02", "verdict": "refuted", "title": "Other", "updated_at": "2026-10-13T08:01:00", "modality": []},
        ]}),
        encoding="utf-8",
    )
    (tmp_path / "director.log").write_text(
        "08:00 [verdict] x01 :: kept | tokens=1/2 cost=0.01\n",
        encoding="utf-8",
    )
    (tmp_path / "director_summary.json").write_text(
        json.dumps({
            "elapsed_s": 100.0, "n_total": 2, "n_kept": 1,
            "n_revised": 0, "n_refuted": 1, "n_failed": 0,
            "tokens_in": 100, "tokens_out": 200,
        }),
        encoding="utf-8",
    )
    (tmp_path / "distillation.json").write_text(
        json.dumps({
            "common_strengths": ["x"],
            "common_weaknesses_in_revised": ["y"],
            "prompt_patches": {"paper_writer": "always include baseline numbers"},
        }),
        encoding="utf-8",
    )
    s = render_snapshot(tmp_path, tmp_path, tmp_path / "distillation.json")
    assert "AutoSCI Dashboard" in s
    assert "kept" in s
    assert "refuted" in s
    assert "distillation" in s.lower()
    assert "always include baseline numbers" in s


def test_dashboard_handles_missing_files_gracefully(tmp_path: Path):
    s = render_snapshot(tmp_path, tmp_path, tmp_path / "distillation.json")
    assert "no distillation.json yet" in s
    assert "Vault" in s


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def test_md_to_html_handles_bold_italic_code():
    html = _md_to_html("Hello **world** and *italic* and `code`.")
    assert "<strong>world</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>code</code>" in html


def test_md_to_html_handles_fenced_code():
    src = "Here is code:\n\n```python\nprint('hi')\n```\n\nDone."
    out = _md_to_html(src)
    assert "<pre><code>" in out
    # The code is HTML-escaped inside <pre><code>...</code></pre>
    assert "print(&#x27;hi&#x27;)" in out


def test_render_paper_html_full():
    paper = """# Test Paper

## Abstract

This is the **abstract**.

## Introduction

Some intro text.

## References

1. Foo, 2024.
2. Bar, 2025.
"""
    html = render_paper_html(paper, title_override="Test", verdict="kept")
    assert "<title>Test</title>" in html
    # The h1 in body uses the title_override
    assert "<h1>Test</h1>" in html
    assert "<strong>abstract</strong>" in html
    assert "kept" in html
    assert "<ol>" in html and "Foo, 2024" in html
    # Sections should be wrapped in <details>
    assert "id=\"abstract\"" in html
    assert "id=\"introduction\"" in html


def test_render_paper_to_file_picks_up_verdict(tmp_path: Path):
    p = tmp_path / "p"
    p.mkdir()
    (p / "paper.md").write_text("# X\n\n## Abstract\n\nfoo\n\n## References\n\n1. a\n", encoding="utf-8")
    (p / "state.json").write_text(
        json.dumps({"verdict": "refuted"}),
        encoding="utf-8",
    )
    out = tmp_path / "out.html"
    render_paper_to_file(p / "paper.md", out)
    body = out.read_text(encoding="utf-8")
    assert "refuted" in body
    assert "<title>p</title>" in body
    assert "id=\"abstract\"" in body
