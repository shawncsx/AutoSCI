"""Tests for v0.7 — verifier, parallel director, compilation."""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autosci.compilation import (
    build_compilation_html,
    build_compilation_markdown,
    parse_paper_sections,
    verdict_from_state,
)
from autosci.verifier import (
    auroc,
    make_synthetic_dataset,
    parse_target_delta,
    train_logistic_head,
    verify_claim,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_synthetic_dataset_shapes():
    X, y = make_synthetic_dataset(n_samples=100, n_features=20, seed=0)
    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert set(y.tolist()).issubset({0, 1})


def test_logistic_head_improves_over_random():
    X, y = make_synthetic_dataset(seed=42)
    # Shuffled labels → logistic should land near chance
    rng = np.random.default_rng(123)
    y_shuf = rng.permutation(y)
    w, b = train_logistic_head(X, y_shuf, n_iter=300, reg=1.0, seed=0)
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    a = auroc(y, p)
    assert 0.4 <= a <= 0.6, f"shuffled AUROC should be ~0.5, got {a}"

    # Real labels → should be >> 0.5
    w, b = train_logistic_head(X, y, n_iter=500, reg=1e-4, seed=0)
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    a = auroc(y, p)
    assert a > 0.85, f"real-labels AUROC should be high, got {a}"


def test_auroc_known_case():
    # Perfect separation → 1.0
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y, p) > 0.99


def test_parse_target_delta_positive():
    assert parse_target_delta("we will achieve +5% AUROC") == pytest.approx(0.05)
    assert parse_target_delta("+5pp over baseline") == pytest.approx(0.05)
    assert parse_target_delta("+ 3 percent improvement") == pytest.approx(0.03)
    assert parse_target_delta("by 7pp") == pytest.approx(0.07)


def test_parse_target_delta_handles_no_number():
    assert parse_target_delta("we will improve representation learning") is None


def test_verify_claim_returns_structured_result():
    res = verify_claim("our method will achieve +5% AUROC over baseline", seed=7)
    assert isinstance(res, VerificationResult)
    assert 0.5 <= res.auroc_our_method <= 1.0
    assert 0.0 <= res.auroc_random_baseline <= 1.0
    assert 0.0 <= res.auroc_logreg_baseline <= 1.0
    assert isinstance(res.verified, bool)
    assert "AUROC" in res.reason


def test_verify_claim_unmet_target_marks_unverified():
    # A claim of +50pp is essentially unattainable on synthetic data; should
    # be marked verified=False when target exists.
    res = verify_claim("we will achieve +50% AUROC over baseline", seed=0)
    assert res.claim_target_delta is not None
    assert res.verified is False


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def test_parse_paper_sections_extracts_title_and_body():
    paper = """# Test Paper Title

## Abstract

This is the abstract.

## Introduction

Some intro.

## References

1. Foo
"""
    out = parse_paper_sections(paper)
    assert out.get("title") == "Test Paper Title"
    assert "Abstract" in out
    assert "This is the abstract" in out["Abstract"]
    # Internal Audit (if present) is excluded
    assert all("Internal" not in k for k in out)


def test_build_compilation_markdown_groups_by_verdict(tmp_path: Path):
    # Create a fake kept + a fake revised paper
    for vid, verdict in (("01_kept_x", "kept"), ("02_revised_y", "revised")):
        d = tmp_path / vid
        d.mkdir()
        (d / "paper.md").write_text(
            f"# Title {vid}\n\n## Abstract\n\nabstract of {vid}\n\n## References\n\n1. a\n",
            encoding="utf-8",
        )
        (d / "state.json").write_text(json.dumps({
            "verdict": verdict,
            "idea_seed": {"id": vid, "modalities": ["ssl"]},
        }), encoding="utf-8")
    md = build_compilation_markdown(tmp_path, kept_only=True)
    assert "AutoSCI Compilation" in md
    assert "Title 01_kept_x" in md
    assert "Title 02_revised_y" in md
    # by default refuted is NOT included
    assert "Refuted" not in md


def test_build_compilation_includes_refuted_when_requested(tmp_path: Path):
    for vid, verdict in (("k1", "kept"), ("r1", "refuted")):
        d = tmp_path / vid
        d.mkdir()
        (d / "paper.md").write_text(
            f"# T {vid}\n\n## Abstract\n\nabs {vid}\n\n## References\n\n1. a\n",
            encoding="utf-8",
        )
        (d / "state.json").write_text(json.dumps({
            "verdict": verdict,
            "idea_seed": {"id": vid, "modalities": []},
        }), encoding="utf-8")
    md = build_compilation_markdown(tmp_path, kept_only=False)
    assert "T k1" in md
    assert "T r1" in md


def test_build_compilation_html_self_contained(tmp_path: Path):
    d = tmp_path / "k1"
    d.mkdir()
    (d / "paper.md").write_text(
        "# T k1\n\n## Abstract\n\nabs\n\n## References\n\n1. a\n",
        encoding="utf-8",
    )
    (d / "state.json").write_text(
        json.dumps({"verdict": "kept", "idea_seed": {"id": "k1", "modalities": []}}),
        encoding="utf-8",
    )
    html = build_compilation_html(tmp_path)
    assert "<!doctype html>" in html
    assert "AutoSCI Compilation" in html
    assert "T k1" in html


# ---------------------------------------------------------------------------
# Director parallel option
# ---------------------------------------------------------------------------


def test_director_parallel_attribute_default():
    from autosci.director import Director
    # Minimal constructor — we just need the parallel attr
    from autosci.llm import LLMClient
    from autosci.kb import KnowledgeBase
    from autosci.pipeline import RunConfig
    from autosci.vault import Vault
    from autosci.memory import DirectorMemory
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase.from_digest_dir(Path("D:/shawn/paper/_digest")) if Path("D:/shawn/paper/_digest").exists() else KnowledgeBase([])
        d = Director(
            llm=LLMClient(api_key="x", client=__import__("httpx").Client(trust_env=False)),
            kb=kb,
            cfg=RunConfig(output_root=Path(td) / "runs"),
            vault=Vault(root=Path(td) / "vault"),
            parallel=3,
        )
        assert d.parallel == 3
