"""Tests for v0.2 — IdeaMiner, MetaPromptBuilder, Autopilot control flow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autosci.agents.idea_miner import compute_axis_signals, dedupe_by_title, GAP_KEYWORDS
from autosci.kb import Doc, KnowledgeBase
from autosci.meta_prompter import STRATEGIES, suggest_strategy
from autosci.schemas import IdeaSeed, Modality


def _mini_kb() -> KnowledgeBase:
    long_para = (
        "graph neural network for microbiome representation learning with abundance "
        "permutation invariance FiLM cross-cohort baseline transformer encoder contrastive "
        "self-supervised pretraining compositional zero-inflated Dirichlet multinomial "
        "log-ratio CLR ALR OTU 97 99 clustering threshold sensitivity leave-one-cohort-out "
        "stability robustness microbiome community taxonomy functional analysis."
    )
    docs = [
        Doc(doc_id=f"d{i:02d}", title=f"Title{i}", year=2024, venue="x",
            sections={"开头/摘要": long_para + f" UniqueDetails{i}."})
        for i in range(8)
    ]
    for d in docs:
        d.fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in d.sections.items())
    return KnowledgeBase(docs)


def test_compute_axis_signals_runs():
    kb = _mini_kb()
    sig = compute_axis_signals(kb)
    assert isinstance(sig, dict)
    for axis in GAP_KEYWORDS:
        assert axis in sig


def test_dedupe_by_title_drops_dups():
    a = IdeaSeed(id="a", title="foo", one_liner="x", angle="y", modalities=[Modality.SSL])
    b = IdeaSeed(id="b", title="foo  ", one_liner="z", angle="w", modalities=[Modality.SSL])
    c = IdeaSeed(id="c", title="bar", one_liner="z", angle="w", modalities=[])
    out = dedupe_by_title([a, b, c])
    assert len(out) == 2
    # first occurrence wins
    assert {s.id for s in out} == {"a", "c"}


def test_strategies_dict_has_baseline_set():
    assert "be_more_skeptical" in STRATEGIES
    assert "cut_to_the_bone" in STRATEGIES
    assert "use_a_different_angle" in STRATEGIES


def test_suggest_strategy_heuristic():
    assert suggest_strategy("重复了") == "be_more_skeptical"
    assert suggest_strategy("长度太长") == "cut_to_the_bone"
    assert suggest_strategy("完全不相关") == "use_a_different_angle"


def test_director_memory_records_and_detects_duplicate(tmp_path: Path):
    from autosci.memory import DirectorMemory
    from autosci.schemas import IdeaSeed, IdeaState, Verdict, RefuteResult, Critique, CritiqueLineItem

    mem = DirectorMemory(path=tmp_path / "mem.jsonl")
    state = IdeaState(idea_seed=IdeaSeed(id="a1", title="GNN for microbiome", one_liner="x", angle="y", modalities=[]))
    state.verdict = Verdict.REFUTED
    state.refute = RefuteResult(verdict="refute", fatal_flaws=[
        {"flaw": "missing baseline", "evidence": "no Random Forest", "fix": "add it"},
    ])
    mem.record(state)
    assert len(mem.entries) == 1
    assert mem.is_duplicate(IdeaSeed(id="a2", title="GNN for microbiome", one_liner="x", angle="y", modalities=[]))
    assert not mem.is_duplicate(IdeaSeed(id="a3", title="DNA language model for cancer", one_liner="x", angle="y", modalities=[]))
    s = mem.summary()
    assert s["by_verdict"].get("refuted") == 1
    assert "baseline" in s["refuted_top_keywords"]


def test_autopilot_control_flow(tmp_path: Path):
    """Run autopilot for 1 cycle with a no-op IdeaMiner and no github."""
    from autosci.autopilot import Autopilot, AutopilotConfig
    from autosci.director import Director
    from autosci.pipeline import RunConfig
    from autosci.vault import Vault

    class _FakeLLM:
        """Auto-fakes every agent prompt + the IdeaMiner call."""

        def chat_json(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int | None = None) -> dict:
            s = system
            if "学术星探" in s or "IdeaSeed" in user:
                # IdeaMiner
                return {"ideas":[
                    {"id":"m01","title":"Mined A","one_liner":"x","angle":"y","modalities":["ssl"],"priority":5,"paper_corpus_hint":""},
                    {"id":"m02","title":"Mined B","one_liner":"x","angle":"y","modalities":["compositional"],"priority":5,"paper_corpus_hint":""},
                ]}
            if "研究员" in s and "可证伪" in s:
                return {"claim":"c","primary_hypothesis":"h","secondary_hypotheses":[],"risks":[]}
            if "文献综述" in s:
                return {"items":[{"cite_key":"[A-2025]","title":"A","year":2025,"venue":"x","relevance":"r","takeaway":"t"}]}
            if "算法设计师" in s:
                return {"architecture":"a","algorithm":"al","training_objective":"to","novelty_bullets":["n"],"ablations_planned":["x"]}
            if "实验设计专家" in s:
                return {"datasets":[{"name":"d","n_samples":"1","phenotype":"p","use":"train"}],
                        "baselines":[{"name":"b","type":"existing","notes":""}],
                        "ablations":[{"name":"a","toggled":"t","rationale":"r"}],
                        "metrics":["AUROC"],"significance_test":"Wilcoxon",
                        "cross_cohort_protocol":"LODO","expected_outcome":"+5pp"}
            if "代笔" in s:
                return {"title":"t","abstract":"a " * 30,"keywords":[],
                        "sections":{"introduction":"i "*60,"related_work":"r "*60,"method":"m "*60,
                                    "experiments":"e "*60,"results_template":"rt "*60,
                                    "discussion":"d "*60,"limitations":"l "*60,"conclusion":"c "*60},
                        "references":["[A-2025]","[B-2025]","[C-2025]","[D-2025]","[E-2025]","[F-2025]","[G-2025]","[H-2025]"]}
            if "可被工程师" in s:
                return {"files":[{"path":"main.py","content":"x=1\n"}],"run_instructions":"run"}
            if "同行评审人" in s:
                return {"summary":"ok","line_items":[]}
            if "挑刺" in s:
                return {"verdict":"keep","fatal_flaws":[],"revise_suggestions":[]}
            return {"ok": True}

    kb = _mini_kb()
    cfg = RunConfig(output_root=tmp_path / "runs")
    vault = Vault(root=tmp_path / "vault")
    director = Director(llm=_FakeLLM(), kb=kb, cfg=cfg, vault=vault)  # type: ignore[arg-type]
    ap_cfg = AutopilotConfig(
        seed_path=None,
        max_seeds_per_cycle=2,
        mine_extra_per_cycle=2,
        max_cycles=2,
        resupply_kb_each_cycle=False,
        keep_alive_path=tmp_path / "autopilot.json",
    )
    ap = Autopilot(llm=_FakeLLM(), kb=kb, director=director, cfg=ap_cfg)  # type: ignore[arg-type]
    rep = ap.run()
    assert rep.ideas_attempted >= 2
    assert rep.ideas_mined >= 2
    assert (tmp_path / "vault" / "manifest.json").exists()
