"""Pipeline smoke test — uses a MOCKED LLM to verify pipeline control flow.

Run with:
    python -m pytest tests/test_pipeline_smoke.py -v

This test does NOT call any external service, so it is fully deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autosci.kb import Doc, KnowledgeBase
from autosci.llm import LLMClient
from autosci.pipeline import ResearchTeam, RunConfig
from autosci.schemas import (
    IdeaRefined,
    IdeaSeed,
    IdeaState,
    LitSurveyItem,
    MethodSpec,
    ExpSpec,
    PaperDraft,
    CodeSketch,
    CodeSketchFile,
    Critique,
    CritiqueLineItem,
    RefuteResult,
)


class FakeLLM:
    """A fake LLMClient that hands back canned JSON responses per prompt."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, *, system: str, user: str, temperature: float = 0.2, max_tokens: int | None = None) -> dict[str, Any]:
        self.calls.append((system[:40], user[:60]))
        s = system
        if "微生物组表示学习" in s and "可证伪" in s:
            return {
                "claim": "Graph > rank-transformer.",
                "primary_hypothesis": "IF we use GAT FiLM THEN AUROC +5pp",
                "secondary_hypotheses": ["H2", "H3"],
                "risks": ["data sparsity"],
            }
        if "文献综述" in s:
            return {"items": [
                {"cite_key":"[MGM2-2026]","title":"MGM2","year":2026,"venue":"x","relevance":"r","takeaway":"t"},
                {"cite_key":"[BiomeGPT-2025]","title":"BiomeGPT","year":2025,"venue":"x","relevance":"r2","takeaway":"t2"},
            ]}
        if "算法设计师" in s:
            return {
                "architecture":"GAT with FiLM, 3 layers.",
                "algorithm":"1. encode abundance 2. GAT 3. pool",
                "training_objective":"contrastive + reconstruction",
                "novelty_bullets":["perm invariance","FiLM ablation","edge prior"],
                "ablations_planned":["no FiLM", "static edges"],
            }
        if "实验设计专家" in s:
            return {
                "datasets":[{"name":"CRC8","n_samples":"1538","phenotype":"CRC","use":"train+eval"}],
                "baselines":[{"name":"MGM","type":"existing","notes":"ref"}],
                "ablations":[{"name":"no FiLM","toggled":"FiLM off","rationale":"see effect"}],
                "metrics":["AUROC","AUPRC"],
                "significance_test":"Wilcoxon",
                "cross_cohort_protocol":"leave-one-cohort-out",
                "expected_outcome":"+5pp AUROC",
            }
        if "代笔" in s:
            return {
                "title":"Permutation-Invariant GAT-FiLM for Microbiome Cross-Cohort Analysis",
                "abstract":"Abstract text " * 30,
                "keywords":["graph","microbiome"],
                "sections": {
                    "introduction": "intro " * 60,
                    "related_work": "rw " * 60,
                    "method": "method " * 80,
                    "experiments": "exp " * 60,
                    "results_template": "results " * 60,
                    "discussion": "discussion " * 60,
                    "limitations": "limitations " * 50,
                    "conclusion": "conclusion " * 50,
                },
                "references":["[MGM2-2026]","[BiomeGPT-2025]","[r3]","[r4]","[r5]","[r6]","[r7]","[r8]"]
            }
        if "可被工程师直接照着填空" in s:
            return {
                "files": [
                    {"path":"experiments/main.py","content":"print('hi')\n"},
                    {"path":"autosci_lite/model.py","content":"class Model: pass\n"},
                ],
                "run_instructions":"pip install numpy\npython experiments/main.py"
            }
        if "同行评审人" in s:
            return {
                "summary":"Reviewer summary text.",
                "line_items":[
                    {"section":"method","anchor":"GAT with FiLM, 3 layers.","severity":"minor","suggestion":"tweak"},
                ],
                "severity_counts":{"minor":1,"major":0,"blocking":0},
            }
        if "挑刺" in s:
            return {"verdict":"keep","fatal_flaws":[],"revise_suggestions":["add more baselines"],"notes":""}
        return {"ok": True}


def _tiny_kb() -> KnowledgeBase:
    long_para = (
        "graph neural network for microbiome representation learning with abundance "
        "permutation invariance FiLM cross-cohort baseline transformer encoder contrastive "
        "self-supervised pretraining compositional zero-inflated Dirichlet multinomial "
        "log-ratio CLR ALR OTU 97 99 clustering threshold sensitivity leave-one-cohort-out "
        "stability robustness microbiome community taxonomy functional analysis."
    )
    docs = [
        Doc(doc_id="MGM2-2026", title="MGM2 microbiome FM", year=2026, venue="x",
            sections={"开头/摘要": long_para + " MGM2 specific details."}),
        Doc(doc_id="BiomeGPT-2025", title="BiomeGPT", year=2025, venue="x",
            sections={"开头/摘要": long_para + " BiomeGPT specifics."}),
        Doc(doc_id="Waypoint-2026", title="Waypoint", year=2026, venue="x",
            sections={"开头/摘要": long_para + " Waypoint specifics."}),
        Doc(doc_id="Random-2024", title="Random Forest for microbiome", year=2024, venue="x",
            sections={"开头/摘要": long_para + " Random Forest specifics."}),
        Doc(doc_id="GAT-2023", title="Graph neural networks for microbiome", year=2023, venue="x",
            sections={"开头/摘要": long_para + " GAT specifics."}),
    ]
    for d in docs:
        d.fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in d.sections.items())
    return KnowledgeBase(docs)


def test_pipeline_smoke(tmp_path: Path):
    cfg = RunConfig(output_root=tmp_path, kb_topk=2)
    fake = FakeLLM()
    kb = _tiny_kb()
    team = ResearchTeam(llm=type("L", (), {"chat_json": fake.chat_json})(), kb=kb, cfg=cfg)  # type: ignore[arg-type]

    seed = IdeaSeed(
        id="smoke01",
        title="Graph > Rank-Transformer",
        one_liner="GAT FiLM beats rank transformer",
        angle="permutation invariance",
        modalities=[],
    )
    state = team.run_one(seed)

    assert state.refined is not None
    assert len(state.lit_survey) >= 2
    assert state.method_spec is not None
    assert state.exp_spec is not None
    assert state.paper_draft is not None
    assert state.code_sketch is not None
    assert state.review is not None
    assert state.refute is not None
    assert state.verdict.value in ("kept", "revised", "refuted")

    run_dir = tmp_path / "smoke01"
    assert (run_dir / "paper.md").exists()
    assert (run_dir / "state.json").exists()
    # Files are placed at the path the agent declared (e.g. 'experiments/main.py'),
    # so the test looks for them at their declared nested path.
    assert (run_dir / "code" / "experiments" / "main.py").exists()
    assert (run_dir / "code" / "autosci_lite" / "model.py").exists()


def test_director_end_to_end_with_fake_llm(tmp_path: Path):
    from autosci.director import Director
    from autosci.vault import Vault

    cfg = RunConfig(output_root=tmp_path / "runs", kb_topk=2)
    fake = FakeLLM()
    kb = _tiny_kb()
    vault = Vault(root=tmp_path / "vault")

    class _L:
        chat_json = fake.chat_json

    director = Director(llm=_L(), kb=kb, cfg=cfg, vault=vault)  # type: ignore[arg-type]
    seeds = [
        IdeaSeed(id=f"smoke{i:02d}", title=f"S{i}", one_liner="x", angle="y", modalities=[])
        for i in range(2)
    ]
    rep = director.run_all(seeds)

    assert rep.n_total == 2
    assert rep.n_kept + rep.n_revised + rep.n_refuted == 2
    assert (tmp_path / "vault" / "manifest.json").exists()
