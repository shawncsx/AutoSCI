"""AutoSCI — Corpus specs (v0.5).

A corpus spec is a small YAML / JSON file that describes a single
research domain: where to read digests, what keywords/modalities it
uses, and what the IdeaMiner should bias toward.

This is what lets AutoSCI leave microbiology. Point a spec at any
``_digest/`` directory and the same pipeline runs.

Default specs:
    data/corpus_specs/microbiome.json
    data/corpus_specs/clinical.json
    data/corpus_specs/materials.json (placeholder)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("autosci.corpus_spec")


@dataclass
class CorpusSpec:
    name: str
    description: str
    digest_dir: str
    paper_corpus_hint: str = ""
    modalities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    seed_idea_hint: str = ""
    embed_mode: Optional[str] = None  # "minilm" | "bge-m3" | "bge-small-zh" | None

    @property
    def id(self) -> str:
        return self.name.lower().replace(" ", "_")


def load_spec(path: Path | str) -> CorpusSpec:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return CorpusSpec(**data)


def load_all_specs(specs_dir: Path | str = "data/corpus_specs") -> list[CorpusSpec]:
    specs_dir = Path(specs_dir)
    if not specs_dir.exists():
        return []
    out: list[CorpusSpec] = []
    for f in sorted(specs_dir.glob("*.json")):
        try:
            out.append(load_spec(f))
        except Exception as e:
            log.warning("bad spec %s: %s", f, e)
    return out


def write_default_specs(specs_dir: Path | str = "data/corpus_specs") -> list[Path]:
    """Write the bundled default specs to ``specs_dir`` (idempotent)."""
    specs_dir = Path(specs_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, data in DEFAULT_SPECS.items():
        target = specs_dir / f"{name}.json"
        if not target.exists():
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(target)
    return written


DEFAULT_SPECS: dict[str, dict] = {
    "microbiome": {
        "name": "Microbiome Representation Learning",
        "description": "AI for metagenomics, OTU/species foundation models, cross-cohort robustness.",
        "digest_dir": "D:\\shawn\\paper\\_digest",
        "paper_corpus_hint": "graph neural network microbiome transformer OTU 99% clustering compositional cross-cohort pretraining",
        "modalities": ["graph_model", "ssl", "compositional", "cross_cohort", "threshold_sensitivity", "multimodal_fusion"],
        "keywords": ["microbiome", "metagenome", "OTU", "16S", "WMS", "alpha-diversity", "cross-cohort", "abundance"],
        "seed_idea_hint": "Graph-based representation learning for microbial communities",
        "embed_mode": "bge-small-zh",
    },
    "clinical": {
        "name": "Clinical Trial & EHR",
        "description": "Clinical outcomes prediction, patient stratification, treatment response from EHR data.",
        "digest_dir": "_digest/clinical",
        "paper_corpus_hint": "patient stratification treatment response EHR tabular foundation model mortality readmission",
        "modalities": ["patient_stratification", "treatment_response", "tabular_foundation"],
        "keywords": ["EHR", "clinical trial", "patient outcome", "mortality", "readmission", "tabular"],
        "seed_idea_hint": "Foundation model for clinical outcomes from longitudinal EHR",
        "embed_mode": "minilm",
    },
    "materials": {
        "name": "Materials Science & Chemistry",
        "description": "Molecular property prediction, catalyst discovery, MOF/co-crystal design.",
        "digest_dir": "_digest/materials",
        "paper_corpus_hint": "molecular property MOF catalyst GNN transformer DFT crystal",
        "modalities": ["molecular_property", "generation", "inverse_design"],
        "keywords": ["MOF", "crystal", "catalyst", "DFT", "SMILES", "molecule"],
        "seed_idea_hint": "Multi-modal molecular property predictor combining SMILES + 3D structure",
        "embed_mode": "minilm",
    },
}
