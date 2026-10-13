"""AutoSCI — Deep paper analyzer (v0.15).

Given a paper (full-text digest or PDF fulltext), produce a structured
5-section analysis that is *useful for proposing follow-up research*:

    1. **Research question** — what the paper is trying to answer.
    2. **Method** — the key technique (1-3 sentences).
    3. **Data** — datasets / corpora used + size.
    4. **Main result** — the headline numbers (1-2 sentences).
    5. **Limitations + opportunities** — what the paper admits is weak,
       and what *new* direction opens up.

The output is JSON-shaped so the IDE LLM (or AutoSCI's LLM) can
post-process into IdeaSeed candidates.

The analyzer is intentionally simple: deterministic extraction from
the digest's sectioned content. No LLM call needed. Quality scales
with how well-structured the source digest is.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from autosci.kb import Doc


log = logging.getLogger("autosci.paper_analyzer")


@dataclass
class PaperAnalysis:
    doc_id: str = ""
    title: str = ""
    year: int = 0
    research_question: str = ""
    method: str = ""
    data: str = ""
    main_result: str = ""
    limitations: str = ""
    opportunities: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # raw inputs (for IDE LLM context)
    abstract_excerpt: str = ""
    fulltext_excerpt: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title, "year": self.year,
            "research_question": self.research_question,
            "method": self.method, "data": self.data, "main_result": self.main_result,
            "limitations": self.limitations,
            "opportunities": self.opportunities,
            "modalities": self.modalities, "keywords": self.keywords,
        }


# Heuristics to extract structured pieces from a paper's digest / fulltext
_SENT_SPLIT = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z])")

_DATASET_HINTS = re.compile(
    r"\b(microbiome|dataset|corpus|samples?|patients?|"
    r"MGnify|TCGA|GEO|metalog|CRC\d+|HMP|"
    r"16S|WMS|MetaPhlAn|HUMAnN|"
    r"ImageNet|CIFAR|MNIST|GLUE|SuperGLUE)\b",
    re.I,
)

_METHOD_HINTS = re.compile(
    r"\b(transformer|GNN|graph neural network|GAT|GIN|"
    r"contrastive learning|self-supervised|pretraining|fine-tuning|"
    r"few-shot|zero-shot|RLHF|"
    r"meta-learning|domain adaptation|"
    r"random forest|XGBoost|LightGBM|"
    r"convolutional|recurrent|"
    r"diffusion model|VAE|GAN|"
    r"prompt|chain-of-thought|"
    r"OTU|taxonomy|phylogenetic)\b",
    re.I,
)

_RESULT_HINTS = re.compile(
    r"\b(\+?\s*\d+\.?\d*\s*%|\+?\s*\d+\.?\d*\s*pp|"
    r"AUROC|AUC|AUPRC|F1|"
    r"accuracy|precision|recall|"
    r"p\s*<|p\s*=|state-of-the-art|SOTA|"
    r"beats? baseline|outperforms?|"
    r"achieves?\s+(\d+\.?\d*)|"
    r"improves?\s+by\s+\d+\.?\d*)\b",
    re.I,
)

_LIMITATION_HINTS = re.compile(
    r"\b(limitation|caveat|future work|do not|does not|"
    r"small sample|small dataset|computationally expensive|"
    r"limited|biased|scalability issue|"
    r"out-of-distribution|shift|"
    r"fails? to|struggle|gap)\b",
    re.I,
)

_NOISE_LINE = re.compile(r"^\s*[\d\.\)\]]+\s*$|^[_\-=*\s]{3,}$")
_PAGE_RE = re.compile(r"^## page \d+\s*$", re.M)


def _clean_text(text: str) -> str:
    """Strip PDF line-number noise and page markers, collapse whitespace."""
    text = _PAGE_RE.sub("", text)
    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or _NOISE_LINE.match(s):
            continue
        if s.isdigit():
            continue
        if len(s) > 30 and not any(c.isalpha() for c in s):
            continue
        out_lines.append(s)
    out = "\n".join(out_lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _first_n_sentences(text: str, n: int = 3) -> str:
    sents = _SENT_SPLIT.split(text)
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:n])


def _find_with_window(text: str, pattern: re.Pattern, before: int = 80, after: int = 200) -> str:
    """Find the first match of pattern and return text around it."""
    m = pattern.search(text)
    if not m:
        return ""
    start = max(0, m.start() - before)
    end = min(len(text), m.end() + after)
    return text[start:end].strip()


def analyze_doc(doc: Doc) -> PaperAnalysis:
    """Run the deterministic extractor on a Doc."""
    fulltext = _clean_text(doc.fulltext or "")
    abstract = _clean_text(
        doc.sections.get("开头/摘要") or next(iter(doc.sections.values()), "")
    )
    if not fulltext and abstract:
        fulltext = abstract

    pa = PaperAnalysis(
        doc_id=doc.doc_id,
        title=doc.title,
        year=doc.year,
        abstract_excerpt=abstract[:800],
        fulltext_excerpt=fulltext[:1500],
    )

    # Research question: usually in the intro or abstract
    intro = doc.sections.get("Introduction") or doc.sections.get("引言") or ""
    intro_clean = _clean_text(intro) if intro else ""
    if not intro_clean and abstract:
        # first 1-2 sentences of abstract
        pa.research_question = _first_n_sentences(abstract, 2)
    elif intro_clean:
        pa.research_question = _first_n_sentences(intro_clean, 2)
    else:
        pa.research_question = _first_n_sentences(fulltext, 2)

    # Method: look for method section or method-hinting snippets
    method_sec = (doc.sections.get("Method") or doc.sections.get("方法")
                  or doc.sections.get("Methods") or doc.sections.get("模型") or "")
    if method_sec:
        method_text = _clean_text(method_sec)
        pa.method = _first_n_sentences(method_text, 4)
    else:
        m = _METHOD_HINTS.search(fulltext)
        if m:
            pa.method = _find_with_window(fulltext, _METHOD_HINTS, before=40, after=180)

    # Data: dataset section or dataset hint
    data_sec = (doc.sections.get("Data") or doc.sections.get("Datasets")
                or doc.sections.get("Experiments") or doc.sections.get("实验") or "")
    if data_sec:
        pa.data = _first_n_sentences(_clean_text(data_sec), 3)
    else:
        m = _DATASET_HINTS.search(fulltext)
        if m:
            pa.data = _find_with_window(fulltext, _DATASET_HINTS, before=20, after=200)

    # Main result: result / experiment section or number-rich sentence
    result_sec = (doc.sections.get("Results") or doc.sections.get("结果")
                  or doc.sections.get("Experiments") or doc.sections.get("Results Template") or "")
    if result_sec:
        result_text = _clean_text(result_sec)
        # find a sentence with a number
        for sent in _SENT_SPLIT.split(result_text):
            if _RESULT_HINTS.search(sent):
                pa.main_result = sent.strip()
                break
        if not pa.main_result:
            pa.main_result = _first_n_sentences(result_text, 2)
    else:
        m = _RESULT_HINTS.search(fulltext)
        if m:
            pa.main_result = _find_with_window(fulltext, _RESULT_HINTS, before=40, after=200)

    # Limitations
    disc_sec = (doc.sections.get("Discussion") or doc.sections.get("Limitations")
                or doc.sections.get("讨论") or doc.sections.get("局限") or "")
    if disc_sec:
        disc_text = _clean_text(disc_sec)
        for sent in _SENT_SPLIT.split(disc_text):
            if _LIMITATION_HINTS.search(sent):
                pa.limitations = sent.strip()
                break
    if not pa.limitations:
        m = _LIMITATION_HINTS.search(fulltext)
        if m:
            pa.limitations = _find_with_window(fulltext, _LIMITATION_HINTS, before=40, after=200)

    # Modalities + keywords
    pa.modalities = list({m.group(0).lower() for m in _METHOD_HINTS.finditer(fulltext)})[:5]
    pa.keywords = [m.group(0) for m in _DATASET_HINTS.finditer(fulltext)][:10]

    # Opportunities: derived from limitations → "address <limit> by <suggestion>"
    if pa.limitations:
        pa.opportunities.append(
            f"Address the paper's self-acknowledged limitation: "
            f"\"{_first_n_sentences(pa.limitations, 1)}\""
        )
    if not pa.method:
        pa.opportunities.append(
            "Reformulate the paper's method with a clearer algorithmic specification."
        )
    if not pa.data:
        pa.opportunities.append(
            "Re-evaluate the method on a complementary public dataset."
        )
    pa.opportunities.append(
        f"Build on {doc.title[:60]!r} with a cross-domain or cross-cohort extension."
    )

    return pa


def analyze_corpus(docs: list[Doc], top_n: int = 5) -> list[PaperAnalysis]:
    out: list[PaperAnalysis] = []
    for d in docs[:top_n]:
        try:
            out.append(analyze_doc(d))
        except Exception as e:
            log.warning("analyze_doc failed on %s: %s", d.doc_id, e)
    return out
