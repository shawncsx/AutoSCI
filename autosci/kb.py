"""AutoSCI — Knowledge Base over the local paper corpus.

Pipeline:
    1. Ingest: scan `_digest/*.digest.md`, parse front-matter, build a corpus.
    2. Index: BM25 over the corpus.
    3. Query: bm25 top-N → optional LLM rerank → top-k.

We default to **BM25 + LLM rerank**, because:
- LLM-based reranking is the most "AutoSCI-like" semantic retrieval — the
  model itself decides what is relevant.
- It avoids forcing the user to install FAISS / sentence-transformers.

Optional embedding index (`--embed-mode faiss`) is left as a future upgrade.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from rank_bm25 import BM25Okapi

log = logging.getLogger("autosci.kb")


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------


@dataclass
class Doc:
    doc_id: str           # filename stem
    title: str
    year: int
    venue: str
    sections: dict[str, str] = field(default_factory=dict)
    fulltext: str = ""    # BM25-friendly concatenation
    tokenized: list[str] = field(default_factory=list)

    @property
    def cite_key(self) -> str:
        return f"[{self.doc_id}]"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


SECTION_RE = re.compile(r"^##\s*\[(.+?)\]\s*$", re.M)


def _strip_pdf_line_numbers(text: str) -> str:
    """PDF→md extractors often emit `word\n<num>\n`. Remove the noise.

    Heuristic: if a stripped line equals a small integer, drop it.
    """
    out_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.isdigit() and len(s) <= 4:
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def parse_digest(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = _strip_pdf_line_numbers(raw)

    # Strip a trailing ".digest" so cite_key stays short and stable.
    doc_id = path.stem
    if doc_id.endswith(".digest"):
        doc_id = doc_id[: -len(".digest")]

    title = doc_id
    year = 2026

    # Pull a sensible title from the first non-empty line, if it looks like a
    # title (no markdown prefix, doesn't end with a number-only-line, ≤ 250 chars)
    for ln in cleaned.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if len(s) > 250:
            break
        if any(c.isalpha() for c in s):
            title = s
            break

    # Try to find a year inside the first 4 KB
    m = re.search(r"\b(20[1-3]\d)\b", cleaned[:4000])
    if m:
        try:
            year = int(m.group(1))
        except Exception:
            pass

    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(cleaned))
    if not matches:
        sections["full"] = cleaned
    else:
        for i, m in enumerate(matches):
            heading = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            sections[heading] = cleaned[start:end].strip()

    fulltext = "\n\n".join(f"[{k}]\n{v}" for k, v in sections.items())
    return Doc(
        doc_id=doc_id,
        title=title,
        year=year,
        venue="arxiv/preprint",
        sections=sections,
        fulltext=fulltext,
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tiny tokenizer: lowercase + split on non-word. Good enough for BM25."""
    text = text.lower()
    text = re.sub(r"[\s\W_]+", " ", text, flags=re.UNICODE)
    return [w for w in text.split() if len(w) >= 2]


class KnowledgeBase:
    """Loads + searches the paper corpus."""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self._bm25: Optional[BM25Okapi] = None

    @classmethod
    def from_digest_dir(cls, digest_dir: Path | str, pattern: str = "*.digest.md") -> "KnowledgeBase":
        digest_dir = Path(digest_dir)
        files = sorted(digest_dir.glob(pattern))
        docs = [parse_digest(p) for p in files]
        log.info("Loaded %d digest docs from %s", len(docs), digest_dir)
        return cls(docs)

    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return
        for d in self.docs:
            d.tokenized = _tokenize(d.fulltext)
        self._bm25 = BM25Okapi([d.tokenized for d in self.docs])

    # --------------------------------------------------------------------
    # search
    # --------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 8,
        *,
        candidate_pool: int = 30,
        rerank_with: Optional["callable"] = None,
        embed_mode: Optional[str] = None,
    ) -> list[Doc]:
        """Return up to ``k`` documents matching ``query``.

        Parameters
        ----------
        query : str
        k : final number we want.
        candidate_pool : how many BM25 candidates to send to the reranker.
        rerank_with : a callable `callable(docs, query) -> list[Doc]`.
            If None, falls back to raw BM25 ordering.
        embed_mode : if set to one of 'minilm' / 'bge-m3' / 'bge-small-zh',
            also runs an embedding-based search and merges the two lists
            with Reciprocal Rank Fusion before reranking.
        """
        self._ensure_index()
        tokens = _tokenize(query)
        if not tokens:
            return []

        bm25_scores = self._bm25.get_scores(tokens)
        bm25_order = bm25_scores.argsort()[::-1][:candidate_pool]
        bm25_top = [self.docs[i] for i in bm25_order]

        # Optional: hybrid with embedding
        if embed_mode:
            try:
                from autosci.embed_index import EmbeddingIndex, hybrid_search
                idx = getattr(self, "_embed_index", None)
                if idx is None or getattr(idx, "mode", None) != embed_mode:
                    idx = EmbeddingIndex(self.docs, mode=embed_mode)
                    self._embed_index = idx
                embed_results = idx.search(query, k=candidate_pool)
                if embed_results:
                    embed_top = [(self.docs[i], sim) for i, sim in embed_results]
                    bm25_top = hybrid_search(bm25_top, embed_top, k=min(len(bm25_top), candidate_pool))
            except Exception as e:
                log.warning("embedding hybrid failed, falling back to BM25: %s", e)

        if rerank_with is not None and len(bm25_top) > k:
            return rerank_with(bm25_top, query)[:k]
        return bm25_top[:k]

    # --------------------------------------------------------------------
    # serialization helpers
    # --------------------------------------------------------------------

    def to_jsonl(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(
                    json.dumps(
                        {
                            "doc_id": d.doc_id,
                            "title": d.title,
                            "year": d.year,
                            "venue": d.venue,
                            "sections": d.sections,
                            "cite_key": d.cite_key,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------------
# LLM rerank helper
# ---------------------------------------------------------------------------

RERANK_SYSTEM = """\
你是一位文献检索专家。给定一组候选论文 + 一个研究 idea，
你的任务是**按相关度从高到低重新排序**，并选出最相关的 {{ k }} 篇。
可用英文或中文输出，但 cite_key 必须严格使用候选中给出的字符串。
"""


def make_llm_reranker(llm_client, k: int = 8) -> "callable":
    """Return a rerank function suitable for ``KB.search(rerank_with=...)``."""

    def _rerank(docs: list[Doc], query: str) -> list[Doc]:
        if not docs:
            return []
        bullets = []
        for i, d in enumerate(docs):
            abstract = (d.sections.get("开头/摘要") or next(iter(d.sections.values()), ""))[:400]
            bullets.append(f"[{i}] {d.cite_key} | {d.title} ({d.year})\n    {abstract[:280]}")
        user = (
            f"【Idea】\n{query}\n\n"
            f"【候选 {len(docs)} 篇】\n" + "\n".join(bullets) +
            f"\n\n请输出 JSON：{{ \"ordered_indices\": [<按相关度从高到低排列的索引，如 [3,0,7,...]>], "
            f"\"selected_relevance\": [\"1-2 句话说明 idx 0 为何最相关\"], "
            f"\"selected\": [<恰好 {k} 个 idx>] }}\n"
            f"\n"
        )
        try:
            payload = llm_client.chat_json(
                system=RERANK_SYSTEM.replace("{{ k }}", str(k)),
                user=user,
                temperature=0.0,
                max_tokens=800,
            )
        except Exception as e:
            log.warning("LLM rerank failed (%s); falling back to BM25 order", e)
            return docs

        # Parse response — support either "selected" or "ordered_indices"
        selected = payload.get("selected") or []
        if not selected:
            ordered = payload.get("ordered_indices") or []
            selected = ordered[:k]
        if not isinstance(selected, list):
            log.warning("LLM rerank returned unexpected shape; falling back")
            return docs[:k]
        chosen: list[Doc] = []
        seen = set()
        for idx in selected:
            try:
                i = int(idx)
            except (ValueError, TypeError):
                continue
            if 0 <= i < len(docs) and i not in seen:
                chosen.append(docs[i])
                seen.add(i)
                if len(chosen) == k:
                    break
        if not chosen:
            return docs[:k]
        # Pad to k
        for d in docs:
            if d not in chosen:
                chosen.append(d)
                if len(chosen) == k:
                    break
        return chosen

    return _rerank


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_docs_for_llm(docs: Iterable[Doc], max_chars: int = 4000, per_doc_chars: int = 600) -> str:
    """Format the docs as a text block suitable for inclusion in prompts.

    Each doc contributes at most ``per_doc_chars`` characters and the
    rendered block is hard-capped at ``max_chars``.
    """
    out = []
    total = 0
    for d in docs:
        abstract = (d.sections.get("开头/摘要") or next(iter(d.sections.values()), ""))
        abstract = abstract[:per_doc_chars]
        block = (
            f"### {d.cite_key} — {d.title} ({d.year})\n"
            f"{abstract}\n"
        )
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)
