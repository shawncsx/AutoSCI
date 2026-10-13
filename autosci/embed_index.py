"""AutoSCI — Embedding index (v0.5 optional).

A drop-in enhancement to ``KnowledgeBase.search`` that pre-computes
sentence embeddings for every doc and uses cosine similarity in addition
to (or instead of) BM25. Falls back to BM25 if the heavy deps aren't
available.

Why optional: ``sentence-transformers`` + ``faiss-cpu`` add ~1 GB of
disk and 600 MB of RAM. We don't want to force every AutoSCI user into
that. The ``--embed-mode`` flag in ``run_autopilot.py`` switches modes.

Currently supported embedding modes:
    - "bm25"        (default): no embedding deps required
    - "minilm":      all-MiniLM-L6-v2 (English)
    - "bge-m3":      BAAI/bge-m3 (multilingual incl. Chinese)

Strategy:
    1. First call to `search()` builds the index lazily.
    2. Indices are persisted to ``data/kb_index_<mode>.faiss`` and
       ``data/kb_index_<mode>.ids.txt`` so subsequent loads are fast.
    3. Result fusion: take top-N from embedding, then top-K from BM25,
       and RRF-merge them. The fused list is passed to the LLM reranker.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from autosci.kb import Doc, KnowledgeBase


log = logging.getLogger("autosci.embed")


EMBED_CACHE_DIR = Path("data/embed_cache")


# ---------------------------------------------------------------------------
# Heavy imports are lazy and optional
# ---------------------------------------------------------------------------

def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception as e:
        log.debug("sentence-transformers not available: %s", e)
        return None


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        log.debug("faiss not available: %s", e)
        return None


# ---------------------------------------------------------------------------
# Embedding mode registry
# ---------------------------------------------------------------------------

EMBED_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-m3": "BAAI/bge-m3",
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",
}


def cache_key(mode: str, docs: list[Doc]) -> str:
    """Stable hash of (mode + concatenated fulltext hashes) for cache invalidation."""
    h = hashlib.sha256(mode.encode("utf-8"))
    for d in docs:
        h.update(d.doc_id.encode("utf-8"))
        h.update(d.fulltext[:1000].encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main index class
# ---------------------------------------------------------------------------


class EmbeddingIndex:
    """Lazy embedding index over a list of ``Doc`` objects.

    The first ``search`` call builds the index (and writes it to cache).
    Subsequent calls load from cache.
    """

    def __init__(
        self,
        docs: list[Doc],
        mode: str = "minilm",
        cache_dir: Path | str = EMBED_CACHE_DIR,
    ) -> None:
        if mode not in EMBED_MODELS:
            raise ValueError(f"Unknown embed mode: {mode!r}; choices: {list(EMBED_MODELS)}")
        self.docs = docs
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = None
        self._model = None
        self._faiss = None
        self._index = None
        self._loaded = False

    # ------------------------------------------------------------------
    # model lazy load
    # ------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        SentenceTransformer = _try_import_sentence_transformers()
        if SentenceTransformer is None:
            return False
        try:
            self._model = SentenceTransformer(EMBED_MODELS[self.mode])
            return True
        except Exception as e:
            log.warning("Failed to load embed model %s: %s", self.mode, e)
            return False

    def _ensure_faiss(self) -> bool:
        if self._index is not None:
            return True
        faiss = _try_import_faiss()
        if faiss is None or self._embeddings is None:
            return False
        import numpy as np  # local import; bundled with sentence-transformers
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        # vectors must be float32 + L2-normalised for inner-product = cosine
        self._index.add(np.ascontiguousarray(self._embeddings.astype("float32")))
        return True

    # ------------------------------------------------------------------
    # build / load
    # ------------------------------------------------------------------

    def _build_or_load(self) -> bool:
        if self._loaded:
            return self._embeddings is not None
        self._loaded = True

        ck = cache_key(self.mode, self.docs)
        vec_path = self.cache_dir / f"{ck}_{self.mode}.npy"
        meta_path = self.cache_dir / f"{ck}_{self.mode}.json"
        if vec_path.exists() and meta_path.exists():
            try:
                import numpy as np
                self._embeddings = np.load(vec_path)
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("n_docs") == len(self.docs):
                    log.info("[embed] loaded cached %s vectors (%d)", self.mode, len(self.docs))
                    return True
            except Exception as e:
                log.debug("embed cache load failed: %s", e)

        if not self._ensure_model():
            log.warning("[embed] sentence-transformers not available; cannot embed")
            return False
        try:
            import numpy as np
            texts = [d.fulltext[:3000] for d in self.docs]
            vecs = self._model.encode(
                texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=32,
            )
            vecs = np.asarray(vecs, dtype="float32")
            self._embeddings = vecs
            np.save(vec_path, vecs)
            meta_path.write_text(
                json.dumps({"n_docs": len(self.docs), "mode": self.mode}, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("[embed] built %s vectors for %d docs", self.mode, len(self.docs))
            return True
        except Exception as e:
            log.warning("[embed] build failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 8) -> list[tuple[int, float]]:
        """Return (doc_idx, similarity) pairs sorted desc.

        Returns empty list if embeddings aren't available.
        """
        if not self._build_or_load():
            return []
        if not self._ensure_model():
            return []
        try:
            import numpy as np
            q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            q = np.asarray(q, dtype="float32")
            if self._ensure_faiss():
                scores, idxs = self._index.search(q, min(k * 2, len(self.docs)))
                return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
            # fallback: numpy dot product
            sims = (self._embeddings @ q[0]).tolist()
            order = sorted(range(len(sims)), key=lambda i: -sims[i])
            return [(i, sims[i]) for i in order[:k]]
        except Exception as e:
            log.warning("[embed] search failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# Hybrid fusion: BM25 + Embedding
# ---------------------------------------------------------------------------

def hybrid_search(
    bm25_top: list[Doc],
    embed_top: list[tuple[Doc, float]],
    k: int,
) -> list[Doc]:
    """Reciprocal Rank Fusion (RRF) of two ranked lists.

    Each list's items are scored as `1 / (rank + 60)` (standard RRF constant),
    summed across lists, then sorted descending.
    """
    RRF_K = 60
    scores: dict[str, float] = {}
    objects: dict[str, Doc] = {}
    for rank, d in enumerate(bm25_top):
        scores[d.doc_id] = scores.get(d.doc_id, 0.0) + 1.0 / (rank + RRF_K)
        objects[d.doc_id] = d
    for rank, (d, _sim) in enumerate(embed_top):
        scores[d.doc_id] = scores.get(d.doc_id, 0.0) + 1.0 / (rank + RRF_K)
        objects[d.doc_id] = d
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    out: list[Doc] = []
    for doc_id, _ in ranked:
        out.append(objects[doc_id])
        if len(out) >= k:
            break
    return out
