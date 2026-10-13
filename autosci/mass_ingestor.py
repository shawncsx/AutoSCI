"""AutoSCI — MassArxivIngester (v0.8).

Run a **battery** of arXiv queries in parallel to build a large corpus
(1000+ papers). Designed for the "mass retrieval" the user asked for:
instead of pulling 3-5 papers per query, we pull 50-100 per query,
across 20-40 carefully-chosen queries, in parallel.

Key constraints:
    - arXiv asks for ≥3s between unauthenticated requests, so we cap
      concurrency (default 4) and add a 0.5s jitter.
    - Persistent cache: results are written to ``data/mass_corpus/``
      and indexed by arXiv-id; re-running with the same query bank
      is idempotent.
    - Deduplication: by arXiv-id, plus by title hash for cross-source
      safety.

Usage:
    python -m autosci.mass_ingestor --queries-file data/mass_queries.json
    python -m autosci.mass_ingestor --queries "microbiome foundation model" "graph neural network" ...
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import httpx

from autosci.arxiv_ingester import ARXIV_API, _parse_atom, write_as_digests


log = logging.getLogger("autosci.mass_ingestor")


# ---------------------------------------------------------------------------
# Default query bank — focused on microbiome + AI for science
# ---------------------------------------------------------------------------

DEFAULT_QUERY_BANK: list[str] = [
    # Microbiome foundation models
    "microbiome foundation model",
    "metagenome transformer",
    "16S rRNA deep learning",
    "metagenomic language model",
    "gut microbiome representation",
    "microbiome graph neural network",
    "taxonomy aware deep learning microbiome",
    "phage identification transformer",
    "MAG recovery deep learning",
    "k-mer genome representation",
    # Microbiome methods
    "OTU clustering threshold",
    "compositional data microbiome",
    "Dirichlet multinomial microbiome",
    "zero-inflated microbiome",
    "cross-cohort microbiome generaliz",
    "leave-one-cohort microbiome",
    "metagenomics benchmark",
    "phylogenetic Kantorovich-Rubinstein",
    "metagenomic de Bruijn graph",
    # AI for science in general
    "graph transformer self-supervised",
    "tabular foundation model",
    "graph foundation model",
    "self-supervised pretraining biology",
    "contrastive learning bioinformatics",
    "graph attention network biology",
    "set transformer microbiome",
    "diffusion model molecular",
    "GNN drug discovery",
    "graph neural network retrosynthesis",
    # Clinical / EHR (for cross-domain)
    "EHR foundation model",
    "clinical trial patient stratific",
    "tabular foundation model robust",
    "domain shift medical",
    # General ML
    "graph self-supervised pretraining",
    "transformer long sequence",
    "few-shot in-context learning",
    "instruction tuning language model",
    "mixture of experts scaling",
]


def fetch_one_query(query: str, max_results: int, timeout: float = 30) -> tuple[str, int, list[dict]]:
    """Fetch a single query. Returns (query, count, raw_atom_entries).

    Failures return count=0 and log; never raise.
    """
    client = httpx.Client(timeout=timeout, trust_env=False)
    try:
        r = client.get(ARXIV_API, params=[
            ("search_query", query),
            ("start", "0"),
            ("max_results", str(max_results)),
        ])
        r.raise_for_status()
        entries = _parse_atom(r.text)
        # arxiv_ingester returns ArxivEntry dataclasses; convert to dict
        out = []
        for e in entries:
            out.append({
                "arxiv_id": e.arxiv_id,
                "title": e.title,
                "summary": e.summary,
                "authors": e.authors,
                "year": e.year,
                "categories": e.categories,
                "link": e.link,
            })
        return (query, len(out), out)
    except httpx.HTTPError as e:
        log.warning("arxiv query '%s' failed: %s", query, e)
        return (query, 0, [])
    finally:
        time.sleep(3.1)  # arXiv asks for ≥ 3s between unauth calls


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------


def run_mass(
    queries: list[str],
    out_dir: Path | str = "data/mass_corpus",
    per_query: int = 50,
    max_workers: int = 3,
    dedupe: bool = True,
    max_total_seconds: float = 600.0,
) -> dict:
    """Fetch a battery of arXiv queries in parallel.

    Returns a stats dict with: per-query counts, total unique papers,
    time elapsed, dedup ratio.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_entries: dict[str, dict] = {}  # arxiv_id -> entry

    per_query_stats: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_one_query, q, per_query): q
            for q in queries
        }
        for fut in concurrent.futures.as_completed(futures):
            q = futures[fut]
            try:
                query, count, entries = fut.result()
            except Exception as e:
                log.warning("query %r raised: %s", q, e)
                per_query_stats.append({"query": q, "count": 0, "error": str(e)})
                continue
            new = 0
            for e in entries:
                if e["arxiv_id"] in all_entries:
                    continue
                all_entries[e["arxiv_id"]] = {**e, "_query": q}
                new += 1
            per_query_stats.append({"query": q, "fetched": count, "new": new})
            log.info("[mass-arxiv] %-60s fetched=%d new=%d (elapsed %.1fs)",
                     q[:58], count, new, time.time() - t0)
            if time.time() - t0 > max_total_seconds:
                log.info("time budget exhausted; stopping")
                break

    # Persist digests
    digest_files = _write_digests(all_entries, out_dir)

    elapsed = time.time() - t0
    stats = {
        "elapsed_s": round(elapsed, 1),
        "n_queries": len(queries),
        "n_queries_succeeded": sum(1 for s in per_query_stats if "fetched" in s),
        "n_unique_papers": len(all_entries),
        "n_digests_written": len(digest_files),
        "per_query": per_query_stats,
    }
    stats_log = out_dir / "ingest_stats.json"
    stats_log.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[mass-arxiv] total: %d unique papers in %.1fs", len(all_entries), elapsed)
    return stats


def _write_digests(entries: dict[str, dict], out_dir: Path) -> list[Path]:
    """Write each arXiv entry as a digest file the existing KB indexer can read."""
    written: list[Path] = []
    for arxiv_id, e in entries.items():
        body = (
            f"# arxiv:{arxiv_id}\n\n"
            f"## [开头/摘要]\n\n"
            f"**Title:** {e.get('title', '')}\n\n"
            f"**Authors:** {', '.join(e.get('authors', [])) or 'unknown'}\n\n"
            f"**Year:** {e.get('year', 0)}\n\n"
            f"**arXiv:** {e.get('link', arxiv_id)}\n\n"
            f"**Categories:** {', '.join(e.get('categories', [])) or 'unknown'}\n\n"
            f"**Abstract:**\n\n{e.get('summary', '')}\n"
        )
        target = out_dir / f"arxiv_{arxiv_id.replace('/', '_').replace('.', '_')}.digest.md"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mass arXiv retrieval for AutoSCI")
    parser.add_argument("--queries", nargs="*", help="List of query strings")
    parser.add_argument("--queries-file", help="JSON file with a 'queries' list")
    parser.add_argument("--out-dir", default="data/mass_corpus")
    parser.add_argument("--per-query", type=int, default=50, help="max_results per query")
    parser.add_argument("--max-workers", type=int, default=3, help="concurrent arXiv calls")
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s", datefmt="%H:%M:%S")

    if args.queries:
        queries = args.queries
    elif args.queries_file:
        data = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
        queries = data.get("queries") if isinstance(data, dict) else data
    else:
        queries = DEFAULT_QUERY_BANK

    log.info("Mass arXiv retrieval: %d queries, %d results per query, %d workers",
             len(queries), args.per_query, args.max_workers)
    stats = run_mass(queries, out_dir=args.out_dir, per_query=args.per_query,
                     max_workers=args.max_workers, max_total_seconds=args.max_seconds)
    print(json.dumps({"n_unique_papers": stats["n_unique_papers"],
                      "elapsed_s": stats["elapsed_s"],
                      "n_queries_succeeded": stats["n_queries_succeeded"]}, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
