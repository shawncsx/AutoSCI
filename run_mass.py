"""AutoSCI — Mass production entry point (v0.8).

One-click: build a 1000+ paper corpus, propose 20+ ideas, run them
all through the Director pipeline, compile a final report.

Pipeline:
    1. Mass arXiv retrieval: 30+ queries × 50 results each → 1000+ papers
    2. OpenAlex fallback for cross-domain papers (clinical, materials, …)
    3. Combine with the local 100-paper microbiome corpus
    4. Wide IdeaMiner: 20 fresh ideas
    5. Director in parallel mode: 4 concurrent idea runs
    6. Compile vault/kept/* into a master HTML report

Usage::

    # Default: 30 queries × 50 = up to 1500 papers, 20 ideas, parallel=4
    python run_mass.py

    # Smaller, faster demo
    python run_mass.py --per-query 20 --ideas 10 --parallel 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _setup_logging() -> None:
    # Silence noisy library loggers; rich handles our visual output.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoSCI mass production")
    parser.add_argument("--per-query", type=int, default=50,
                        help="arXiv max_results per query (default 50)")
    parser.add_argument("--queries-file", default=None,
                        help="Path to JSON with 'queries' list; default = built-in bank")
    parser.add_argument("--openalex-queries", nargs="*", default=[],
                        help="Optional list of OpenAlex queries for cross-domain coverage")
    parser.add_argument("--ideas", type=int, default=20,
                        help="How many ideas to mine and run (default 20)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="How many ideas to run concurrently through Director (default 4)")
    parser.add_argument("--max-seconds", type=int, default=900,
                        help="Hard cap on the arXiv retrieval stage (default 15 min)")
    parser.add_argument("--runs-dir", default="runs_mass")
    parser.add_argument("--vault-dir", default="vault_mass")
    parser.add_argument("--no-arxiv", action="store_true", help="skip arXiv retrieval")
    parser.add_argument("--no-openalex", action="store_true", help="skip OpenAlex")
    parser.add_argument("--no-biorxiv", action="store_true", help="skip bioRxiv")
    parser.add_argument("--no-pubmed", action="store_true", help="skip PubMed/PMC")
    parser.add_argument("--no-semanticscholar", action="store_true", help="skip Semantic Scholar")
    parser.add_argument("--biorxiv-queries", nargs="*", default=[
        "microbiome", "metagenomics", "16S microbiome", "microbiome foundation model"
    ], help="Queries for bioRxiv search endpoint")
    parser.add_argument("--pubmed-queries", nargs="*", default=[
        "microbiome machine learning", "metagenomic foundation model",
        "colorectal cancer microbiome", "gut microbiome deep learning",
    ], help="Queries for PubMed")
    parser.add_argument("--s2-queries", nargs="*", default=[
        "microbiome foundation model", "graph neural network microbiome",
        "transformer metagenomics", "microbiome self-supervised",
    ], help="Queries for Semantic Scholar")
    parser.add_argument("--s2-api-key", default="", help="Optional S2 API key (raises rate limit)")
    parser.add_argument("--keep-existing-kb", action="store_true", default=True,
                        help="add the new papers to the existing local KB instead of replacing")
    args = parser.parse_args(argv)

    _setup_logging()
    log = logging.getLogger("autosci.mass")

    from autosci.director import Director
    from autosci.llm import LLMClient
    from autosci.kb import KnowledgeBase
    from autosci.pipeline import RunConfig
    from autosci.vault import Vault
    from autosci.memory import DirectorMemory
    from autosci.rich_console import (
        banner, section, status, success, error, verdict_table, console,
    )

    console.print()
    banner()

    paper_root = Path(r"D:\shawn\paper")
    local_digest = paper_root / "_digest"
    if not local_digest.exists():
        log.warning("Local digest dir not found at %s", local_digest)

    # ------------------------------------------------------------------
    # 1. Mass arXiv retrieval
    # ------------------------------------------------------------------
    mass_dir = Path("data/mass_corpus")
    n_arxiv = 0
    if not args.no_arxiv:
        from autosci.mass_ingestor import run_mass as run_mass_arxiv, DEFAULT_QUERY_BANK
        if args.queries_file:
            data = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
            queries = data.get("queries") if isinstance(data, dict) else data
        else:
            queries = DEFAULT_QUERY_BANK
        section("STEP 1 — Mass arXiv retrieval")
        console.print(f"  [grey58]{len(queries)} queries × {args.per_query} results, "
                      f"max_workers={min(args.parallel, 4)}[/]")
        with status("Fetching from arXiv…"):
            stats = run_mass_arxiv(
                queries, out_dir=mass_dir,
                per_query=args.per_query,
                max_workers=min(args.parallel, 4),
                max_total_seconds=args.max_seconds,
            )
        n_arxiv = stats["n_unique_papers"]
        console.print(f"  [green]✓[/] {n_arxiv} unique arXiv papers in {stats['elapsed_s']}s")

    # ------------------------------------------------------------------
    # 2. OpenAlex cross-domain
    # ------------------------------------------------------------------
    n_oa = 0
    if not args.no_openalex and args.openalex_queries:
        from autosci.openalex import fetch_openalex, write_as_digests
        oa_dir = Path("data/openalex_corpus")
        oa_dir.mkdir(parents=True, exist_ok=True)
        section("STEP 2 — OpenAlex cross-domain retrieval")
        for q in args.openalex_queries:
            with status(f"OpenAlex: {q}…"):
                works = fetch_openalex(q, max_results=50)
            files = write_as_digests(works, oa_dir)
            n_oa += len(files)
            console.print(f"  [grey58]•[/] {q[:50]}: {len(files)} digests")
        console.print(f"  [green]✓[/] {n_oa} OpenAlex digests total")

    # ------------------------------------------------------------------
    # 2b. bioRxiv retrieval
    # ------------------------------------------------------------------
    n_biorxiv = 0
    if not args.no_biorxiv:
        from autosci.biorxiv import fetch_biorxiv_search, write_as_digests as biorxiv_write
        bx_dir = Path("data/biorxiv_corpus")
        bx_dir.mkdir(parents=True, exist_ok=True)
        section("STEP 2b — bioRxiv retrieval")
        for q in args.biorxiv_queries:
            with status(f"bioRxiv: {q}…"):
                papers = fetch_biorxiv_search(q, max_records=30)
            files = biorxiv_write(papers, bx_dir)
            n_biorxiv += len(files)
            console.print(f"  [grey58]•[/] {q[:50]}: {len(files)} digests")
        console.print(f"  [green]✓[/] {n_biorxiv} bioRxiv digests total")

    # ------------------------------------------------------------------
    # 2c. PubMed retrieval
    # ------------------------------------------------------------------
    n_pubmed = 0
    if not args.no_pubmed:
        from autosci.pubmed import search_pubmed, write_as_digests as pubmed_write
        pm_dir = Path("data/pubmed_corpus")
        pm_dir.mkdir(parents=True, exist_ok=True)
        section("STEP 2c — PubMed retrieval (Cell / Nature / Lancet…)")
        for q in args.pubmed_queries:
            with status(f"PubMed: {q}…"):
                papers = search_pubmed(q, max_records=30, mindate="2022/01/01")
            files = pubmed_write(papers, pm_dir)
            n_pubmed += len(files)
            console.print(f"  [grey58]•[/] {q[:50]}: {len(files)} digests")
        console.print(f"  [green]✓[/] {n_pubmed} PubMed digests total")

    # ------------------------------------------------------------------
    # 2d. Semantic Scholar retrieval
    # ------------------------------------------------------------------
    n_s2 = 0
    if not args.no_semanticscholar:
        from autosci.semanticscholar import fetch_s2, write_as_digests as s2_write
        s2_dir = Path("data/semanticscholar_corpus")
        s2_dir.mkdir(parents=True, exist_ok=True)
        section("STEP 2d — Semantic Scholar retrieval (ICML/NeurIPS/CVPR…)")
        for q in args.s2_queries:
            with status(f"S2: {q}…"):
                papers = fetch_s2(q, max_results=30, year_from="2022", api_key=args.s2_api_key)
            files = s2_write(papers, s2_dir)
            n_s2 += len(files)
            console.print(f"  [grey58]•[/] {q[:50]}: {len(files)} digests")
        console.print(f"  [green]✓[/] {n_s2} Semantic Scholar digests total")

    # ------------------------------------------------------------------
    # 3. Build combined KB
    # ------------------------------------------------------------------
    log.info("STEP 3: Build combined KB")
    digest_dirs = []
    if local_digest.exists() and args.keep_existing_kb:
        digest_dirs.append(local_digest)
    if mass_dir.exists():
        digest_dirs.append(mass_dir)
    oa_dir = Path("data/openalex_corpus")
    if oa_dir.exists():
        digest_dirs.append(oa_dir)
    bx_dir = Path("data/biorxiv_corpus")
    if bx_dir.exists():
        digest_dirs.append(bx_dir)
    pm_dir = Path("data/pubmed_corpus")
    if pm_dir.exists():
        digest_dirs.append(pm_dir)
    s2_dir = Path("data/semanticscholar_corpus")
    if s2_dir.exists():
        digest_dirs.append(s2_dir)

    # merge all into one big KB
    from autosci.kb import Doc
    all_docs: list[Doc] = []
    for d in digest_dirs:
        from autosci.kb import parse_digest
        for f in sorted(d.glob("*.digest.md")):
            all_docs.append(parse_digest(f))
    log.info("  → combined KB has %d docs", len(all_docs))

    # build a KnowledgeBase from the merged docs (use from_digest_dir on a
    # temp dir trick: use the first dir to seed, then mutate .docs)
    base_kb = KnowledgeBase.from_digest_dir(digest_dirs[0]) if digest_dirs else KnowledgeBase([])
    base_kb.docs = all_docs
    base_kb._bm25 = None
    kb = base_kb

    # ------------------------------------------------------------------
    # 4. Wide IdeaMiner
    # ------------------------------------------------------------------
    section("STEP 4 — Wide IdeaMiner")
    console.print(f"  [grey58]target: {args.ideas} ideas[/]")
    from autosci.agents.idea_miner import propose_wide_ideas, dedupe_by_title
    with status(f"Mining {args.ideas} ideas (8 angles)…"):
        ideas = propose_wide_ideas(kb, llm, total_n=args.ideas)
    ideas = dedupe_by_title(ideas)
    console.print(f"  [green]✓[/] {len(ideas)} fresh ideas")

    # ------------------------------------------------------------------
    # 5. Director in parallel mode
    # ------------------------------------------------------------------
    section("STEP 5 — Director in parallel")
    console.print(f"  [grey58]parallel={args.parallel}, ideas={len(ideas)}[/]")
    cfg = RunConfig(output_root=Path(args.runs_dir), kb_topk=6)
    vault = Vault(root=Path(args.vault_dir))
    mem = DirectorMemory(path=Path(args.runs_dir) / "director_memory.jsonl")
    director = Director(
        llm=llm, kb=kb, cfg=cfg, vault=vault,
        memory=mem, parallel=args.parallel, auto_distill_every=5,
    )
    t0 = time.time()
    with status(f"Running Director on {len(ideas)} ideas (parallel={args.parallel})…"):
        rep = director.run_all(ideas)
    elapsed = time.time() - t0
    console.print(f"  [green]✓[/] Director done in {elapsed:.1f}s")
    console.print(verdict_table(rep.per_idea))

    # ------------------------------------------------------------------
    # 6. Compile final report
    # ------------------------------------------------------------------
    section("STEP 6 — Compile final report")
    compilation_path = Path(args.vault_dir) / "compilation.html"
    from autosci.compilation import build_compilation_html
    html = build_compilation_html(Path(args.vault_dir) / "kept", kept_only=True)
    compilation_path.write_text(html, encoding="utf-8")
    console.print(f"  [green]✓[/] {compilation_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    summary = {
        "elapsed_s": round(elapsed, 1),
        "n_corpus_arxiv": n_arxiv,
        "n_corpus_openalex": n_oa,
        "n_corpus_biorxiv": n_biorxiv,
        "n_corpus_pubmed": n_pubmed,
        "n_corpus_s2": n_s2,
        "n_corpus_total": len(all_docs),
        "n_ideas_attempted": rep.n_total,
        "n_kept": rep.n_kept,
        "n_revised": rep.n_revised,
        "n_refuted": rep.n_refuted,
        "n_failed": rep.n_failed,
        "tokens_in": rep.tokens_in,
        "tokens_out": rep.tokens_out,
        "compilation_path": str(compilation_path),
    }
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.runs_dir, "mass_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    section("Summary")
    console.print_json(json.dumps(summary, ensure_ascii=False, indent=2))
    success("Mass run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
