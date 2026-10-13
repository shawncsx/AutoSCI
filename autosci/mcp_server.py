"""AutoSCI — MCP server (v0.13).

Exposes AutoSCI's pipeline + KB tools to any MCP-compatible client
(Trae, WorkBuddy, Codex, Cursor, Claude Desktop, Cline, etc.).

Run with::

    python -m autosci.mcp_server

Then point your IDE's MCP config at this server. Example
``claude_desktop_config.json``::

    {
      "mcpServers": {
        "autosci": {
          "command": "python",
          "args": ["-m", "autosci.mcp_server"]
        }
      }
    }

Tools exposed
-------------
    autosci_status                 — show KB / vault / model / running stats
    autosci_search_kb              — search the local corpus (BM25 + LLM rerank)
    autosci_run_one                — run a single research idea end-to-end
    autosci_propose_ideas          — IdeaMiner: mine new ideas from the corpus
    autosci_list_vault             — list kept/revised/refuted vault entries
    autosci_read_paper             — read a paper.md from the vault
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("autosci.mcp")


def _lazy_mcp():
    """Try to import the official MCP Python SDK; raise a clear error if absent."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        return Server, stdio_server, Tool, TextContent
    except Exception as e:
        raise RuntimeError(
            "MCP Python SDK not installed. Install with `pip install mcp`."
        ) from e


def _build_state(runs_dir: Path, vault_dir: Path, distill_path: Path) -> dict:
    """One-shot snapshot of the on-disk state."""
    snap: dict = {"ok": True}
    # KB size
    digest_dirs = []
    local = Path(r"D:\shawn\paper\_digest")
    if local.exists():
        digest_dirs.append(local)
    for d in ("data/mass_corpus", "data/openalex_corpus",
              "data/biorxiv_corpus", "data/pubmed_corpus",
              "data/semanticscholar_corpus", "data/pdf_fulltext"):
        p = Path(d)
        if p.exists():
            digest_dirs.append(p)
    total = 0
    for d in digest_dirs:
        total += sum(1 for _ in d.glob("*.digest.md")) + sum(1 for _ in d.glob("*.fulltext.md"))
    snap["kb_papers"] = total
    snap["kb_sources"] = [str(d) for d in digest_dirs]
    # Vault
    manifest_path = vault_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snap["vault_items"] = len(manifest.get("items", []))
            from collections import Counter
            vc = Counter(it.get("verdict", "?") for it in manifest.get("items", []))
            snap["vault_by_verdict"] = dict(vc)
        except Exception:
            snap["vault_items"] = -1
    else:
        snap["vault_items"] = 0
    # Model
    snap["model"] = os.environ.get("AGNES_MODEL", "agnes-3.0-flash")
    snap["base_url"] = os.environ.get("AGNES_BASE_URL", "https://api.agnes-ai.cn/v1")
    snap["runs_dir"] = str(runs_dir)
    snap["vault_dir"] = str(vault_dir)
    return snap


def _text(text: str) -> list[Any]:
    Server, _, _, TextContent = _lazy_mcp()
    return [TextContent(type="text", text=text)]


def make_server(runs_dir: Path, vault_dir: Path, distill_path: Path):
    Server, stdio_server, Tool, _ = _lazy_mcp()
    server = Server("autosci")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(name="autosci_status",
                 description="Show AutoSCI state: KB size, vault counts, model, paths.",
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="autosci_search_kb",
                 description="Search the local corpus (BM25 + LLM rerank). Returns up to top-k digest paths.",
                 inputSchema={"type": "object", "properties": {
                     "query": {"type": "string"},
                     "top_k": {"type": "integer", "default": 5},
                 }, "required": ["query"]}),
            Tool(name="autosci_propose_ideas",
                 description="IdeaMiner: mine N new ideas grounded in the corpus.",
                 inputSchema={"type": "object", "properties": {
                     "n": {"type": "integer", "default": 5},
                 }}),
            Tool(name="autosci_run_one",
                 description="Run a single research idea end-to-end through the Director pipeline.",
                 inputSchema={"type": "object", "properties": {
                     "idea_id": {"type": "string"},
                     "title": {"type": "string"},
                     "one_liner": {"type": "string"},
                     "angle": {"type": "string"},
                 }, "required": ["title", "one_liner", "angle"]}),
            Tool(name="autosci_list_vault",
                 description="List vault entries (kept/revised/refuted).",
                 inputSchema={"type": "object", "properties": {
                     "verdict": {"type": "string", "enum": ["kept", "revised", "refuted", "any"]},
                 }}),
            Tool(name="autosci_read_paper",
                 description="Read a paper.md from the vault by idea id.",
                 inputSchema={"type": "object", "properties": {
                     "idea_id": {"type": "string"},
                     "max_chars": {"type": "integer", "default": 12000},
                 }, "required": ["idea_id"]}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[Any]:
        from autosci.llm import LLMClient
        from autosci.kb import KnowledgeBase
        from autosci.schemas import IdeaSeed, Modality
        from autosci.agents.idea_miner import propose_wide_ideas

        if name == "autosci_status":
            return _text(json.dumps(_build_state(runs_dir, vault_dir, distill_path),
                                    ensure_ascii=False, indent=2))

        if name == "autosci_search_kb":
            q = arguments.get("query", "")
            k = int(arguments.get("top_k", 5))
            # build a minimal KB from local + mass sources
            digest_dirs = []
            local = Path(r"D:\shawn\paper\_digest")
            if local.exists():
                digest_dirs.append(local)
            for d in ("data/mass_corpus", "data/openalex_corpus", "data/biorxiv_corpus",
                      "data/pubmed_corpus", "data/semanticscholar_corpus", "data/pdf_fulltext"):
                p = Path(d)
                if p.exists():
                    digest_dirs.append(p)
            docs = []
            for d in digest_dirs:
                from autosci.kb import parse_digest
                for f in d.glob("*.digest.md"):
                    docs.append(parse_digest(f))
            base = KnowledgeBase.from_digest_dir(digest_dirs[0]) if digest_dirs else KnowledgeBase([])
            base.docs = docs
            base._bm25 = None
            hits = base.search(q, k=k)
            return _text(json.dumps([{"doc_id": d.doc_id, "title": d.title,
                                       "year": d.year} for d in hits],
                                      ensure_ascii=False, indent=2))

        if name == "autosci_propose_ideas":
            n = int(arguments.get("n", 5))
            llm = LLMClient()
            digest_dirs = []
            local = Path(r"D:\shawn\paper\_digest")
            if local.exists():
                digest_dirs.append(local)
            for d in ("data/mass_corpus", "data/openalex_corpus", "data/biorxiv_corpus",
                      "data/pubmed_corpus", "data/semanticscholar_corpus", "data/pdf_fulltext"):
                p = Path(d)
                if p.exists():
                    digest_dirs.append(p)
            base = KnowledgeBase.from_digest_dir(digest_dirs[0]) if digest_dirs else KnowledgeBase([])
            from autosci.kb import parse_digest
            base.docs = [parse_digest(f) for d in digest_dirs for f in d.glob("*.digest.md")]
            base._bm25 = None
            ideas = propose_wide_ideas(base, llm, total_n=n)
            return _text(json.dumps([s.model_dump() for s in ideas],
                                      ensure_ascii=False, indent=2))

        if name == "autosci_run_one":
            seed = IdeaSeed(
                id=arguments.get("idea_id", f"mcp_idea"),
                title=arguments.get("title", "Untitled"),
                one_liner=arguments.get("one_liner", ""),
                angle=arguments.get("angle", ""),
                modalities=[Modality.SSL],
            )
            from autosci.pipeline import ResearchTeam, RunConfig
            llm = LLMClient()
            cfg = RunConfig(output_root=runs_dir)
            from autosci.kb import KnowledgeBase, parse_digest
            digest_dirs = [Path(r"D:\shawn\paper\_digest")] + [
                Path(d) for d in ("data/mass_corpus", "data/openalex_corpus", "data/biorxiv_corpus",
                              "data/pubmed_corpus", "data/semanticscholar_corpus", "data/pdf_fulltext")
                if Path(d).exists()
            ]
            base = KnowledgeBase.from_digest_dir(digest_dirs[0])
            base.docs = [parse_digest(f) for d in digest_dirs for f in d.glob("*.digest.md")]
            base._bm25 = None
            team = ResearchTeam(llm, base, cfg)
            state = team.run_one(seed)
            from autosci.vault import Vault
            v = Vault(root=vault_dir)
            v.bootstrap()
            try:
                v.record(state, paper_md_path=Path(state.paper_path) if state.paper_path else None)
            except Exception as e:
                log.warning("vault.record failed: %s", e)
            return _text(json.dumps({"id": seed.id, "verdict": state.verdict.value,
                                       "history": state.history_log[-6:]},
                                      ensure_ascii=False, indent=2))

        if name == "autosci_list_vault":
            verdict_filter = arguments.get("verdict", "any")
            manifest_path = vault_dir / "manifest.json"
            if not manifest_path.exists():
                return _text("(empty vault)")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            items = manifest.get("items", [])
            if verdict_filter != "any":
                items = [it for it in items if it.get("verdict") == verdict_filter]
            return _text(json.dumps(items[-20:], ensure_ascii=False, indent=2))

        if name == "autosci_read_paper":
            iid = arguments.get("idea_id", "")
            max_chars = int(arguments.get("max_chars", 12000))
            for sub in ("kept", "refuted"):
                p = vault_dir / sub / iid / "paper.md"
                if p.exists():
                    return _text(p.read_text(encoding="utf-8", errors="replace")[:max_chars])
            return _text(f"(no paper.md for {iid})")

        return _text(f"unknown tool: {name}")

    return server, stdio_server


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s :: %(message)s", datefmt="%H:%M:%S")
    runs_dir = Path(os.environ.get("AUTOSCI_RUNS_DIR", "runs"))
    vault_dir = Path(os.environ.get("AUTOSCI_VAULT_DIR", "vault"))
    distill_path = Path(os.environ.get("AUTOSCI_DISTILL", "data/distillation.json"))
    server, stdio_server = make_server(runs_dir, vault_dir, distill_path)
    log.info("AutoSCI MCP server ready (stdio); tools: autosci_status, autosci_search_kb, "
             "autosci_propose_ideas, autosci_run_one, autosci_list_vault, autosci_read_paper")
    import asyncio
    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
