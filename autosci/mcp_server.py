"""AutoSCI — MCP server (v0.14, passive-mode-first).

This server exposes TWO classes of tools to any MCP-compatible client
(Trae, WorkBuddy, Codex, Cursor, Claude Desktop, Cline, …):

    ACTIVE tools (need AGNES_API_KEY, AutoSCI's own LLM does the work):
        autosci_status
        autosci_search_kb
        autosci_propose_ideas
        autosci_run_one
        autosci_list_vault
        autosci_read_paper

    PASSIVE tools (no LLM needed — IDE's LLM does the work using the
    templates + data we return):
        autosci_idea_anchors         — top-k KB chunks to ground an idea
        autosci_idea_template        — IdeaSeed JSON template
        autosci_pipeline_prompts     — system+user prompts for all 10 agents
        autosci_section_template     — IMRaD section template
        autosci_review_template      — Reviewer/Refuter scoring template
        autosci_save_idea_draft      — write a draft to vault
        autosci_save_paper_section   — append/update a single section

If your IDE has its own LLM (Claude Desktop = Claude, Cursor = any),
you don't need AutoSCI's API key at all. You just:

    1. Connect this MCP server to your IDE
    2. Tell the IDE: "use autosci_idea_anchors + autosci_idea_template to
       propose 5 ideas. Then for the best one, use autosci_pipeline_prompts
       to get the prompts and run each step yourself. Save with autosci_save_idea_draft."

The user prompt templates are designed to work with any chat-completion
API (system / user format). The IDE's LLM just fills them in.

Run with:
    python -m autosci.mcp_server
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
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        return Server, stdio_server, Tool, TextContent
    except Exception as e:
        raise RuntimeError(
            "MCP Python SDK not installed. Install with `pip install mcp`."
        ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_digest_dirs() -> list[Path]:
    dirs = []
    local = Path(r"D:\shawn\paper\_digest")
    if local.exists():
        dirs.append(local)
    for d in ("data/mass_corpus", "data/openalex_corpus", "data/biorxiv_corpus",
              "data/pubmed_corpus", "data/semanticscholar_corpus", "data/pdf_fulltext"):
        p = Path(d)
        if p.exists():
            dirs.append(p)
    return dirs


def _build_state(runs_dir: Path, vault_dir: Path, distill_path: Path) -> dict:
    snap: dict = {"ok": True}
    digest_dirs = _collect_digest_dirs()
    total = sum(
        sum(1 for _ in d.glob("*.digest.md")) + sum(1 for _ in d.glob("*.fulltext.md"))
        for d in digest_dirs
    )
    snap["kb_papers"] = total
    snap["kb_sources"] = [str(d) for d in digest_dirs]
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
    snap["llm_required"] = bool(os.environ.get("AGNES_API_KEY"))
    snap["model"] = os.environ.get("AGNES_MODEL", "agnes-3.0-flash")
    snap["base_url"] = os.environ.get("AGNES_BASE_URL", "https://api.agnes-ai.cn/v1")
    snap["runs_dir"] = str(runs_dir)
    snap["vault_dir"] = str(vault_dir)
    return snap


def _text(text: str) -> list[Any]:
    Server, _, _, TextContent = _lazy_mcp()
    return [TextContent(type="text", text=text)]


def _all_digest_docs() -> list:
    """Load every *.digest.md (and *.fulltext.md) under our known dirs."""
    from autosci.kb import parse_digest
    out = []
    for d in _collect_digest_dirs():
        for f in d.glob("*.digest.md"):
            try:
                out.append(parse_digest(f))
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Passive-mode tools (no LLM needed)
# ---------------------------------------------------------------------------


def _kb_search(query: str, k: int) -> list[dict]:
    from autosci.kb import KnowledgeBase, parse_digest
    digest_dirs = _collect_digest_dirs()
    if not digest_dirs:
        return []
    docs = []
    for d in digest_dirs:
        for f in d.glob("*.digest.md"):
            docs.append(parse_digest(f))
    base = KnowledgeBase.from_digest_dir(digest_dirs[0])
    base.docs = docs
    base._bm25 = None
    hits = base.search(query, k=k)
    return [{"doc_id": d.doc_id, "title": d.title, "year": d.year,
             "abstract_excerpt": (d.sections.get("开头/摘要") or next(iter(d.sections.values()), ""))[:600]}
            for d in hits]


def _pipeline_prompts() -> dict[str, dict[str, str]]:
    """Return {agent: {system, user_template}} for all 10 steps.

    The user_template is a short description of what the IDE LLM
    should fill in. We do NOT pre-render the Jinja2 source — the
    IDE LLM is expected to fill it in itself.
    """
    from autosci.prompts import export_pipeline_prompts
    return export_pipeline_prompts()


def _save_idea_draft(idea_id: str, title: str, one_liner: str, angle: str,
                       modalities: list[str] | str, paper_corpus_hint: str,
                       vault_dir: Path) -> dict:
    """Write a new idea to the vault's `ideas/` folder so Director can pick it up later."""
    ideas_dir = vault_dir / "ideas"
    ideas_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(modalities, str):
        modalities = [m.strip() for m in modalities.split(",") if m.strip()]
    data = {
        "id": idea_id,
        "title": title,
        "one_liner": one_liner,
        "angle": angle,
        "modalities": modalities,
        "priority": 5,
        "paper_corpus_hint": paper_corpus_hint,
    }
    out = ideas_dir / f"{idea_id}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(out), "idea": data}


def _save_paper_section(idea_id: str, section: str, content: str, verdict: str,
                        vault_dir: Path) -> dict:
    """Write a paper section to the vault (kept or refuted bucket)."""
    bucket = "kept" if verdict in ("kept", "revised") else "refuted"
    target = vault_dir / bucket / idea_id / "paper.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
    new = _upsert_section(existing, section, content)
    target.write_text(new, encoding="utf-8")
    return {"path": str(target), "section": section, "verdict": verdict}


def _upsert_section(text: str, section: str, content: str) -> str:
    """Insert or replace a `## {section}` block in `text`."""
    pattern = f"## {section}\n"
    if pattern in text:
        before, _, after = text.partition(pattern)
        # find the next `## ` heading or EOF
        idx = after.find("\n## ")
        body = after[:idx] if idx >= 0 else after
        new_body = f"\n{content.strip()}\n\n"
        return before + pattern + new_body + (after[idx:] if idx >= 0 else "")
    # else: append
    sep = "" if text.endswith("\n") or not text else "\n"
    return text + sep + f"## {section}\n\n{content.strip()}\n\n"


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def make_server(runs_dir: Path, vault_dir: Path, distill_path: Path):
    Server, stdio_server, Tool, _ = _lazy_mcp()
    server = Server("autosci")

    @server.list_tools()
    async def list_tools():
        return [
            # === ACTIVE tools (require AGNES_API_KEY) ===
            Tool(name="autosci_status",
                 description="Show AutoSCI state: KB size, vault counts, model, paths.",
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="autosci_search_kb",
                 description="Search the local corpus (BM25). Returns up to top-k digest paths with abstract excerpts.",
                 inputSchema={"type": "object", "properties": {
                     "query": {"type": "string"},
                     "top_k": {"type": "integer", "default": 5},
                 }, "required": ["query"]}),
            Tool(name="autosci_propose_ideas",
                 description="Use AutoSCI's LLM to mine N new ideas grounded in the corpus. Requires AGNES_API_KEY.",
                 inputSchema={"type": "object", "properties": {
                     "n": {"type": "integer", "default": 5},
                 }}),
            Tool(name="autosci_run_one",
                 description="Run a single research idea end-to-end through AutoSCI's pipeline. Requires AGNES_API_KEY.",
                 inputSchema={"type": "object", "properties": {
                     "idea_id": {"type": "string"},
                     "title": {"type": "string"},
                     "one_liner": {"type": "string"},
                     "angle": {"type": "string"},
                     "modalities": {"type": "array", "items": {"type": "string"}},
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

            # === PASSIVE tools (no LLM needed — IDE LLM does work) ===
            Tool(name="autosci_idea_anchors",
                 description=(
                     "[PASSIVE] Return top-k KB abstracts (no LLM call) to ground a new idea. "
                     "The IDE's LLM should use these as anchors to propose 3-5 ideas."
                 ),
                 inputSchema={"type": "object", "properties": {
                     "query": {"type": "string"},
                     "top_k": {"type": "integer", "default": 10},
                 }, "required": ["query"]}),
            Tool(name="autosci_idea_template",
                 description=(
                     "[PASSIVE] Return the IdeaSeed JSON schema + example. "
                     "The IDE's LLM should fill in the placeholder fields to produce new ideas."
                 ),
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="autosci_pipeline_prompts",
                 description=(
                     "[PASSIVE] Return the system+user prompt templates for all 10 pipeline agents. "
                     "The IDE's LLM should walk through each step, calling its own chat-completion "
                     "API to fill in the JSON, then call autosci_save_paper_section between steps."
                 ),
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="autosci_section_template",
                 description=(
                     "[PASSIVE] Return the IMRaD section template (introduction, related_work, method, "
                     "experiments, results_template, discussion, limitations, conclusion) so the IDE's LLM "
                     "knows what to write."
                 ),
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="autosci_review_template",
                 description=(
                     "[PASSIVE] Return the Reviewer/Refuter scoring template. IDE's LLM should fill it "
                     "in to evaluate a draft."
                 ),
                 inputSchema={"type": "object", "properties": {}}),
            Tool(name="autosci_save_idea_draft",
                 description=(
                     "Save a proposed idea to the vault's ideas/ folder so Director (or future runs) "
                     "can pick it up."
                 ),
                 inputSchema={"type": "object", "properties": {
                     "idea_id": {"type": "string"},
                     "title": {"type": "string"},
                     "one_liner": {"type": "string"},
                     "angle": {"type": "string"},
                     "modalities": {"type": "array", "items": {"type": "string"}},
                     "paper_corpus_hint": {"type": "string", "default": ""},
                 }, "required": ["idea_id", "title", "one_liner", "angle"]}),
            Tool(name="autosci_save_paper_section",
                 description="Append or update a single IMRaD section in vault/kept/<id>/paper.md.",
                 inputSchema={"type": "object", "properties": {
                     "idea_id": {"type": "string"},
                     "section": {"type": "string", "enum": [
                         "Abstract", "Introduction", "Related Work", "Method",
                         "Experiments", "Results Template", "Discussion",
                         "Limitations", "Conclusion"
                     ]},
                     "content": {"type": "string"},
                     "verdict": {"type": "string", "default": "kept",
                                 "enum": ["kept", "revised", "refuted"]},
                 }, "required": ["idea_id", "section", "content"]}),

            # ── v0.15: deep analysis + idea scoring + themes ──
            Tool(name="autosci_analyze_paper",
                 description=(
                     "[PASSIVE] Return a structured 5-section analysis of a paper "
                     "(research question / method / data / main result / limitations + opportunities). "
                     "No LLM — pure heuristic extraction from the KB digest. "
                     "If doc_id is empty, picks the first KB paper."
                 ),
                 inputSchema={"type": "object", "properties": {
                     "doc_id": {"type": "string", "default": ""},
                 }}),
            Tool(name="autosci_analyze_corpus",
                 description=(
                     "[PASSIVE] Return structured analysis for the top-N papers in the KB."
                 ),
                 inputSchema={"type": "object", "properties": {
                     "top": {"type": "integer", "default": 5},
                 }}),
            Tool(name="autosci_score_idea",
                 description=(
                     "[PASSIVE] Score an idea on novelty / feasibility / impact (0-1 each) and "
                     "return 3 variants (extension / break / cross-field). No LLM call."
                 ),
                 inputSchema={"type": "object", "properties": {
                     "title": {"type": "string"},
                     "one_liner": {"type": "string"},
                     "angle": {"type": "string"},
                     "modalities": {"type": "array", "items": {"type": "string"}},
                     "priority": {"type": "integer", "default": 5},
                     "paper_corpus_hint": {"type": "string", "default": ""},
                 }, "required": ["title", "one_liner", "angle"]}),
            Tool(name="autosci_analyze_themes",
                 description=(
                     "[PASSIVE] Cluster the KB into 3-5 themes. For each theme: label, "
                     "core paper, year range, trending flag, gap suggestion. No LLM call."
                 ),
                 inputSchema={"type": "object", "properties": {
                     "top_n": {"type": "integer", "default": 5},
                 }}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[Any]:
        # ── ACTIVE tools ──
        if name == "autosci_status":
            return _text(json.dumps(_build_state(runs_dir, vault_dir, distill_path),
                                    ensure_ascii=False, indent=2))

        if name == "autosci_search_kb":
            q = arguments.get("query", "")
            k = int(arguments.get("top_k", 5))
            return _text(json.dumps(_kb_search(q, k), ensure_ascii=False, indent=2))

        if name == "autosci_propose_ideas":
            n = int(arguments.get("n", 5))
            try:
                from autosci.llm import LLMClient
                from autosci.kb import KnowledgeBase
                from autosci.agents.idea_miner import propose_wide_ideas
                llm = LLMClient()
                digest_dirs = _collect_digest_dirs()
                base = KnowledgeBase.from_digest_dir(digest_dirs[0]) if digest_dirs else KnowledgeBase([])
                from autosci.kb import parse_digest
                base.docs = [parse_digest(f) for d in digest_dirs for f in d.glob("*.digest.md")]
                base._bm25 = None
                ideas = propose_wide_ideas(base, llm, total_n=n)
                return _text(json.dumps([s.model_dump() for s in ideas],
                                      ensure_ascii=False, indent=2))
            except Exception as e:
                return _text(f"autosci_propose_ideas needs AGNES_API_KEY. ({e})")

        if name == "autosci_run_one":
            try:
                from autosci.llm import LLMClient
                from autosci.pipeline import ResearchTeam, RunConfig
                from autosci.schemas import IdeaSeed, Modality
                from autosci.kb import KnowledgeBase, parse_digest
                from autosci.vault import Vault
                seed = IdeaSeed(
                    id=arguments.get("idea_id", "mcp_idea"),
                    title=arguments["title"],
                    one_liner=arguments["one_liner"],
                    angle=arguments["angle"],
                    modalities=[Modality(m) for m in arguments.get("modalities", []) if m in Modality.__members__.values() or m in [e.value for e in Modality]],
                )
                llm = LLMClient()
                cfg = RunConfig(output_root=runs_dir)
                digest_dirs = _collect_digest_dirs()
                base = KnowledgeBase.from_digest_dir(digest_dirs[0])
                base.docs = [parse_digest(f) for d in digest_dirs for f in d.glob("*.digest.md")]
                base._bm25 = None
                team = ResearchTeam(llm, base, cfg)
                state = team.run_one(seed)
                v = Vault(root=vault_dir); v.bootstrap()
                try:
                    v.record(state, paper_md_path=Path(state.paper_path) if state.paper_path else None)
                except Exception as e:
                    log.warning("vault.record failed: %s", e)
                return _text(json.dumps({"id": seed.id, "verdict": state.verdict.value,
                                          "history": state.history_log[-6:]},
                                         ensure_ascii=False, indent=2))
            except Exception as e:
                return _text(f"autosci_run_one needs AGNES_API_KEY. ({e})")

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

        # ── PASSIVE tools (no LLM needed) ──
        if name == "autosci_idea_anchors":
            q = arguments.get("query", "")
            k = int(arguments.get("top_k", 10))
            return _text(json.dumps({"query": q, "anchors": _kb_search(q, k)},
                                    ensure_ascii=False, indent=2))

        if name == "autosci_idea_template":
            return _text(json.dumps({
                "schema": {
                    "id": "kebab-case-unique",
                    "title": "One-line paper title",
                    "one_liner": "≤30 char motivation",
                    "angle": "2-3 sentence novelty",
                    "modalities": ["graph_model", "ssl", "compositional",
                                    "cross_cohort", "threshold_sensitivity",
                                    "multimodal_fusion"],
                    "priority": 1,
                    "paper_corpus_hint": "search keywords",
                },
                "example": {
                    "id": "perm_invariant_gat_film",
                    "title": "Permutation-Invariant GAT-FiLM for Microbiome Cross-Cohort Robustness",
                    "one_liner": "Use GAT + FiLM to replace rank-encoded sequence Transformer",
                    "angle": "First graph-based permutation-invariant model with direct abundance fusion",
                    "modalities": ["graph_model", "multimodal_fusion"],
                    "priority": 8,
                    "paper_corpus_hint": "graph neural network microbiome FiLM abundance",
                },
                "instructions": (
                    "Read autosci_idea_anchors(query=...) first. Combine the "
                    "abstract excerpts into 3-5 ideas. For each, fill the schema above. "
                    "Then call autosci_save_idea_draft for each."
                ),
            }, ensure_ascii=False, indent=2))

        if name == "autosci_pipeline_prompts":
            return _text(json.dumps(_pipeline_prompts(), ensure_ascii=False, indent=2))

        if name == "autosci_section_template":
            from autosci.prompts import REQUIRED_SECTIONS
            return _text(json.dumps({
                "required_sections_in_order": list(REQUIRED_SECTIONS),
                "instructions": (
                    "For each section, write 250-800 words in academic English. "
                    "Cite [cite_key] for any claim. End with the section header on its own line, "
                    "then the body. Call autosci_save_paper_section after each."
                ),
            }, ensure_ascii=False, indent=2))

        if name == "autosci_review_template":
            return _text(json.dumps({
                "schema": {
                    "summary": "1-paragraph critique",
                    "line_items": [
                        {"section": "<name>", "anchor": "<quoted>",
                         "severity": "minor|major|blocking",
                         "suggestion": "what to fix"}
                    ],
                    "severity_counts": {"minor": 0, "major": 0, "blocking": 0},
                },
                "instructions": (
                    "Read the full paper. Output at least 5 line-items, each with a "
                    "specific anchor quote. Severity must be one of minor / major / blocking. "
                    "Sum severity_counts from line_items."
                ),
            }, ensure_ascii=False, indent=2))

        if name == "autosci_save_idea_draft":
            return _text(json.dumps(_save_idea_draft(
                idea_id=arguments["idea_id"],
                title=arguments["title"],
                one_liner=arguments["one_liner"],
                angle=arguments["angle"],
                modalities=arguments.get("modalities", []),
                paper_corpus_hint=arguments.get("paper_corpus_hint", ""),
                vault_dir=vault_dir,
            ), ensure_ascii=False, indent=2))

        if name == "autosci_save_paper_section":
            return _text(json.dumps(_save_paper_section(
                idea_id=arguments["idea_id"],
                section=arguments["section"],
                content=arguments["content"],
                verdict=arguments.get("verdict", "kept"),
                vault_dir=vault_dir,
            ), ensure_ascii=False, indent=2))

        # ── v0.15: deep analysis + idea scoring + theme discovery ──
        if name == "autosci_analyze_paper":
            from autosci.paper_analyzer import analyze_doc
            target_id = arguments.get("doc_id", "")
            target = next((d for d in _kb_search("", k=999) for _ in [None]), None)  # noqa
            # try the digest files
            for d in _all_digest_docs():
                if d.doc_id == target_id or not target_id:
                    a = analyze_doc(d)
                    return _text(json.dumps(a.to_dict(), ensure_ascii=False, indent=2))
            return _text(f"(no doc found for {target_id})")

        if name == "autosci_analyze_corpus":
            from autosci.paper_analyzer import analyze_corpus
            docs = _all_digest_docs()
            top = int(arguments.get("top", 5))
            analyses = analyze_corpus(docs, top_n=top)
            return _text(json.dumps([a.to_dict() for a in analyses],
                                    ensure_ascii=False, indent=2))

        if name == "autosci_score_idea":
            from autosci.idea_scorer import score_idea, make_variants
            from autosci.schemas import IdeaSeed, Modality
            seed = IdeaSeed(
                id=arguments.get("idea_id", "user_idea"),
                title=arguments["title"],
                one_liner=arguments.get("one_liner", ""),
                angle=arguments.get("angle", ""),
                modalities=[Modality(m) for m in arguments.get("modalities", [])],
                priority=int(arguments.get("priority", 5)),
                paper_corpus_hint=arguments.get("paper_corpus_hint", ""),
            )
            docs = _all_digest_docs()
            score = score_idea(seed, docs)
            variants = make_variants(seed)
            return _text(json.dumps({
                "score": score,
                "variants": [v.model_dump() for v in variants],
            }, ensure_ascii=False, indent=2))

        if name == "autosci_analyze_themes":
            from autosci.theme_analyzer import analyze_themes
            docs = _all_digest_docs()
            n = int(arguments.get("top_n", 5))
            ta = analyze_themes(docs, target_themes=n)
            return _text(json.dumps(ta.to_dict(), ensure_ascii=False, indent=2))

        return _text(f"unknown tool: {name}")

    return server, stdio_server


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s :: %(message)s", datefmt="%H:%M:%S")
    runs_dir = Path(os.environ.get("AUTOSCI_RUNS_DIR", "runs"))
    vault_dir = Path(os.environ.get("AUTOSCI_VAULT_DIR", "vault"))
    distill_path = Path(os.environ.get("AUTOSCI_DISTILL", "data/distillation.json"))
    server, stdio_server = make_server(runs_dir, vault_dir, distill_path)
    log.info("AutoSCI MCP server v0.14 ready (stdio); 12 tools: "
             "6 ACTIVE (need AGNES_API_KEY) + 6 PASSIVE (no LLM).")
    import asyncio
    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
