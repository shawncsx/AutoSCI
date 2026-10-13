# Cursor / VS Code / Cline — Passive AutoSCI Workflow

> Same as `claude_desktop_research_workflow.md`, but tuned for Cursor's
> MCP tool use. Cursor exposes AutoSCI tools in the Agent panel; just refer
> to them by name in your chat.

---

## Tools available

`autosci_idea_anchors`, `autosci_idea_template`, `autosci_pipeline_prompts`,
`autosci_section_template`, `autosci_review_template`,
`autosci_save_idea_draft`, `autosci_save_paper_section` (and the active
ones: `autosci_search_kb`, `autosci_propose_ideas`, `autosci_run_one`).

## In chat, ask for any of:

- "Use AutoSCI to find 3 ideas on microbiome foundation models"
- "Draft a paper for idea `perm_invariant_gat_film`"
- "Review the paper at `vault/kept/01_graph_vs_rank_transformer/paper.md`"
- "List all refuted runs from this week"

Cursor's LLM (default: gpt-4o) will use the AutoSCI MCP tools and your
own chat to do the work. You can override the model with the small
"AI" dropdown in the composer.

For best results:
1. Set the project root to `D:\shawn\git\AutoSCI` so paths work.
2. Add `AGNES_API_KEY` to your shell environment if you want
   `autosci_run_one` to also work (active mode).
3. Otherwise, just stay in passive mode and let Cursor do the LLM work.
