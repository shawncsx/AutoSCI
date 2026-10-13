# Claude Desktop / Trae / Cursor / WorkBuddy — Passive AutoSCI Workflow

> Copy-paste this entire prompt into your AI IDE's chat to use AutoSCI's
> passive-mode MCP tools. The IDE's LLM does all the thinking; AutoSCI just
> provides data, templates, and state.

---

You have access to the AutoSCI MCP server with these tools:

**Active (require AGNES_API_KEY — AutoSCI's LLM does the work):**
- `autosci_search_kb(query, top_k)` — search local KB
- `autosci_propose_ideas(n)` — AutoSCI LLM proposes ideas
- `autosci_run_one(...)` — AutoSCI runs full pipeline

**Passive (no LLM needed — YOU do the work):**
- `autosci_idea_anchors(query, top_k)` — return top-k KB abstracts (no LLM)
- `autosci_idea_template` — return IdeaSeed JSON schema
- `autosci_pipeline_prompts` — return system+user prompts for all 10 pipeline agents
- `autosci_section_template` — return IMRaD section template
- `autosci_review_template` — return Reviewer/Refuter scoring template
- `autosci_save_idea_draft(...)` — save a proposed idea to vault
- `autosci_save_paper_section(...)` — append a section to a paper.md

---

## Suggested workflow

When the user asks for "research on X" or "run AutoSCI on Y":

### Step 1 — Ground in the literature
Call `autosci_idea_anchors(query="<user's topic>", top_k=10)`.
Read the abstracts. Use them as factual anchors for everything below.

### Step 2 — Propose 3-5 ideas
Call `autosci_idea_template` to get the schema. For each idea:
- `id` — short kebab-case, unique
- `title` — single-line paper title
- `one_liner` — ≤30 char motivation
- `angle` — 2-3 sentence novelty claim
- `modalities` — from the enum
- `paper_corpus_hint` — search keywords for downstream

Save each via `autosci_save_idea_draft(...)`.

### Step 3 — For the best idea, run the pipeline
Call `autosci_pipeline_prompts` to get 10 (system, user_template) pairs.
For each step (in order: idea_closer → lit_surveyer → … → refuter):

1. Fill the user_template with current state (call previous tools as needed).
2. Call your own chat-completion API (or your own LLM) to get a JSON response.
3. Validate the response against the step's Pydantic schema (e.g. IdeaRefined).
4. Save the result for the next step.

If you don't have an LLM handy, stop and tell the user "I need an LLM endpoint
to run pipeline steps 1-10. Please configure one in your IDE."

### Step 4 — Write the paper section by section
Call `autosci_section_template` to get the IMRaD section order.
For each section:

1. Write 250-800 words in academic English.
2. Cite `[cite_key]` for any factual claim.
3. Call `autosci_save_paper_section(idea_id=<id>, section=<name>, content=<text>)`.

### Step 5 — Review and refuter
Call `autosci_review_template` to get the scoring rubric.
For each criterion, output line-items. Save as a section
`## Internal Audit` via `autosci_save_paper_section`.

### Step 6 — Verdict
Decide `kept` / `revised` / `refuted` based on blocking items.
If you decide `kept` or `revised`, the paper ends up in `vault/kept/<id>/`.
If `refuted`, it goes to `vault/refuted/<id>/`.
Use `autosci_save_paper_section(section="Internal Audit", verdict=<choice>, ...)`.

---

## Quick start commands you can use

| What you want | Tools to chain |
|---|---|
| "Find 3 ideas on X" | `autosci_idea_anchors` → `autosci_idea_template` → 3× `autosci_save_idea_draft` |
| "Draft paper for idea Y" | `autosci_pipeline_prompts` → 10× (you + your LLM) → `autosci_section_template` → 8× `autosci_save_paper_section` |
| "Review paper Y" | `autosci_read_paper` → `autosci_review_template` → 1× `autosci_save_paper_section(section="Internal Audit")` |
| "List previous runs" | `autosci_list_vault` |

---

## Why this is better than AutoSCI's own LLM

- You (the IDE's LLM) already know the user's full conversation context.
- You can be creative with the user_template — AutoSCI just provides the skeleton.
- You have access to the IDE's tools (file editing, terminal, web search) for free.
- The user pays for one LLM call (yours), not 10 (AutoSCI's).

If you want to use AutoSCI's LLM directly (skip the IDE for some steps),
still call `autosci_propose_ideas(n)` and `autosci_run_one(...)`.
