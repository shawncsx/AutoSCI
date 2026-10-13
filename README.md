# AutoSCI — Distributed Research-Agent Pipeline

> A reverse-engineered clone of OpenAI's "math manuscript factory" — scaled down to a single
> Windows machine, retargeted at AI + bioinformatics (microbiome representation learning).

`autosci` is a small Python package that, given a local paper corpus and a list of seed ideas,
runs **8 specialised sub-agents in sequence** for each idea and produces a master's-level research
draft (IMRaD) plus a code skeleton and an internal audit. The design is a deliberate
recreation of OpenAI's math workflow that produced 719 manuscripts from ~4,000 candidate
problems: **Director → Research Team → keep / retry / drop**, with each sub-agent's task kept
narrow enough to be reliable.

In **autopilot mode** the agent is fully self-fueling: when the seed backlog runs out,
`IdeaMiner` reads your local corpus and proposes new ideas; `MetaPromptBuilder` lets any
agent rewrite its own prompt on the fly; `GitHubIngestor` can pull in third-party repo
READMEs as additional grounding; **arXivIngester** can pull fresh abstracts from arXiv;
**Idea templates** let you pre-shape the kinds of ideas that get proposed. You launch it
and walk away.

---

## ✨ Highlights

- **Three-layer architecture** — Knowledge Base / Director / Research Team.
- **10 specialised sub-agents** (v0.4: added `CitationGrounder` and `SectionValidator`):
  `IdeaCloser → LitSurveyer → MethodDesigner → ExpDesigner → PaperWriter → CitationGrounder → SectionValidator → CodeSketcher → Reviewer → Refuter`.
- **IdeaMiner** generates *new* seed ideas by reading your corpus — the autopilot no longer
  needs a hand-curated backlog to keep running.
- **Idea templates** (YAML) let you constrain the shape of newly proposed ideas.
- **arXivIngester** can pull fresh abstracts from arXiv in real time (3s politeness).
- **BM25 + LLM reranking** for semantic literature retrieval (no FAISS, no Torch).
- **MetaPromptBuilder** — agents can rewrite their own user-prompt if their first attempt
  under-performs (six built-in strategies, easy to extend).
- **GitHubIngestor** — optional pull of repo READMEs (you provide the list — no guessing).
- **OpenAI-compatible HTTP client** — works against any `/v1/chat/completions` endpoint,
  including the configured Agnes endpoint. Per-agent timeout override (paper_writer 180s,
  code_sketcher 120s, …) prevents the long-output agents from being cut off.
- **Token accounting** — every call counts; Director writes per-idea + grand-total cost
  to the summary and to each `recipe.json`.
- **Self-audit loop**: every draft is graded by a Reviewer, a Refuter, a SectionValidator
  and a CitationGrounder. The Director decides `keep / revise / refute` from all four signals.
- **Cross-idea memory** — refuted topics are remembered; next cycle avoids them.
- **Recipe + audit log**: every vault entry has a `recipe.json` (model / KB / token cost)
  and a `runs/director.log` is appended for every decision.
- **MIT licensed**, ~2,500 LOC of clean Python, runs in <2 GB RAM on CPU.

---

## 🛞 Autopilot mode

```powershell
# Default: 5 seed ideas, then IdeaMiner takes over, indefinitely:
python run_autopilot.py

# Limited burn:
python run_autopilot.py --max-cycles 2 --max-ideas-per-cycle 3

# Boring one-shot demo (the v0.1 way):
python run_demo.py
```

`Autopilot` keeps a `runs/autopilot_state.json` checkpoint so a Ctrl-C / power outage
will not lose results from earlier cycles.

---

## 📦 Repo layout

```
.
├── autosci/
│   ├── llm.py                       # OpenAI-compatible client (httpx), retries, JSON-mode
│   ├── schemas.py                   # Pydantic models for ALL inter-agent state
│   ├── prompts.py                   # Central prompt library (8 agents)
│   ├── kb.py                        # Digest ingest + BM25 + LLM rerank
│   ├── github_ingestor.py           # Pull READMEs of curated repos into the KB
│   ├── arxiv_ingester.py            # arXiv public Atom API + digest writer
│   ├── idea_template.py             # YAML-driven idea templates
│   ├── memory.py                    # Director's cross-idea memory
│   ├── meta_prompter.py             # Agents rewrite their own prompts
│   ├── autopilot.py                 # Continuous, self-fueling loop
│   ├── agents/
│   │   ├── _base.py                 # AgentRunner base
│   │   ├── idea_miner.py            # Self-discovers NEW seed ideas from KB
│   │   ├── idea_closer.py
│   │   ├── lit_surveyer.py
│   │   ├── method_designer.py
│   │   ├── exp_designer.py
│   │   ├── paper_writer.py
│   │   ├── citation_grounder.py     # v0.4 new — verifies [cite_key] in sections
│   │   ├── section_validator.py     # v0.4 new — independent LLM rates each IMRaD section
│   │   ├── code_sketcher.py
│   │   ├── reviewer.py
│   │   └── refuter.py
│   ├── pipeline.py                  # ResearchTeam orchestrator (10 steps)
│   ├── director.py                  # Director loop: pick → run → keep/drop
│   └── vault.py                     # Archive + manifest
├── data/
│   └── seed_ideas.json              # 5 default seed ideas
├── tests/                           # 15 tests, all green
├── docs/superpowers/{specs,plans}/  # design + implementation plan
├── run_demo.py                      # v0.1 one-shot entry
├── run_autopilot.py                 # v0.2 zero-touch entry
├── pyproject.toml
├── .env.example
└── LICENSE                          # MIT
```

---

## 🚀 Quick start

### 1. Install

```powershell
git clone <this repo>            # not required for local use
cd AutoSCI
pip install -e .                 # or: pip install -r requirements if you prefer
```

> Tested on Python 3.11+ / Windows PowerShell 5.1. No GPU, no PyTorch, no internet required
> beyond the LLM endpoint.

### 2. Configure

```powershell
copy .env.example .env
# Edit AGNES_* if you point elsewhere. Defaults:
#   AGNES_BASE_URL=https://api.agnes-ai.cn/v1
#   AGNES_MODEL=agnes-3.0-flash
```

### 3. Place a paper corpus

The default driver expects a folder of `*.digest.md` files (already a standard in the
microbiome literature). Point `AUTOSCI_PAPER_ROOT` to yours:

```ini
AUTOSCI_PAPER_ROOT=D:\shawn\paper
AUTOSCI_DIGEST_SUBDIR=_digest
```

### 4. Run

```powershell
# 5 seed ideas (≈ 12-15 minutes):
python run_demo.py

# Limit + custom output:
python run_demo.py --max-ideas 2 --runs-dir ./my_runs --vault-dir ./my_vault
```

Outputs:

- `runs/<idea_id>/paper.md` — IMRaD draft
- `runs/<idea_id>/state.json` — full structured state
- `runs/<idea_id>/code/` — Python skeleton
- `runs/director_summary.json` — kept/revised/refuted counts

---

## 🧪 Tests

```powershell
$env:PYTHONPATH='.'
python -m pytest tests/test_smoke_import.py tests/test_kb.py -v      # offline
$env:AUTOSCI_RUN_NET_TESTS='1'
python -m pytest tests/test_llm.py -v                                # uses live Agnes
```

---

## 🧠 How it maps to OpenAI's math pipeline

| OpenAI math                                       | AutoSCI                                          |
|---------------------------------------------------|--------------------------------------------------|
| Internal model + 3h "thinking compute" per problem | `LLMClient.chat_json` + 8-step agent chain       |
| ~4,000 problems → 372 result families → 719 papers| N seed ideas → kept / revised / refuted          |
| Hidden semantic KB                                | Local BM25 + LLM rerank over 100 digests         |
| Curators that keep adding / pruning ideas         | `IdeaMiner` (auto-discovered seed ideas)         |
| Withdraw + patch + re-publish                     | Director drops to `vault/refuted/` or marks revise|
| Hidden prompt-tuning & orchestration              | `MetaPromptBuilder` rewrites its own prompts     |
| Anthropic-style orthogonal RLHF / scaffold trials | `Strategy` keys in `STRATEGIES` registry         |
| Lean / formal audit                               | Reviewer + Refuter + line-items + flaw reports   |
| Unknown prompt templates                          | Centralised in `autosci/prompts.py`              |

> 🔎 See `docs/superpowers/specs/2026-10-12-autosci-design.md` for the full hypothesis.

---

## ✏️ Customising the run

- **Add your own seed ideas**: edit `data/seed_ideas.json`, or point `--seeds` at your own file.
- **Let the agent discover ideas**: stop providing `--seeds` and let `IdeaMiner` work.
- **Add curated GitHub repos** for KB enrichment: edit `autosci/github_ingestor.py`
  `DEFAULT_REPOS`, or pass `autopilot.github_repos=[RepoSpec(...)]` from Python.
- **Have an agent rewrite its prompt**: call `meta_prompter.build_override(...)` from any
  agent's `run` when a step fails (six built-in strategies in `STRATEGIES`).
- **Swap the model**: change `AGNES_MODEL` in `.env`. The prompt-library is bilingual (Chinese
  system prompts work fine with `agnes-3.0-flash` and most Chinese-tuned models); if you switch
  to a Western model you'll likely want to translate `prompts.py` to English.
- **Tune retrieval**: change `kb_topk` (default 8) and `lit_pool` (default 30).
- **Adjust the verdict policy**: edit `_set_verdict` in `autosci/pipeline.py`.

---

## 🛣️ Roadmap

| Version | Aim                                                                    |
|---------|------------------------------------------------------------------------|
| 0.1     | end-to-end pipeline + 5 default seeds + Reviewer/Refuter audit ✅        |
| 0.2     | Autopilot + IdeaMiner + MetaPromptBuilder + GitHubIngestor ✅           |
| 0.3     | Token accounting + tuned verdict policy + Director memory + audit log + Recipe ✅ |
| 0.4     | CitationGrounder + SectionValidator + arXivIngester + Idea templates ✅ |
| 0.5     | Sentence-Transformers + FAISS (replace BM25), multi-domain YAML specs  |
| 1.0     | Production: hypothesis verifier (run small synthetic experiment to reject false positives) |

---

## 📜 License

MIT (see `LICENSE`).

---

## 🙏 Acknowledgements

- OpenAI's `github.com/openai/math` for showing that the bottleneck is not the model — it's the
  workflow around it.
- 华大 Junhua Li and 林宇翔 (Yi-Xiang Lin) for the supervisor's design intuition
  (graph > sequence-transformer, multi-source node features, sequence-similarity prior edges).
- The 100 papers in `D:\shawn\paper` that make all of this possible.
