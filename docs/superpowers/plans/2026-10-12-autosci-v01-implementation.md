# AutoSCI v0.1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) 或 superpowers:executing-plans 来按 task 执行。本计划的 step 用 `- [ ]` checkbox 形式标记进度。

**Goal:** 实现 AutoSCI —— 一个分布式研究 agent 流水线，在用户本地 `D:\shawn\paper` 的 100 篇微生物组表示学习论文基础上，自动产出"硕士级"研究 markdown 报告。

**Architecture:** 三层结构 — Knowledge Base (`bm25 + LLM rerank`, 可升级 sentence-transformers+faiss) / Director (调度官，循环跑 N 个 seed idea) / Research Team (8 子 agent 串行：IdeaCloser→LitSurveyer→MethodDesigner→ExpDesigner→PaperWriter→CodeSketcher→Reviewer→Refuter)。所有 step 落盘 `runs/<project>/`，状态可恢复。

**Tech Stack:** Python 3.11+；`httpx`（OpenAI 兼容 → agnes-3.0-flash）；`pydantic`（schema）；`rank-bm25`（检索）；`pytest`（测试）；纯文件系统 + sqlite-mock 索引（用 `jsonl`）。

## 全局约束（来自 spec）

- 工作目录：`D:\shawn\git\AutoSCI`
- API：`https://api.agnes-ai.cn/v1` + model `agnes-3.0-flash`，key 通过 `.env` 提供
- 目标 KB：`D:\shawn\paper\_digest\` 全部 digest（100 篇）
- 每个产物：IMRaD + ≥3 baseline + ≥1 ablation + ≥8 篇本地引用 + 伪实验代码
- 输出语言：中文（prompt）+ 论文 markdown 中英文混合（与现有 digest 风格一致）
- 不引入：LangChain / AutoGen / torch / GPU 真训练
- 全程不联网写论文（除 API 调用），不下载新论文
- License：MIT

## 文件结构

```
D:\shawn\git\AutoSCI\
├── pyproject.toml              # 依赖声明
├── .env.example                # API key 模板
├── .gitignore
├── README.md                   # 跑法 + 设计稿索引
├── LICENSE                     # MIT
├── run_demo.py                 # 顶层 demo 入口
├── data\
│   └── seed_ideas.json         # 5 个 seed ideas，覆盖微生物组 5 个子方向
├── autosci\
│   ├── __init__.py
│   ├── llm.py                  # Agnes/OpenAI 兼容客户端 + retries
│   ├── schemas.py              # Pydantic: IdeaState, StepIO, Paper, Critique...
│   ├── prompts.py              # 8 个 agent 的 system + user prompt 模板
│   ├── kb.py                   # ingest + bm25 + LLM reranker
│   ├── agents\
│   │   ├── __init__.py
│   │   ├── _base.py            # AgentRunner 抽象（含 system_prompt、JSON 模式、温度）
│   │   ├── idea_closer.py
│   │   ├── lit_surveyer.py
│   │   ├── method_designer.py
│   │   ├── exp_designer.py
│   │   ├── paper_writer.py
│   │   ├── code_sketcher.py
│   │   ├── reviewer.py
│   │   └── refuter.py
│   ├── pipeline.py             # ResearchTeam：8 步串行编排
│   ├── director.py             # 调度官 + 撤回/修补循环
│   └── vault.py                # 归档 + 索引
└── tests\
    ├── test_llm.py
    ├── test_kb.py
    ├── test_agents.py
    └── test_pipeline_smoke.py
```

## 接口契约（全局）

```python
# autosci/schemas.py
class IdeaSeed(BaseModel):
    id: str
    title: str
    one_liner: str
    angle: str            # 创新点定位
    modalities: list[str] # graph / ssl / compositional / threshold_sensitivity / cross_cohort

class IdeaRefined(BaseModel):    # IdeaCloser 输出
    claim: str
    hypothesis: list[str]        # ≥3 可证伪假设

class LitSurveyItem(BaseModel):
    cite_key: str               # e.g. "[MGM2-2026]"
    title: str
    year: int
    relevance: str              # 一句话
    takeaway: str               # 一句话"我们能/不能/借鉴/超越…"

class MethodSpec(BaseModel):     # MethodDesigner 输出
    architecture: str            # 一段系统结构描述
    algorithm: str               # 算法伪代码（文字描述即可）
    novelty_bullets: list[str]

class ExpSpec(BaseModel):        # ExpDesigner 输出
    datasets: list[dict]
    baselines: list[dict]
    ablations: list[dict]
    metrics: list[str]
    significance_test: str
    expected_outcome: str

class PaperDraft(BaseModel):     # PaperWriter 输出
    title: str
    abstract: str
    sections: dict[str, str]     # {"introduction": "...", "related_work": "...", ...}
    references: list[str]        # cite keys list

class CodeSketch(BaseModel):     # CodeSketcher 输出
    files: list[dict]            # [{path: "experiments/main.py", content: "..."}]
    run_instructions: str

class Critique(BaseModel):       # Reviewer 输出（line-item）
    summary: str
    line_items: list[dict]       # [{section, anchor, severity, suggestion}]
    severity_counts: dict[str, int]

class RefuteResult(BaseModel):   # Refuter 输出
    verdict: Literal["keep", "revise", "refute"]
    fatal_flaws: list[dict]      # [{flaw, evidence, fix}]
    revise_suggestions: list[str]

class IdeaState(BaseModel):      # 一个 idea 完整状态
    idea_seed: IdeaSeed
    refined: IdeaRefined | None = None
    lit_survey: list[LitSurveyItem] | None = None
    method_spec: MethodSpec | None = None
    exp_spec: ExpSpec | None = None
    paper_draft: PaperDraft | None = None
    code_sketch: CodeSketch | None = None
    review: Critique | None = None
    refute: RefuteResult | None = None
    verdict: Literal["kept", "revised", "refuted", "pending"] = "pending"
    paper_path: str | None = None
    history_log: list[str]
```

---

## Task 1: 仓库脚手架 + 依赖 + License

**Files:** 
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`

**Interfaces:** 
- 后续所有 Task 通过 `from autosci.llm import LLMClient` 拿到 model，依赖 `.env` 提供 `AGNES_BASE_URL / AGNES_API_KEY / AGNES_MODEL`。

- [ ] **Step 1:** 写 `pyproject.toml`，声明依赖 `httpx>=0.27, pydantic>=2.5, rank-bm25>=0.2.2, pytest>=7.4, python-dotenv>=1.0, jinja2>=3.1`。

- [ ] **Step 2:** 写 `.env.example` 包含 `AGNES_BASE_URL=https://api.agnes-ai.cn/v1`, `AGNES_API_KEY=sk-LYFGTYGx4AF0JjyWs1ukBomO8iSQKzwz1SbUkvwm9SheppRu`, `AGNES_MODEL=agnes-3.0-flash`, `AUTOSCI_MAX_RETRIES=3`, `AUTOSCI_LLM_TIMEOUT_S=60`。

- [ ] **Step 3:** 写 `.gitignore`（`__pycache__`, `.env`, `runs/`, `*.pyc`, `.pytest_cache`, `dist/`）。

- [ ] **Step 4:** 写 `LICENSE` 文本（MIT，作者 2026）。

- [ ] **Step 5:** Commit: `git init && git add -A && git commit -m "chore: scaffold + license (MIT)"`。

## Task 2: schemas.py（全部 Pydantic 模型）

**Files:** Create `autosci/schemas.py`，定义上面列出的全部 BaseModel。

- [ ] **Step 1:** 实现所有 class，含枚举字段与默认值。
- [ ] **Step 2:** 加 `model_dump_json(ensure_ascii=False, indent=2)` 帮助方法。
- [ ] **Step 3:** 简易自测（直接 `python -c "from autosci.schemas import *"` 不报错即可），commit。

## Task 3: llm.py（Agnes/OpenAI 兼容客户端）

**Files:** Create `autosci/llm.py`，写 LLMClient，构造 `chat(messages, *, temperature, json_mode, response_format) -> dict`。

**Interfaces:**
- 消费：`AGNES_*` 环境变量，`temperature` / `json_mode`。
- 产出：`{"content": str, "tokens": int, "raw": ...}`。

- [ ] **Step 1:** 失败测试 `tests/test_llm.py::test_chat_smoke`（用一个最小 messages，期望含 "content" key + tokens > 0）。
- [ ] **Step 2:** 实现 `chat`，含 retries（指数回退）+ timeout + json_mode 强制 `response_format={"type": "json_object"}`。
- [ ] **Step 3:** 让测试过，commit。

## Task 4: prompts.py（所有 agent prompt）

**Files:** Create `autosci/prompts.py`，函数 `system_<agent>() -> str` + `user_<agent>(state_dict) -> list[dict]`。

- [ ] **Step 1:** 为 8 个 agent 各写中文 system prompt（含"角色 / 必输出字段 / 反例"）。
- [ ] **Step 2:** 为每个 agent 写 user prompt，包含 {state_dict} 占位符（IdeaState 的子集）。
- [ ] **Step 3:** 用 jinja2 模板化 user prompt；commit。

## Task 5: kb.py（ingest + bm25 + LLM rerank）

**Files:** Create `autosci/kb.py` + `autosci/agents/_ingest.py`（CLI ingest）。

- [ ] **Step 1:** 写 `_ingest.py`，扫 `D:\shawn\paper\_digest\*.digest.md` → 解析 front-matter / 标题 / 摘要段落 → 输出 `data/kb_corpus.jsonl`（每行 `{id, title, year, relevance, fulltext}`）。
- [ ] **Step 2:** 失败测试 `tests/test_kb.py::test_search_returns_topk`：先 ingest → 调用 `kb.search(query, k=8)`，断言返回 8 篇且含关键词命中。
- [ ] **Step 3:** 实现 `KB` 类：build 内存 bm25 index；`search(query, k=8)` 默认走 LLM rerank（调用 LLMClient chat，给 top-30 候选，让它重排序选 8 篇）。
- [ ] **Step 4:** 测试通过，commit。

## Task 6: agents/_base.py + 8 个 agent

**Files:** Create `autosci/agents/_base.py` + 8 个子 agent；每个 agent 都是 `class XxxAgent:` 内含 `run(state: IdeaState) -> IdeaState`。

- [ ] **Step 1:** 写 `_base.AgentRunner`，负责：取 system_prompt + user_prompt + LLM 调用 + 解析为对应 schema（Pydantic）+ 写错误时重做一次。
- [ ] **Step 2:** 写 8 个 agent，分别 `run`（IdeaCloser→IdeaRefined 等）。每个 agent 先做最小实现 + 单测 `tests/test_agents.py::test_<agent_name>_parses_valid_state`，验证它能把假 state 转出对应 schema。
- [ ] **Step 3:** 8 个都过了，commit。

## Task 7: pipeline.py（ResearchTeam 8 步编排）

**Files:** Create `autosci/pipeline.py`。

- [ ] **Step 1:** 失败测试 `tests/test_pipeline_smoke.py::test_research_team_writes_files`：给一个 IdeaSeed，跑 ResearchTeam.run，断言 `runs/<id>/paper.md` 存在 + `state.json` 全字段非空。
- [ ] **Step 2:** 实现 `class ResearchTeam`：顺序调用 8 个 agent，每步把 IdeaState 序列化到 `runs/<id>/steps/<seq>_<agent>.json` + 渲染到 markdown。
- [ ] **Step 3:** 测试通过，commit。

## Task 8: data/seed_ideas.json（5 个 idea）

**Files:** Create `data/seed_ideas.json`。

- [ ] **Step 1:** 写 5 个 seed idea：
  1. 图 Transformer 替代排序-位置编码的微生物组表示学习（导师核心设想）
  2. OTU 聚类阈值 97% vs 99% 的稳定性实证（导师空白点）
  3. 序列相似度先验 + 丰度直接融合的图 SSL 预训练（多源融合）
  4. 图基础模型跨队列 CRC 鲁棒性（跨 cohort / distribution shift）
  5. compositional 约束下的图模型 + 零膨胀处理（compositional data）
- 每条包含 id/title/one_liner/angle/modalities。

## Task 9: director.py（调度官 + 撤回/修补循环）

**Files:** Create `autosci/director.py`。

- [ ] **Step 1:** 失败测试 `tests/test_director.py::test_director_runs_seed_ideas_and_keeps_or_drops`：load seed_ideas.json，跑 Director.run_all，断言 vault 里有 5 份产物，其中 ≥1 kept。
- [ ] **Step 2:** 实现 `Director`：循环读 backlog → 起 ResearchTeam → 写 vault；记录 keep/revise/refute；日志 `runs/director.log`。
- [ ] **Step 3:** 测试通过。

## Task 10: vault.py

**Files:** Create `autosci/vault.py`。

- [ ] **Step 1:** 实现 `class Vault`：维护 `vault/kept/`, `vault/refuted/`, `vault/manifest.json`；提供 `record(state, paper_md, code_paths)` 接口。
- [ ] **Step 2:** Director 调它。

## Task 11: run_demo.py 顶层入口 + README

**Files:** Create `run_demo.py` + 完善 `README.md`。

- [ ] **Step 1:** 写 `run_demo.py`：`from autosci.director import Director; Director(seed_ideas_path=...).run_all()`，加 `--max-ideas` / `--output-dir` 参数。
- [ ] **Step 2:** 写 README：包含 `如何跑 / 设计稿 / 示例输出 / 怎么改 prompt / License`。
- [ ] **Step 3:** 跑一次 demo（5 idea，每份使用合理 max_tokens，估 30–120 分钟），把产物写进 `vault/`。
- [ ] **Step 4:** Commit final + 给用户回归报告。

---

## 自审（spec vs plan 覆盖）

- spec §1 (OpenAI 流程推测) — 不需要单独 task，由 Director + Pipeline 实现。✓
- spec §2.1 目标 — Task 9 + Task 11 实现（Director 跑 5 idea）。✓
- spec §2.3 硕士级 6 条 — Reviewer / Refuter + PaperWriter 校验。✓
- spec §3 架构三层 — Task 5/6/9。✓
- spec §4 技术栈 — Task 1 显式约定。✓
- spec §5 仓库结构 — 全部对齐。✓
- spec §6 实施步骤 — Task 1–11 与之 1:1 对应。✓
- spec §7 风险与缓解 — retries (Task 3) / KB chunk (Task 5) / novelty 检查（Director 内） / 数据不接真（Task 11 默认） / API retry（Task 3）。✓
- spec §8 验收 — Task 11 完成时验证。✓
