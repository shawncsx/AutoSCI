# AutoSCI 设计稿（v0.1）

日期：2026-10-12
作者：Mavis
目的：复现 OpenAI 数学项目的"分布式研究 agent 生产线"，并改造用于"AI + 生信（微生物组表示学习）"领域，目标是 **批量产出硕士级研究成果**（不是证明开放数学猜想）。

---

## 1. OpenAI 数学项目的流程推测

### 1.1 公开材料里能"看到"的部分
- 719 篇手稿 / 372 个研究家族 / ~4000 个候选问题
- 每个成果平均 3 小时 ChatGPT Pro 算力
- "绝大多数都用同一套流程"，只在少数（zeta 零点无零区、CM Hodge 猜想）改动
- ~42% 顶级结果做了 Lean 形式化
- "withdrawn" 文件多次出现，说明会大批量失败
- `history.md` 显式记录撤回、修订 → **有个中央 orchestrator 维护全局状态**

### 1.2 我们对它的猜测（AutoSCI 假设）
按你的直觉 → 我把它具体化：

| 层 | OpenAI 数学项目里… | AutoSCI 里的对应物 |
|---|---|---|
| 战略层 | 一个主任研究员/小团队给模型"喂问题" | `Director`（调度官） |
| 战术层 | 模型独立做 3h 研究出论文 | `Research Team`（一个子 agent 集群，串行/并行） |
| 基础层 | 一致的问题 interface、写作模板、评判规则 | AutoSCI Pipeline（标准 6 步） + Prompt 模板 |
| 知识层 | 应该是大量 Lean/Mathlib 知识 + 私藏语义检索 | `Knowledge Base`（paper 库 + FAISS/chroma 检索） |
| 评测层 | Lean 形式化、人工精校、撤销/修订机制 | `Reviewer`（自动审稿）+ `Refuter`（寻 bug）+ 版本化 |

推论的关键一句话：**OpenAI 并不是"一个超能力 agent 直接证明 BSD 猜想"。OpenAI 是"一个调度官，把 4000 个题目扔给 worker，worker 跑 3h 写论文，调度官审稿、决定哪些保留、哪些撤回"。**

这正好匹配你"拆分到每个子 agent 的活比较简单"的洞察。

---

## 2. AutoSCI 的设计目标

### 2.1 目标
**用一套"调度官 + 子 agent 集群"，对接用户的 `D:\shawn\paper` 库，自动产出"硕士级别"微生物组表示学习的研究 markdown 报告。**

### 2.2 非目标
- 不证明开放数学猜想
- 不预训练大规模模型（在 v0.1 阶段跳过真实训练，只产出方法学 + 实验代码骨架）
- 不替代人类同行评议
- 不直接发顶刊
- 不做 Lean/Coq 形式化

### 2.3 "硕士级别"的可验收条件
每份报告必须满足：
1. 提出 1 个具体可证伪的科学假说（"X 方法比 Y 在 Z 数据集上 AUC 提升 5pp"）。
2. 引用至少 8 篇本地论文（从 KB 中检索出来，并写入 references）。
3. 实验设计完整：≥3 baseline、≥1 ablation、明确数据切分（含 leave-one-cohort-out）、指标 + 显著性检验。
4. 给出"伪实验代码骨架"（不一定要真跑通，但能放在仓库里）。
5. 报告结构符合 IMRaD（introduction / related work / method / experiments / conclusion），长度 10–20 页 markdown。
6. **带 self-review 与可被撤回**：Director 把不可挽救的产出标为 "refuted" 并归档。

---

## 3. 架构

```
                       ┌──────────────────────────────────────┐
                       │           Knowledge Base              │
                       │ ────────────────────────────────     │
                       │ • paper corpus  (D:/shawn/paper)    │
                       │ • digests (_digest/)                 │
                       │ • FAISS index (chroma 或简易版)       │
                       │ • glossary / 三导师质疑 / 数据登记     │
                       └──────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Director （主控）                          │
│ ──────────────────────────────────────────────────────────────────  │
│ • 维护 idea backlog（pending / active / done / refuted）              │
│ • 选择下一个 idea （贪心 + 去重 + novelty）                           │
│ • 调度下面的 Research Team                                              │
│ • 接收 Reviewer / Refuter 报告，决定 keeper / retry / drop           │
└─────────────────────────────────────────────────────────────────────┘
                           │                ▲                ▲
                           ▼                │                │
                  ┌─────────────────┐        │                │
                  │ Research Team    │────────┘                │
                  │ (per idea)       │                         │
                  │                  │                         │
                  │ 1. IdeaCloser    │                         │
                  │ 2. LitSurveyer   │                         │
                  │ 3. MethodDesigner│                         │
                  │ 4. ExpDesigner   │                         │
                  │ 5. PaperWriter   │                         │
                  │ 6. CodeSketcher  │                         │
                  │ 7. Reviewer      │────────────────────────-│
                  │ 8. Refuter       │ (回传到 Director)        │
                  └─────────────────┘                         │
                                                               │
                              Claude / Agnes OpenAI-compatible API
                              (model: agnes-3.0-flash)
```

### 3.1 Agent 角色
| 角色 | 输入 | 输出 | 是否需要工具 |
|---|---|---|---|
| IdeaCloser | 一个候选 idea | 一句话主诉 + 5 个可证伪假设 | LLM |
| LitSurveyer | 主诉 + KB | related-work 段落 + 引用列表 | LLM + 检索 |
| MethodDesigner | 主诉 + KB | 系统图 + 算法步骤 + 1 段创新描述 | LLM |
| ExpDesigner | 方法 + KB | 数据切分 / baseline / 指标 / 显著性 / 预期结果 | LLM |
| PaperWriter | 上面所有 | 完整 IMRaD markdown 草稿 | LLM |
| CodeSketcher | 方法 + 假数据 | 伪实验代码骨架（python/numpy） | LLM + 文件写入 |
| Reviewer | 草稿 v1 | 5–10 条 line-item 改进意见 | LLM |
| Refuter | 草稿 v1 | 找漏洞（统计、构建、外部文献一致性）→"撤回 / 修订"建议 | LLM |

子 agent 全部基于同一种接口：`def run(state: State) -> State`。
状态 (state) 是一个 dataclass，串行在磁盘上 (`runs/<project>/state.json`)，每个 step 写一个 markdown 到 `runs/<project>/steps/<step_name>.md`。

### 3.2 Director 的算法
简化版：
```
while budget_left and pending_ideas_nonempty:
    idea = pick_next(backlog)        # novelty + 上次失败原因避免
    if idea in_done_with_same_conclusion: continue
    team = spawn_research_team(idea)
    draft = team.run_full_pipeline()
    score = review(draft)            # Reviewer 打分
    refuted = refuter(draft)         # Refuter 找漏洞
    decision = director_decide(score, refuted)
    record(idea, draft, decision)    # keep / retry / drop
    write_to_vault(idea, draft)
```

### 3.3 Knowledge Base
- 把 `D:\shawn\paper` 的 PDF 都解析（已经有的 `_digest` 优先，没有再读 PDF），产出统一 schema：`title / year / method / data / limitations / relevance_tags / abstract_short`。
- 对 `_digest` 做轻量向量化（用 sentence-transformers 或者直接调用 LLM 出嵌入；v0.1 可以用最朴素的 bm25 + 关键词）。
- 存到 `kb.sqlite`（一套轻量表）或 `kb.jsonl` + `kb.index.faiss`。
- 暴露工具：`kb.search(query, k=8)`  → 给 LLM 当 context。

### 3.4 关键子模块
- `Pipeline`：主编排（Director 内部循环 + ResearchTeam 内部 8 步）。
- `Agents`：8 个子 agent，每个都是 prompt + 一个 system message + 一些约束。
- `KB`：论文库 + 检索器。
- `LLMClient`：agnes-3.0-flash 的薄封装（OpenAI 兼容协议），含 retries + 速率限制 + token 计数。
- `Prompts`：模板化所有的 prompt（系统消息、用户消息、JSON schema 约束）。
- `Reports`：统一输出 schema。

---

## 4. 技术栈

- **语言**：Python 3.11+（不需要 C/C++；不需要 GPU）。
- **依赖**：
  - `httpx` 或 `openai` 客户端（兼容 OpenAI API 协议 → 直接打到 `https://api.agnes-ai.cn/v1`）
  - `pydantic`（state / output schema）
  - `pypdf` 或 `pdfplumber` 解析 PDF 备用（v0.1 可不解析，先用已有 `_digest`）
  - `rank-bm25` 做轻量检索；可选 `sentence-transformers` / `faiss` 升级
  - `pytest` 跑离线测试
  - 写文档：`jinja2`、`markdown-it-py`
- **可移植**：纯 Python，运行只需 pip install + 配置 .env（含 API key），无 GPU、无云。
- **存储**：纯文件系统 + sqlite；一切可 git 化。

### 不用的东西
- 不引入 LangChain / AutoGen / LangGraph 等巨型框架（v0.1 自己手写一个 `Pipeline` 就够，避免被依赖绑架）。
- 不引入向量数据库服务器（v0.1 内存 + 文件）。
- 不引入 PyTorch / sklearn（v0.1 不跑真训练）。

---

## 5. 仓库结构

```
D:\shawn\git\AutoSCI\
├── README.md                # 项目说明 + 怎么跑
├── pyproject.toml           # 依赖
├── .env.example             # API_KEY / BASE_URL / MODEL 模板
├── docs\
│   └── superpowers\         # 设计稿在这里
├── autosci\
│   ├── __init__.py
│   ├── llm.py               # Agnes/OpenAI 兼容客户端
│   ├── kb.py                # 知识库 + 检索器
│   ├── prompts.py           # 所有 prompt 模板
│   ├── schemas.py           # Pydantic schemas
│   ├── agents\
│   │   ├── __init__.py
│   │   ├── idea_closer.py
│   │   ├── lit_surveyer.py
│   │   ├── method_designer.py
│   │   ├── exp_designer.py
│   │   ├── paper_writer.py
│   │   ├── code_sketcher.py
│   │   ├── reviewer.py
│   │   └── refuter.py
│   ├── pipeline.py          # ResearchTeam 编排
│   ├── director.py          # 调度官
│   └── vault.py             # 归档 / 索引
├── data\
│   └── ideas\               # seed ideas
├── runs\                    # 运行结果（每个 idea 一个文件夹）
└── tests\
    ├── test_llm.py
    ├── test_kb.py
    └── test_pipeline_smoke.py
```

---

## 6. 实施步骤（roadmap）

### v0.1（这次要做完的）
1. **脚手架**：仓库结构 + 依赖 + README + .env。
2. **LLMClient**：agnes-3.0-flash 调用 + 重试 + token log。
3. **KB**：把 `D:\shawn\paper\_digest\*.digest.md` 全部入库；暴露 `kb.search(query, k=N)`。
4. **Prompts**：8 个 agent 的 prompt 模板（中文 prompt，便于复用）。
5. **Agents（全部 stub）**：8 个 agent，每个有"骨架 + 一段"提示词模板 + schema 验证 + 运行函数。
6. **Pipeline**：跑通 8 步的 ResearchTeam，跑完在一个 idea 上产出 `runs/<project>/paper.md`。
7. **Director**：跑一个最简单的循环：默认 3 个 seed idea，逐个跑，把结果写进 vault。
8. **测试**：LLM smoke + KB 检索 + pipeline 烟雾测试。
9. **README**：清晰写出怎么跑、怎么换 idea、怎么改 prompt。

### v0.2（后续）
- 加真实数据接入（用 metalog 的一个小子集）
- 加轻量级 self-supervised benchmark（如：在 metalog 上做 MLM/对比学习）
- 加真实训练 + 评测脚本（在不强 GPU 的情况下）

### 不在 v0.1/v0.2 做
- 部署为 SaaS
- 形式化证明 / Lean 集成

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| agnes-3.0-flash 不够强，写出来的论文太浅 | 提示词分层 + 注入相关 digest；评分门槛；如多次 retry 仍差 → drop |
| 检索 context 太大把 prompt 撑爆 | 严格 top-k + truncate；KB chunk 化 |
| 想法撞车 / 重复 | Director 用 novelty 检查 + 元数据登记 |
| 评审滥竽充数 | 强制 Reviewer 必须给出"具体到段落/章节"的 line-item，否则重做 |
| 用户没数据出大实验 | v0.1 明确不跑真实验，标注 "experiment outline only" |
| API 不稳定 / 中断 | LLMClient 重试 + 每步 checkpoint；任何 step 中断可断点恢复 |

---

## 8. 验收与验收方式

按 superpowers 习惯 + 你的"硕士级别"标准：

1. **跑一遍完整流水线**，输入默认 3 个 seed idea，产出 3 份 `paper.md`，每份都过 self-review。
2. 抽样 1 份，由你（用户）人工看是否像硕士开题。
3. `autosci --demo` 一条命令可重现结果。
4. 失败案例（refuted）也归档，作为后续迭代数据。

---

## 9. 开放问题（请你确认）

- A. **跑多少份报告当 demo？** 默认 3 份（cover 不同子方向：图模型、SSL、组合数据），时间约 30–90 分钟。
- B. **是否要接入真实数据？** v0.1 默认不接（用"实验方案 + 伪数据"），避免长时计算与超 API 限额。
- C. **关键 KB 索引是否要花时间做好？** v0.1 用 bm25 + 已有 digest。
- D. **许可证**：默认 MIT，便于后续公开。

如果 A/B/C/D 都默认，那就开干。
