# AutoSCI — 分布式研究 Agent 流水线 / Distributed Research-Agent Pipeline

> **复刻 OpenAI 的"数学论文工厂"流水线**，但跑在单机 Windows 上、面向 AI + 生信（微生物组表示学习）场景。
>
> *A reverse-engineered clone of OpenAI's "math manuscript factory" — scaled down to a single Windows machine, retargeted at AI + bioinformatics (microbiome representation learning).*

AutoSCI 是一个 ~2,500 行的 Python 包。给定本地论文语料 + 一批种子 idea，它会**对每个 idea 串行调度 10 个专业子 agent**，产出一份"硕士级别"研究草稿（IMRaD 完整结构）+ 伪代码骨架 + 内部审计报告。整个设计是 OpenAI 数学项目工作流的可执行复刻 —— OpenAI 用这套工作流从 ~4,000 个候选问题里产出 719 篇论文：**Director → Research Team → keep / retry / drop**，每个子 agent 的职责被切得足够窄以保证稳定。

**核心思想**：OpenAI 的瓶颈从来不在模型，而在模型**周围**的工作流。AutoSCI 试图把这个工作流用 ~2,500 行 Python 写出来，让任何研究组都能在自己笔记本上批量产硕士级论文。

---

## ✨ 三大亮点

### 1️⃣ 调度官 + 子 agent 集群（10 个）
- **Director（调度官）** — 选 idea、调 Research Team、决定 keep/revised/refute
- **Research Team（10 个子 agent 串行）**：
  `IdeaCloser → LitSurveyer → MethodDesigner → ExpDesigner → PaperWriter → CitationGrounder → SectionValidator → CodeSketcher → Reviewer → Refuter`
- **每步任务极窄**：IdeaCloser 只产 1 段 claim + 1 个假设；SectionValidator 只给 8 个 section 打 1-5 分

### 2️⃣ 5 源论文库（k 越高越丰富）
| 源 | 覆盖 | 默认 |
|---|---|---|
| **arXiv** | 物理 / CS / ML / quant bio 预印本 | 30+ query 并发 |
| **OpenAlex** | 跨学科 2.5 亿论文（最广覆盖）| 4 query |
| **bioRxiv** | 生物学预印本（microbiome 主战场）| 4 query |
| **PubMed/PMC** | 临床 + 基础生物 36M+（**含 Cell/Nature/Lancet**）| 4 query |
| **Semantic Scholar** | 1.95 亿论文 + 引文图（**含 ICML/NeurIPS/CVPR**）| 4 query |

跨源 **dedup** 走 3 级优先级：DOI > arXiv-id > 标题 hash

### 3️⃣ Claude-Code 风格终端 UI（Python `rich` 实现）
- ASCII art banner
- ● Tool name + ⎿ result 缩进（Claude Code 标志性样式）
- ⠋ dots spinner
- verdict 颜色表格（kept 绿 / revised 黄 / refuted 红）
- 实时 status bar
- REPL：`❯ /help /status /ideas /vault /run /mass /demo /exit`

---

## 🛞 两种运行模式

### A. 单次 demo（5 个 idea，约 12-15 分钟）
```powershell
python run_demo.py
```

### B. Autopilot 连续自循环（生产模式）
```powershell
# 默认 5 个 seed + 持续从 KB 挖新 idea
python run_autopilot.py

# 限制版本
python run_autopilot.py --max-cycles 2 --max-ideas-per-cycle 3

# Mass production（拉 5 源 + Wide IdeaMiner + parallel Director）
python run_mass.py                          # 默认 30 query × 50 = 1500 papers
python run_mass.py --per-query 30 --ideas 10  # 小规模
```

`Autopilot` 状态持久化到 `runs/autopilot_state.json`，Ctrl-C / 断电都不丢。

---

## 📦 仓库结构

```
.
├── autosci/                          # 核心 ~2,500 行 Python
│   ├── llm.py                        # OpenAI 兼容客户端（重试 / JSON-mode / token 累计）
│   ├── schemas.py                    # 全部 Pydantic 模型
│   ├── prompts.py                    # 10 agent 集中 prompt 库
│   ├── kb.py                         # Digest 摄入 + BM25 + LLM rerank
│   ├── rich_console.py               # Claude-Code 风格 rich 输出
│   ├── mass_ingestor.py              # 30+ query 并发拉 arXiv
│   ├── openalex.py / biorxiv.py /    # 4 源 client
│   │   pubmed.py / semanticscholar.py
│   ├── dedup.py                      # 跨源 DOI/arxiv/title-hash dedup
│   ├── idea_miner.py /               # Wide IdeaMiner + template
│   │   idea_template.py
│   ├── memory.py / distiller.py /    # Director 跨 idea 记忆 + 自动蒸馏
│   │   predictor.py
│   ├── verifier.py                    # 伪实验模拟器（numpy 跑 small experiment）
│   ├── html_report.py /               # HTML 报告 + 多 idea 编译
│   │   compilation.py
│   ├── cli_dashboard.py /             # 实时 dashboard + REPL
│   │   meta_prompter.py
│   ├── director.py / pipeline.py     # 调度官 + ResearchTeam 编排
│   ├── autopilot.py / vault.py       # 连续循环 + 归档
│   └── agents/                       # 10 个子 agent
├── data/
│   ├── seed_ideas.json               # 5 个默认 idea
│   ├── idea_templates.json           # 3 个 idea 模板
│   └── corpus_specs/                 # microbiome / clinical / materials
├── tests/                            # 128 测试函数，全过
├── docs/superpowers/{specs,plans}/   # 设计稿 + 实施计划
├── run_demo.py / run_autopilot.py /  # 3 个入口
│   run_mass.py / run_dashboard.py /
│   run_cli.py
├── pyproject.toml
├── LICENSE                           # MIT
└── README.md
```

---

## 🚀 5 分钟上手

### 1. 安装

```powershell
git clone https://github.com/shawncsx/AutoSCI.git
cd AutoSCI
pip install -e .
```

> 需要 Python 3.11+。无 GPU / 无 PyTorch / 除 LLM endpoint 外不需联网。

### 2. 配置

```powershell
copy .env.example .env
# 默认：
#   AGNES_BASE_URL=https://api.agnes-ai.cn/v1
#   AGNES_MODEL=agnes-3.0-flash
# 修改你的 API key
```

### 3. 准备语料

把 `*.digest.md` 文件放到一个目录里（论文摘录 + 元信息）。在 `.env` 里设：

```ini
AUTOSCI_PAPER_ROOT=D:\shawn\paper
AUTOSCI_DIGEST_SUBDIR=_digest
```

### 4. 跑

```powershell
# Demo（5 个 idea，约 12-15 分钟）
python run_demo.py

# 限数量
python run_demo.py --max-ideas 2 --runs-dir ./my_runs --vault-dir ./my_vault

# 一键 mass：5 源 + 20 idea + parallel=4
python run_mass.py
```

### 5. 跑 REPL（实时交互）

```powershell
python run_cli.py
```

```
❯ /help
❯ /status
❯ /ideas
❯ /vault
❯ /run --ideas 3
❯ /mass --per-query 30 --ideas 10
❯ /exit
```

---

## 🧪 测试

```powershell
$env:PYTHONPATH='.'
python -m pytest tests/ -q                              # 128 测试
$env:AUTOSCI_RUN_NET_TESTS='1'
python -m pytest tests/test_llm.py -v                 # 实测 Agnes 端点
```

**目前测试覆盖**：15 个测试文件，128 个测试函数。涵盖 schema round-trip、KB 检索、pipeline 烟雾、resilience（LLM 429/坏 JSON）、verdict policy、distillation、citation grounding、section validation、embedding 索引、corpus spec、dedup、verifier 真实数字、compilation、rich console、autopilot 等。

---

## 🧠 与 OpenAI 数学项目的对照表

| OpenAI 数学项目 | AutoSCI 复刻 |
|---|---|
| 内部模型 + 每个问题 3 小时推理 | `LLMClient.chat_json` + 10 步 agent 链 |
| ~4,000 问题 → 372 家族 → 719 论文 | N seed → kept / revised / refuted |
| 隐藏的语义 KB | 本地 BM25 + LLM rerank（100+ 摘要）|
| 策划员持续加/剪 idea | `IdeaMiner`（自动从 KB 挖）|
| 撤回 + 补 + 重发 | Director 标记 revise / drop |
| 隐藏的 prompt 调优 | `MetaPromptBuilder` 自改 prompt |
| Lean 形式化审计 | Reviewer + Refuter + line-items + 缺陷报告 |
| 7 阶段内部审稿 | **Director 7 个 agent + 1 仲裁 = 8 阶段** |

---

## 🛣️ 版本路线

| 版本 | 进展 |
|---|---|
| **v0.1** ✅ | 端到端 + 5 seed + Reviewer/Refuter 审计 |
| **v0.2** ✅ | Autopilot + IdeaMiner + MetaPromptBuilder + GitHubIngestor |
| **v0.3** ✅ | Token 累计 + verdict policy 调优 + Director 记忆 + 审计 + Recipe |
| **v0.4** ✅ | CitationGrounder + SectionValidator + arXivIngester + Idea 模板 |
| **v0.5** ✅ | Embedding 索引（可选）+ 多 corpus specs + Self-distillation + 失败预测 |
| **v0.6** ✅ | 自动蒸馏（每 N 个 idea）+ CLI dashboard + HTML 报告 |
| **v0.7** ✅ | HypothesisVerifier（numpy 伪实验）+ Director 并行 + 多 idea 编译 |
| **v0.8** ✅ | mass arXiv（30+ query 并发）+ OpenAlex（2.5亿）+ Wide IdeaMiner + run_mass |
| **v0.9** ✅ | bioRxiv + PubMed（含 Cell/Nature）+ Semantic Scholar + 跨源 dedup |
| **v0.10** ✅ | Claude-Code 风格 rich CLI（banner / tool_call / table / tree / REPL）|
| **v0.11** 🚧 | sklearn 真实实验 + 中文 README + 1.0 roadmap 规划 |

---

## ✏️ 自定义

- **加 seed idea** → 改 `data/seed_ideas.json` 或 `--seeds` 指向你文件
- **让 IdeaMiner 自主挖** → 不传 `--seeds`
- **加 GitHub repo 进 KB** → 改 `autosci/github_ingestor.py` 的 `DEFAULT_REPOS`
- **agent 自改 prompt** → 在任何 agent 的 `run` 里调 `meta_prompter.build_override(...)`
- **换模型** → 改 `.env` 的 `AGNES_MODEL`
- **调检索参数** → `kb_topk=8` / `lit_pool=30`
- **改 verdict 政策** → 编辑 `autosci/pipeline.py` 的 `_set_verdict`
- **加新子 agent** → 写 `autosci/agents/<name>.py`（继承 `AgentBase`），注册到 `agents/__init__.py` + `pipeline.py`

---

## 📜 License

MIT — 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- **OpenAI** 公开的 `github.com/openai/math` 仓库 — 让我们看到：瓶颈不在模型，而在模型**周围**的工作流
- **华大基因 Junhua Li（李俊华）**、**林宇翔 (Yi-Xiang Lin)** — 提出"图模型 > 序列 Transformer"、"多源节点特征"、"序列相似度先验边"等关键设计直觉
- **OpenAlex / arXiv / PubMed / bioRxiv / Semantic Scholar** — 让 5 源 + 跨源去重成为可能
- **`D:\shawn\paper`** 中的 100 篇微生物组论文 — 整个项目的语料底座
- **rich** (Python) — 让我们用纯 Python 复现了 Claude Code 的视觉
- **agnos-ai (Agnes)** — 提供稳定且廉价的 OpenAI 兼容 endpoint

---

## 📮 链接

- GitHub: https://github.com/shawncsx/AutoSCI
- 设计稿: [`docs/superpowers/specs/2026-10-12-autosci-design.md`](docs/superpowers/specs/2026-10-12-autosci-design.md)
- 实施计划: [`docs/superpowers/plans/2026-10-12-autosci-v01-implementation.md`](docs/superpowers/plans/2026-10-12-autosci-v01-implementation.md)

<br>

> 一个人用 Claude Code 风格 CLI，把 OpenAI 数学项目的流水线复刻到自己笔记本上、面向 AI + 生信。这就是 AutoSCI。
