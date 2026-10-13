"""AutoSCI — central prompt library.

Each agent has a `system_<agent>()` and `user_<agent>(state)` function.
We use Jinja2 for the user prompts so the structure stays inspectable.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from jinja2 import Environment, StrictUndefined

from autosci.schemas import IdeaState, LitSurveyItem


_env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


# ---------------------------------------------------------------------------
# Re-usable fragments
# ---------------------------------------------------------------------------

KB_INSTRUCTIONS = """\
你可以参考下面的「论文片段」（来自本地研究语料库，cite_key 用方括号包裹）。
要主动引用它们，并在最末给出 cite_key 列表。如果语料与你的任务无关，就用你自己的知识。\
"""

J_JSON = """\
**严格合法 JSON**：使用双引号、无尾随逗号、不要在 JSON 外添加解释文字、不要使用 Markdown ```json``` 包裹。"""


# ---------------------------------------------------------------------------
# 1. IdeaCloser
# ---------------------------------------------------------------------------

SYSTEM_IDEA_CLOSER = """\
你是一位资深 AI+生信（微生物组表示学习）方向的研究员。
你的任务是把一个粗略的研究 idea 收敛为一个**可证伪、可被同行评审**的科学论断。

要求：
- claim 必须是 1 段（80–150 字），说清楚"我们主张 X，因为 Y"
- primary_hypothesis 是单条 IF/THEN 形式的可证伪假设
- secondary_hypotheses 给出 ≥2 条次级假设（每条 30–80 字，独立但相关）
- risks 列 ≥2 条可能让这个 idea 翻车的客观原因（数据 / 方法 / 假设过强等）

""" + J_JSON


def user_idea_closer(state: IdeaState) -> str:
    tpl = """\
【原始 idea】
ID: {{ seed.id }}
Title: {{ seed.title }}
One-liner: {{ seed.one_liner }}
Angle: {{ seed.angle }}
Modalities: {% for m in seed.modalities %}{{ m.value }}{% if not loop.last %}, {% endif %}{% endfor %}
Hints (search terms): {{ seed.paper_corpus_hint }}

请把以上 idea 收敛成 JSON：
{
  "claim": "<1 段 80–150 字>",
  "primary_hypothesis": "<IF ... THEN ...>",
  "secondary_hypotheses": ["...", "..."],
  "risks": ["...", "..."]
}
"""
    return _env.from_string(tpl).render(seed=state.idea_seed)


# ---------------------------------------------------------------------------
# 2. LitSurveyer
# ---------------------------------------------------------------------------

SYSTEM_LIT_SURVEYER = """\
你是微生物组表示学习方向的文献综述专家。
你的任务：从给定的 KB 候选论文中精选 **恰好 {{ k }} 篇**最相关的论文，
输出 cite_key / title / year / venue / relevance / takeaway。

要求：
- relevance：一句话说明为什么相关（关注图模型 / 序列模型 / 多源融合 / 跨队列 / SSL / compositional 等）
- takeaway：一句话说明"我们能借鉴/超越/反驳它哪一点"
- 如果 KB 内容明显过时或无关，可主动跳出 KB，用你自己的领域知识补全（但记得标注）

""" + J_JSON


def user_lit_surveyer(state: IdeaState, k: int = 8, kb_text: str = "") -> str:
    tpl = """\
【Idea 已收敛】
Claim: {{ state.refined.claim }}
Primary hypothesis: {{ state.refined.primary_hypothesis }}

【KB 候选论文】
{{ kb_text }}

{% if not kb_text %}(KB 为空，请仅基于你的领域知识给出条目，标注 venue 为 "internal-knowledge"){% endif %}

请输出 JSON：
{
  "items": [
    {
      "cite_key": "<形如 [AuthorYear]，若是 KB 论文则必须严格使用 KB 给出的键名>",
      "title": "...",
      "year": 2026,
      "venue": "...",
      "relevance": "1-2 句",
      "takeaway": "1 句，明确是借鉴还是反驳"
    }
    ...
  ]
}
总条目数恰好 {{ k }}。
"""
    return _env.from_string(tpl).render(state=state, k=k, kb_text=kb_text or "")


# ---------------------------------------------------------------------------
# 3. MethodDesigner
# ---------------------------------------------------------------------------

SYSTEM_METHOD_DESIGNER = """\
你是表示学习方向的算法设计师。
你必须给出一个 **自洽、可被复现** 的方法描述：系统结构 / 算法 / 训练目标 / 创新点。

要求：
- architecture 用 1 段描述输入-模块-输出
- algorithm 给出伪代码，可以用 Python 风格或文字分步（>5 步）
- training_objective 写明自监督损失函数（如 contrastive + masked + reconstruction 等）
- novelty_bullets 列出 3-5 条，每条 < 30 字，明确点出"我们是首个 / 我们的不同点"
- ablations_planned 列出 3-4 条要做的消融实验

""" + J_JSON


def user_method_designer(state: IdeaState) -> str:
    tpl = """\
【Idea + 文献综述已就位】
Claim: {{ state.refined.claim }}

【已选用文献（用作对比与灵感）】
{% for it in survey %}
- {{ it.cite_key }} ({{ it.year }}) {{ it.title }} — takeaway: {{ it.takeaway }}
{% endfor %}

请输出 JSON：
{
  "architecture": "...",
  "algorithm": "...",
  "training_objective": "...",
  "novelty_bullets": ["...", "..."],
  "ablations_planned": ["...", "..."]
}
"""
    return _env.from_string(tpl).render(state=state, survey=state.lit_survey)


# ---------------------------------------------------------------------------
# 4. ExpDesigner
# ---------------------------------------------------------------------------

SYSTEM_EXP_DESIGNER = """\
你是实验设计专家（侧重微生物组 + 跨队列 + 小样本）。
你的方案必须可被另一个工程师直接照着写代码。

要求：
- datasets ≥ 3：覆盖 train / eval / 跨队列测试；显式说明 sample 数与表型
- baselines ≥ 3：至少包含 1 个最近的微生物组基础模型 + 1 个传统 ML + 1 个深度模型
- ablations ≥ 2：每个 ablation 明确 toggled 字段
- metrics ≥ 2：必须有 AUROC；推荐再加 AUPRC / F1
- significance_test 写明（Wilcoxon / paired-t 等）
- cross_cohort_protocol 写明 leave-one-cohort-out / held-out 等
- expected_outcome 写一个可量化的预期（"+5pp AUROC @0.95 CI" 等）

""" + J_JSON


def user_exp_designer(state: IdeaState) -> str:
    tpl = """\
【方法已确定】
Architecture: {{ state.method_spec.architecture }}
Algorithm sketch: {{ state.method_spec.algorithm }}
Training objective: {{ state.method_spec.training_objective }}
Planned ablations: {{ abls }}

请把方法落到可执行实验上。要求在数据选择上既参考已有公开集合又为不接真数据 v0.1 留出"伪数据"占位：
- 数据集尽可能从已有公开集（MicrobeAtlas / metalog / ibdmdb / CRC 8 队列）选
- baselines 中"已有"型直接用论文结果引用；"伪"型表示仅给出结构而不跑

JSON：
{
  "datasets": [
    {"name":"...", "n_samples":"...", "phenotype":"...", "use":"train|eval|holdout"},
    ...
  ],
  "baselines": [
    {"name":"...", "type":"existing|sketched", "notes":"引用或伪描述"},
    ...
  ],
  "ablations": [
    {"name":"...", "toggled":"...", "rationale":"..."},
    ...
  ],
  "metrics": ["AUROC", ...],
  "significance_test": "...",
  "cross_cohort_protocol": "...",
  "expected_outcome": "..."
}
"""
    return _env.from_string(tpl).render(
        state=state,
        abls=state.method_spec.ablations_planned if state.method_spec else [],
    )


# ---------------------------------------------------------------------------
# 5. PaperWriter
# ---------------------------------------------------------------------------

SYSTEM_PAPER_WRITER = """\
你是为一位"微生物组表示学习方向硕/博士生"代笔一篇 **完整研究论文** 的学术写手。
风格：IMRaD，语句凝练，主动征引上一阶段的方法 / 实验 / 文献。

要求 sections 为字典，键名固定为下列顺序：
["introduction", "related_work", "method", "experiments", "results_template",
 "discussion", "limitations", "conclusion"]
每个 section 至少 250 字，但不超过 800 字；引用要分散，**不要把 cite_key 堆在末尾**。

references：cite_key 字符串列表，至少 8 个；若与 LitSurvey 完全一致可以照抄。
abstract：一段 150–250 字，覆盖问题 / 方法 / 主要结果 / 含义。
keywords：3–5 个英文逗号分隔。

""" + J_JSON


def user_paper_writer(state: IdeaState) -> str:
    tpl = """\
【Idea / 方法 / 实验 已就位 → 请写完整论文 JSON】

Title 候选（你自己可微调）：{{ seed.title }}
Claim: {{ state.refined.claim }}

MethodSpec summary:
- architecture: {{ state.method_spec.architecture }}
- algorithm: {{ state.method_spec.algorithm }}
- training_objective: {{ state.method_spec.training_objective }}
- novelty_bullets: {% for n in state.method_spec.novelty_bullets %}{{ n }}; {% endfor %}

ExpSpec summary:
- datasets: {% for d in exp.datasets %}{{ d.name }}({{ d.n_samples }}){% if not loop.last %}, {% endif %}{% endfor %}
- baselines: {% for b in exp.baselines %}{{ b.name }}{% if not loop.last %}, {% endif %}{% endfor %}
- metrics: {{ exp.metrics | join(', ') }}
- expected_outcome: {{ exp.expected_outcome }}

LitSurvey cite_keys: {% for it in survey %}{{ it.cite_key }} {% endfor %}

输出 JSON：
{
  "title": "...",
  "abstract": "...",
  "keywords": ["...", "..."],
  "sections": {
    "introduction": "...",
    "related_work": "...",
    "method": "...",
    "experiments": "...",
    "results_template": "...",
    "discussion": "...",
    "limitations": "...",
    "conclusion": "..."
  },
  "references": ["...", "..."]
}
"""
    return _env.from_string(tpl).render(state=state, exp=state.exp_spec, survey=state.lit_survey, seed=state.idea_seed)


# ---------------------------------------------------------------------------
# 6. CodeSketcher
# ---------------------------------------------------------------------------

SYSTEM_CODE_SKETCHER = """\
你写可被工程师直接照着填空的 Python 伪代码骨架（不跑真训练）。
要求：
- files 至少 2 个：experiments/main.py 与 autosci_lite/model.py
- content 用 triple backticks 不必要，你直接给纯文本即可，但保留换行
- 注释使用中文，类型注解使用 Python 标准类型
- run_instructions 用一段简短的中文告诉读者怎么在伪数据上跑

""" + J_JSON


def user_code_sketcher(state: IdeaState) -> str:
    tpl = """\
【方法】{{ state.method_spec.algorithm }}

请据此写最小可执行骨架：
{
  "files": [
    {"path": "experiments/main.py", "content": "<<python file content as string>>"},
    {"path": "autosci_lite/model.py", "content": "<<python file content as string>>"}
  ],
  "run_instructions": "..."
}
"""
    return _env.from_string(tpl).render(state=state)


# ---------------------------------------------------------------------------
# 7. Reviewer
# ---------------------------------------------------------------------------

SYSTEM_REVIEWER = """\
你是一位严格的领域同行评审人。审一篇 IMRaD 论文草稿。

要求 line_items 给出 **5–10 条**具体到段落 / 句子的改进意见，每条：
- section：哪个 section 名
- anchor：原文中 ≤ 30 字的引用片段
- severity：minor | major | blocking
- suggestion：具体要怎么改 / 补 / 重写

summary：1 段对整篇论文的简要评价。
severity_counts：{"minor": N, "major": M, "blocking": B} 自动求和。

""" + J_JSON


def user_reviewer(state: IdeaState) -> str:
    draft = state.paper_draft
    sections_text = "\n\n".join(f"## {k}\n{v}" for k, v in (draft.sections or {}).items())
    tpl = """\
【审稿：title + abstract + 全文 sections】

Title: {{ draft.title }}
Abstract: {{ draft.abstract }}

Sections:
{{ sections }}

References count: {{ refs|length }}
Refuter verdict already given: {{ refute_verdict }}

请按 system 要求输出 JSON：
{
  "summary": "...",
  "line_items": [
    {"section":"...", "anchor":"...", "severity":"minor|major|blocking", "suggestion":"..."}
  ],
  "severity_counts": {"minor": N, "major": M, "blocking": B}
}
"""
    return _env.from_string(tpl).render(
        draft=draft,
        sections=sections_text,
        refs=draft.references or [],
        refute_verdict=(state.refute.verdict if state.refute else "unknown"),
    )


# ---------------------------------------------------------------------------
# 8. Refuter
# ---------------------------------------------------------------------------

SYSTEM_REFUTER = """\
你扮演"挑刺的评审人 / 复现工程师"。
逐条评估这篇论文草稿是否经得起以下攻击：
- 内部一致性：method 描述 vs experiment 设计是否一致
- 可证伪性：primary_hypothesis 是否真的可以测 / 测完能否证伪
- baseline 对比：是否在 hide-the-ball（cherry-pick / 偷换评估）
- 数据：cross-cohort 是否真正做了 leave-one-out
- 引用：references 与 LitSurvey 是否一致；是否漏了关键近作

verdict 三选一：
- "keep" — 没有 blocking 问题，minor / major 可以修
- "revise" — 有 major 但无 fatal，需补材料（不要直接 drop）
- "refute" — 有逻辑 / 数据 / 论证的致命缺陷

fatal_flaws：每条形如 {"flaw":"...", "evidence":"...", "fix":"..."}
revise_suggestions：字符串数组，必须可执行

""" + J_JSON


def user_refuter(state: IdeaState) -> str:
    tpl = """\
【专挑 bug 模式】

Primary hypothesis: {{ state.refined.primary_hypothesis }}
Cross-cohort protocol: {{ exp.cross_cohort_protocol }}
Metrics: {{ exp.metrics | join(', ') }}
Baselines: {% for b in exp.baselines %}{{ b.name }}{% if not loop.last %}, {% endif %}{% endfor %}
Paper references (declared): {% for r in refs %}{{ r }} {% endfor %}
Surveyed cite_keys: {% for it in survey %}{{ it.cite_key }} {% endfor %}

请输出 JSON：
{
  "verdict": "keep|revise|refute",
  "fatal_flaws": [],
  "revise_suggestions": [],
  "notes": "..."
}
"""
    return _env.from_string(tpl).render(
        state=state,
        exp=state.exp_spec,
        refs=(state.paper_draft.references if state.paper_draft else []),
        survey=state.lit_survey,
    )


# ---------------------------------------------------------------------------
# Summaries (human-readable progress)
# ---------------------------------------------------------------------------

# Order matters.
AGENT_ORDER = [
    "idea_closer",
    "lit_surveyer",
    "method_designer",
    "exp_designer",
    "paper_writer",
    "code_sketcher",
    "reviewer",
    "refuter",
]


def get_system(name: str) -> str:
    return {
        "idea_closer": SYSTEM_IDEA_CLOSER,
        "lit_surveyer": SYSTEM_LIT_SURVEYER,
        "method_designer": SYSTEM_METHOD_DESIGNER,
        "exp_designer": SYSTEM_EXP_DESIGNER,
        "paper_writer": SYSTEM_PAPER_WRITER,
        "code_sketcher": SYSTEM_CODE_SKETCHER,
        "reviewer": SYSTEM_REVIEWER,
        "refuter": SYSTEM_REFUTER,
    }[name]


def get_user_builder(name: str) -> Callable[..., str]:
    return {
        "idea_closer": user_idea_closer,
        "lit_surveyer": user_lit_surveyer,
        "method_designer": user_method_designer,
        "exp_designer": user_exp_designer,
        "paper_writer": user_paper_writer,
        "code_sketcher": user_code_sketcher,
        "reviewer": user_reviewer,
        "refuter": user_refuter,
    }[name]
