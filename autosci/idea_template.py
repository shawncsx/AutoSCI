"""AutoSCI — Idea template loader (v0.4 new).

Idea templates let a user (or a previous AutoSCI run) constrain the
shape of newly proposed ideas. The format is a small YAML / JSON file:

    {
      "templates": [
        {
          "id": "graph_x_method",
          "name": "Graph model for X method",
          "method_anchor": "Graph Attention Network",
          "dataset_anchor": "MicrobeAtlas / metalog.embl.de",
          "evaluation_anchor": "leave-one-cohort-out",
          "novelty_constraint": "must NOT replicate Waypoint/MGM2's rank encoding",
          "modalities": ["graph_model", "cross_cohort"]
        },
        ...
      ]
    }

The IdeaMiner's prompt is augmented with the active templates, and
``propose_ideas_from_template(llm, kb, template)`` returns ideas that
match the template's structure (single, focused, with all anchors).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from autosci.llm import LLMClient
from autosci.schemas import IdeaSeed, Modality


log = logging.getLogger("autosci.idea_template")


@dataclass
class IdeaTemplate:
    id: str
    name: str
    method_anchor: str = ""
    dataset_anchor: str = ""
    evaluation_anchor: str = ""
    novelty_constraint: str = ""
    modalities: list[str] = field(default_factory=list)
    priority: int = 5
    paper_corpus_hint: str = ""

    def render_prompt_block(self) -> str:
        """Render this template as a directive block for the IdeaMiner prompt."""
        parts = [f"### Template: {self.name} (`{self.id}`)"]
        if self.method_anchor:
            parts.append(f"- **Method anchor (must use):** {self.method_anchor}")
        if self.dataset_anchor:
            parts.append(f"- **Dataset anchor (must use):** {self.dataset_anchor}")
        if self.evaluation_anchor:
            parts.append(f"- **Evaluation anchor (must use):** {self.evaluation_anchor}")
        if self.novelty_constraint:
            parts.append(f"- **Novelty constraint:** {self.novelty_constraint}")
        if self.modalities:
            parts.append(f"- **Modalities:** {', '.join(self.modalities)}")
        if self.paper_corpus_hint:
            parts.append(f"- **Search hint:** {self.paper_corpus_hint}")
        return "\n".join(parts)


def load_templates(path: Path | str) -> list[IdeaTemplate]:
    """Load templates from a JSON file."""
    path = Path(path)
    if not path.exists():
        log.info("No template file at %s; using empty list", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("templates") if isinstance(data, dict) and "templates" in data else data
    if not isinstance(raw, list):
        return []
    out: list[IdeaTemplate] = []
    for entry in raw:
        try:
            out.append(IdeaTemplate(**entry))
        except TypeError as e:
            log.warning("Skip bad template entry: %s", e)
    return out


SYSTEM_WITH_TEMPLATES = """\
你是一位立志不断开新研究课题的「学术星探」。当前任务：根据**一组明确的 idea 模板**生成
IdeaSeed JSON。

每个模板规定：
- 必须使用的方法、必须使用的数据、必须使用的评估协议
- 不能违反的 novelty 约束
- 候选的 modalities

每个 idea **必须严格对应一个模板**，所有锚点都不能省，但可以增加局部创新（不过不能与
novelty_constraint 矛盾）。模板之间不应重复。

输出 JSON 数组（用 `{"ideas":[...]}` 包裹），严格合法 JSON。"""


def propose_from_template(
    llm: LLMClient,
    kb,  # KnowledgeBase
    template: IdeaTemplate,
    n: int = 1,
) -> list[IdeaSeed]:
    """Produce ``n`` IdeaSeed instances that follow ``template``."""
    sample_docs = kb.docs if len(kb.docs) <= 30 else kb.docs[:30]
    from autosci.kb import render_docs_for_llm
    kb_text = render_docs_for_llm(sample_docs, max_chars=3500, per_doc_chars=300)

    user = (
        f"【模板】\n{template.render_prompt_block()}\n\n"
        f"【候选 KB（节选）】\n{kb_text}\n\n"
        f"请按模板生成恰好 {n} 个 IdeaSeed。\n"
        "格式：{\"ideas\":[{\"id\":\"...\",\"title\":\"...\",\"one_liner\":\"...\","
        "\"angle\":\"...\",\"modalities\":[...],\"priority\":N,"
        "\"paper_corpus_hint\":\"...\"}, ...]}"
    )

    if hasattr(llm, "with_timeout"):
        llm.with_timeout("idea_miner")
    try:
        payload = llm.chat_json(system=SYSTEM_WITH_TEMPLATES, user=user, temperature=0.5, max_tokens=1500)
    finally:
        if hasattr(llm, "reset_timeout"):
            llm.reset_timeout()

    raw = payload.get("ideas") if isinstance(payload, dict) else payload
    if isinstance(payload, list):
        raw = payload
    if not isinstance(raw, list):
        log.warning("idea template response malformed: %r", payload)
        return []

    out: list[IdeaSeed] = []
    for entry in raw[:n]:
        try:
            # Force modalities from template if not provided
            if "modalities" not in entry or not entry["modalities"]:
                entry["modalities"] = template.modalities
            seed = IdeaSeed.model_validate(entry)
            out.append(seed)
        except Exception as e:
            log.debug("template-idea dropped: %s", e)
    return out
