"""AutoSCI — PromptDistiller (v0.5).

Reads every *kept* paper in the vault and asks the LLM what made them
good. The output is a single ``distillation.json`` file:

    {
      "common_strengths": ["uses concrete numbers", "explicit ablation", ...],
      "common_weaknesses_in_revised": ["vague baselines", ...],
      "prompt_patches": {
        "method_designer": "Always state the activation function choice explicitly.",
        "paper_writer": "When describing method, include a complexity analysis (O(n^2)).",
        ...
      }
    }

The orchestrator (``director.py``) loads this file on startup and
appends the patches to the corresponding agent's user prompt — so
AutoSCI literally self-improves from its own successful runs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from autosci.llm import LLMClient
from autosci.schemas import PaperDraft


log = logging.getLogger("autosci.distiller")


DISTILLATION_SYSTEM = """\
你是一名 AutoSCI 的自我蒸馏员。给你若干 AutoSCI 已 keep 的论文 + 若干 revised 的论文，
找出「让 keep 比 revised 好的具体写作/方法模式」。

要求：
- common_strengths：3-5 条 keep 论文共有的好模式
- common_weaknesses_in_revised：3-5 条 revised 论文反复犯的错
- prompt_patches：3-5 条具体可执行的 prompt 补丁，每条形如
    "<agent_name>": "<一段 ≤ 60 字的中文补丁，可直接加到该 agent 的 user prompt 末尾>"

严格合法 JSON 输出。
"""

DISTILL_PATH = Path("data/distillation.json")


@dataclass
class DistillationReport:
    common_strengths: list[str]
    common_weaknesses_in_revised: list[str]
    prompt_patches: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "common_strengths": self.common_strengths,
                "common_weaknesses_in_revised": self.common_weaknesses_in_revised,
                "prompt_patches": self.prompt_patches,
            },
            ensure_ascii=False,
            indent=2,
        )

    def patches_for(self, agent_name: str) -> str:
        if agent_name in self.prompt_patches:
            return "\n\n[distillation patch] " + self.prompt_patches[agent_name]
        return ""


def load_distillation(path: Path | str = DISTILL_PATH) -> DistillationReport | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DistillationReport(
            common_strengths=data.get("common_strengths", []),
            common_weaknesses_in_revised=data.get("common_weaknesses_in_revised", []),
            prompt_patches=data.get("prompt_patches", {}),
        )
    except Exception as e:
        log.warning("Failed to load distillation: %s", e)
        return None


def collect_papers(vault_root: Path | str) -> tuple[list[dict], list[dict]]:
    """Walk the vault and return (kept_papers, revised_papers) as small dicts."""
    vault_root = Path(vault_root)
    kept: list[dict] = []
    revised: list[dict] = []
    for sub, target in [("kept", kept), ("refuted", revised), ("kept", kept)]:
        # 'kept' contains both kept and revised; we approximate: if reason.md
        # is present it's refuted, otherwise treat as kept.
        pass
    # Re-walk with the proper classification:
    for bucket, store in [("kept", kept), ("refuted", revised)]:
        d = vault_root / bucket
        if not d.exists():
            continue
        for p in d.glob("*/paper.md"):
            store.append({
                "id": p.parent.name,
                "path": str(p),
                "text": _safe_read(p),
            })
    return kept, revised


def _safe_read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:
        return ""


def run_distillation(
    llm: LLMClient,
    vault_root: Path | str = "vault",
    out_path: Path | str = DISTILL_PATH,
) -> DistillationReport | None:
    kept, revised = collect_papers(vault_root)
    if not kept and not revised:
        log.info("no vault papers to distill")
        return None
    if not kept:
        log.info("no kept papers; skipping distillation")
        return None
    # Build prompt
    sample_kept = "\n\n".join(f"### {p['id']}\n{p['text']}" for p in kept[:3])
    sample_revised = "\n\n".join(f"### {p['id']}\n{p['text']}" for p in revised[:3]) or "(无)"
    user = (
        f"【已 keep 的论文（节选）】\n{sample_kept}\n\n"
        f"【已 revised 的论文（节选）】\n{sample_revised}\n\n"
        "请按 system 要求输出严格合法 JSON。"
    )
    if hasattr(llm, "with_timeout"):
        llm.with_timeout("idea_miner")  # similar long-form reasoning budget
    try:
        payload = llm.chat_json(system=DISTILLATION_SYSTEM, user=user, temperature=0.3, max_tokens=1800)
    except Exception as e:
        log.warning("distillation LLM call failed: %s", e)
        return None
    finally:
        if hasattr(llm, "reset_timeout"):
            llm.reset_timeout()

    try:
        report = DistillationReport(
            common_strengths=list(payload.get("common_strengths", []) or []),
            common_weaknesses_in_revised=list(payload.get("common_weaknesses_in_revised", []) or []),
            prompt_patches=dict(payload.get("prompt_patches", {}) or {}),
        )
    except Exception as e:
        log.warning("bad distillation payload: %s", e)
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report.to_json(), encoding="utf-8")
    log.info("wrote distillation report -> %s", out_path)
    return report
