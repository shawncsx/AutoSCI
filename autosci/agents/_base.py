"""AutoSCI — base class for all per-step agents.

Each sub-agent in the ResearchTeam shares the same plumbing:
- A system prompt (from prompts.get_system)
- A user prompt builder (from prompts.get_user_builder)
- A Pydantic schema for the validated output
- An AgentRunner that:
    1. Builds the messages,
    2. Calls LLMClient.chat_json,
    3. Validates with Pydantic,
    4. Tries one repair pass on validation failure,
    5. Returns the validated pydantic instance.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from autosci.llm import LLMClient
from autosci.prompts import get_system, get_user_builder


log = logging.getLogger("autosci.agents")


def _load_distillation_patch(name: str) -> str:
    """Load any prompt patches distilled from prior runs."""
    try:
        from autosci.distiller import load_distillation
        d = load_distillation()
        if d is not None:
            return d.patches_for(name)
    except Exception:
        pass
    return ""

P = TypeVar("P", bound=BaseModel)


class AgentBase:
    """Base class — subclass and provide `name` + `output_schema`."""

    name: str = ""
    output_schema: Type[P] = BaseModel  # type: ignore[assignment]

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build_messages(self, state: Any, **kwargs: Any) -> list[dict[str, str]]:
        """Compose messages. Default = system + user(template)."""
        system = get_system(self.name)
        builder = get_user_builder(self.name)
        user = builder(state, **kwargs) if self.name == "lit_surveyer" else builder(state)
        # v0.5: append any distilled patch from prior kept runs
        user = user + _load_distillation_patch(self.name)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _coerce(self, payload: Any) -> P:
        """Map LLM payload dict → pydantic model. Tolerant to wrapping keys."""
        # The model sometimes wraps in `{"result": {...}}` or `{"items": [...]}`.
        if not isinstance(payload, dict):
            raise ValueError(f"payload not dict: {payload!r}")
        # Try as-is first; on failure, try common wrappers.
        candidates = [payload]
        if "result" in payload and isinstance(payload["result"], dict):
            candidates.append(payload["result"])
        if "data" in payload and isinstance(payload["data"], dict):
            candidates.append(payload["data"])
        if "items" in payload and isinstance(payload["items"], list):
            # For LitSurveyer-style: items + maybe top-level fields
            new = dict(payload)
            new.pop("items")
            new["items"] = payload["items"]
            candidates.append(new)
        last_err: Optional[Exception] = None
        for c in candidates:
            try:
                return self.output_schema.model_validate(c)
            except ValidationError as e:
                last_err = e
                continue
        raise ValueError(
            f"Could not coerce payload into {self.output_schema.__name__}: {last_err}"
        )

    def run(self, state: Any, **kwargs: Any) -> P:
        messages = self.build_messages(state, **kwargs)
        temperature = kwargs.pop("temperature", 0.2)
        # JSON mode is on by default; we want strict schema, so we go via chat_json.
        # Apply per-agent timeout override before the call.
        if hasattr(self.llm, "with_timeout"):
            self.llm.with_timeout(self.name)
        try:
            payload = self.llm.chat_json(
                system=messages[0]["content"],
                user=messages[1]["content"],
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens"),
            )
        except Exception as e:
            log.error("[%s] LLM call failed: %s", self.name, e)
            raise
        finally:
            if hasattr(self.llm, "reset_timeout"):
                self.llm.reset_timeout()

        try:
            return self._coerce(payload)
        except ValueError as e:
            log.warning("[%s] first-parse failed (%s); retrying repair", self.name, e)
            repair_user = (
                "你上一次的回复无法被解析为我们的 JSON schema。请重新分析问题并**严格**按下面 schema 输出。"
                "只输出 JSON，不要任何解释文字。\n"
                f"原问题：\n{messages[1]['content']}\n"
                f"原回复（可能错）：\n{payload}\n"
            )
            if hasattr(self.llm, "with_timeout"):
                self.llm.with_timeout(self.name)
            try:
                payload2 = self.llm.chat_json(
                    system=(
                        "你是一名'严格 JSON 输出'专家。请只输出合法 JSON。"
                        "使用双引号、无尾随逗号、不要附加说明。"
                    ),
                    user=repair_user,
                    temperature=0.0,
                )
            finally:
                if hasattr(self.llm, "reset_timeout"):
                    self.llm.reset_timeout()
            try:
                return self._coerce(payload2)
            except ValueError as e2:
                log.error("[%s] second-parse failed: %s", self.name, e2)
                raise
