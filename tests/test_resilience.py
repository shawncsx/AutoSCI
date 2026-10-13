"""Resilience tests — verify AutoSCI degrades gracefully when LLM misbehaves.

We deliberately break the LLM (timeout, bad JSON, parse error) and check
that the runner recovers, retries, or fails fast with a clear message.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from autosci.agents._base import AgentBase
from autosci.llm import LLMClient
from autosci.schemas import IdeaRefined, IdeaState, IdeaSeed


class _BadJsonLLM:
    """Returns invalid payloads the first N times, then valid."""

    def __init__(self, fail_first_n: int = 1) -> None:
        self.fail_first_n = fail_first_n
        self.calls = 0

    def chat_json(self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int | None = None) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.fail_first_n:
            # Missing required fields -> coerce() will reject
            return {"unrelated": "shape", "claim_is": "absent_here"}
        # Recovery: always return the good payload
        return {
            "claim": "test claim",
            "primary_hypothesis": "h1",
            "secondary_hypotheses": [],
            "risks": [],
        }
        if "researcher" in system or "可证伪" in system:
            return {
                "claim": "test claim",
                "primary_hypothesis": "h1",
                "secondary_hypotheses": [],
                "risks": [],
            }
        # Fallback for tests that don't go through the schema-specific path
        return {"ok": True}


def test_idea_closer_recovers_from_bad_json():
    fake = _BadJsonLLM(fail_first_n=1)
    agent = AgentBase.__new__(AgentBase)
    agent.name = "idea_closer"
    agent.output_schema = IdeaRefined
    agent.llm = fake
    state = IdeaState(idea_seed=IdeaSeed(id="x", title="t", one_liner="o", angle="a", modalities=[]))
    out = agent.run(state, temperature=0.0)
    assert isinstance(out, IdeaRefined)
    assert out.claim == "test claim"
    assert fake.calls == 2  # bad + recovery


def test_idea_closer_gives_up_after_two_failures():
    fake = _BadJsonLLM(fail_first_n=5)  # never recovers
    agent = AgentBase.__new__(AgentBase)
    agent.name = "idea_closer"
    agent.output_schema = IdeaRefined
    agent.llm = fake
    state = IdeaState(idea_seed=IdeaSeed(id="x", title="t", one_liner="o", angle="a", modalities=[]))
    with pytest.raises(ValueError):
        agent.run(state, temperature=0.0)


def test_llm_client_handles_429_transient(monkeypatch):
    """Simulate a 429 from the API and verify retry logic kicks in."""
    import httpx

    class _MockResp:
        def __init__(self, status: int, body: dict | None = None) -> None:
            self.status_code = status
            self._body = body or {}
            self.request = httpx.Request("POST", "http://x")
        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=self.request, response=self)
        def json(self):
            return self._body

    class _MockClient:
        def __init__(self) -> None:
            self.posts = 0
        def post(self, *args, **kwargs):
            self.posts += 1
            if self.posts <= 1:
                return _MockResp(429)
            return _MockResp(200, {
                "choices":[{"message":{"content":"ok"}}],
                "usage":{"prompt_tokens":10,"completion_tokens":5},
            })

    fake_client = _MockClient()
    llm = LLMClient(api_key="x", client=fake_client)  # type: ignore[arg-type]
    result = llm.chat(
        [{"role":"system","content":"y"},{"role":"user","content":"hi"}],
        temperature=0.0, json_mode=False,
    )
    assert result.content == "ok"
    assert llm.total_tokens_in == 10
    assert llm.total_tokens_out == 5
    assert llm.call_count == 1
    assert llm.total_cost_cny > 0


def test_llm_client_token_accumulator_and_cost():
    """Two successful calls should accumulate tokens + cost."""
    import httpx

    class _MockResp:
        def __init__(self, body: dict) -> None:
            self.status_code = 200
            self._body = body
            self.request = httpx.Request("POST", "http://x")
        def raise_for_status(self): pass
        def json(self): return self._body

    class _MockClient:
        def post(self, *args, **kwargs):
            return _MockResp({
                "choices":[{"message":{"content":"ok"}}],
                "usage":{"prompt_tokens":100,"completion_tokens":50},
            })

    llm = LLMClient(api_key="x", client=_MockClient(), cost_per_1k_in=0.001, cost_per_1k_out=0.002)  # type: ignore[arg-type]
    llm.chat([{"role":"system","content":"s"},{"role":"user","content":"u"}], temperature=0.0, json_mode=False)
    llm.chat([{"role":"system","content":"s"},{"role":"user","content":"u"}], temperature=0.0, json_mode=False)
    assert llm.total_tokens_in == 200
    assert llm.total_tokens_out == 100
    # 200/1000*0.001 + 100/1000*0.002 = 0.0002 + 0.0002 = 0.0004
    assert abs(llm.total_cost_cny - 0.0004) < 1e-6
    assert llm.call_count == 2


def test_with_timeout_changes_default():
    import httpx
    llm = LLMClient(api_key="x", client=httpx.Client(trust_env=False))  # type: ignore[arg-type]
    llm.with_timeout("paper_writer")
    assert llm._per_call_timeout == 180.0
    llm.reset_timeout()
    assert llm._per_call_timeout is None
