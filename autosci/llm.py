"""AutoSCI — LLM client (Agnes / OpenAI-compatible).

Wraps httpx with retries, JSON-mode, and structured-output parsing.
We intentionally avoid openai-sdk to keep dependency footprint light.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()  # picks up .env

log = logging.getLogger("autosci.llm")


@dataclass
class ChatResult:
    content: str
    tokens_in: int
    tokens_out: int
    raw: dict
    elapsed_s: float


class LLMClient:
    """Thin OpenAI-compatible chat client tuned for the Agnes endpoint.

    Parameters
    ----------
    base_url : default from env AGNES_BASE_URL.
    api_key  : default from env AGNES_API_KEY.
    model    : default from env AGNES_MODEL.
    timeout  : HTTP timeout seconds (default 60s).
    max_retries : retry budget for transient failures (default 3).
    profile  : an optional ModelProfile (from model_registry); when set
               it overrides the legacy base_url/api_key/model args.
    router   : an optional ModelRouter; when set, ``with_timeout(agent_name)``
               also routes to the right profile for that agent.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        cost_per_1k_in: Optional[float] = None,
        cost_per_1k_out: Optional[float] = None,
        profile: Any = None,
        router: Any = None,
    ) -> None:
        # v0.13 — accept a ModelProfile directly
        self._profile = profile
        self._router = router
        if profile is not None:
            self.base_url = profile.base_url.rstrip("/")
            self.api_key = profile.api_key
            self.model = profile.model
            self.cost_per_1k_in = profile.cost_per_1k_in
            self.cost_per_1k_out = profile.cost_per_1k_out
            self.timeout = profile.timeout
        else:
            self.base_url = (
                base_url
                or os.environ.get("AGNES_BASE_URL")
                or "https://api.agnes-ai.cn/v1"
            ).rstrip("/")
            self.api_key = api_key or os.environ.get("AGNES_API_KEY") or ""
            self.model = model or os.environ.get("AGNES_MODEL") or "agnes-3.0-flash"
            self.cost_per_1k_in = cost_per_1k_in if cost_per_1k_in is not None else float(
                os.environ.get("AUTOSCI_COST_IN", "0.002")
            )
            self.cost_per_1k_out = cost_per_1k_out if cost_per_1k_out is not None else float(
                os.environ.get("AUTOSCI_COST_OUT", "0.006")
            )
            self.timeout = timeout or float(os.environ.get("AUTOSCI_LLM_TIMEOUT_S") or 60)
        self.max_retries = max_retries or int(os.environ.get("AUTOSCI_MAX_RETRIES") or 3)
        # Per-call timeout overrides (long-output agents can set higher)
        self._per_call_timeout: Optional[float] = None
        # Lifetime token + cost accumulators (read by Director for reporting)
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.total_cost_cny: float = 0.0
        self.call_count: int = 0
        # `trust_env=False` stops httpx from auto-reading HTTP_PROXY/HTTPS_PROXY/NO_PROXY
        # from the OS environment. Some Windows setups set NO_PROXY with values
        # that httpx 0.28 mis-parses (treating them as proxy URLs).
        self._client = client or httpx.Client(timeout=self.timeout, trust_env=False)
        if not self.api_key:
            log.warning("AGNES_API_KEY not set; calls will fail unless provided at runtime.")

    # ------------------------------------------------------------------
    # per-agent timeout overrides
    # ------------------------------------------------------------------

    AGENT_TIMEOUTS: dict[str, float] = {
        # long-output agents that can take 1-3 minutes
        "paper_writer": 180.0,
        "code_sketcher": 120.0,
        "refuter": 90.0,
        "method_designer": 90.0,
        "exp_designer": 90.0,
        "citation_grounder": 60.0,
        "section_validator": 120.0,  # batched eval may output up to 8 section ratings
        # shorter-output agents
        "idea_closer": 60.0,
        "lit_surveyer": 90.0,
        "reviewer": 90.0,
        "idea_miner": 120.0,
        "meta_prompter": 60.0,
        "rerank": 60.0,
    }

    def with_timeout(self, agent_name: str) -> "LLMClient":
        """Return a lightweight wrapper view that sets a per-call timeout.

        v0.13: if a router is attached, switch this client's base_url /
        model / api_key to the resolved profile for ``agent_name`` so the
        actual HTTP call hits the right provider.
        """
        t = self.AGENT_TIMEOUTS.get(agent_name, self.timeout)
        # We can't easily subclass the client per-call, so we mutate _per_call_timeout
        # (callers should restore after their call).
        self._per_call_timeout = t
        # v0.13 — per-agent profile routing
        if self._router is not None:
            try:
                prof = self._router.resolve(agent_name)
                self.base_url = prof.base_url.rstrip("/")
                self.api_key = prof.api_key
                self.model = prof.model
                self.cost_per_1k_in = prof.cost_per_1k_in
                self.cost_per_1k_out = prof.cost_per_1k_out
            except Exception as e:
                log.debug("router.resolve(%s) failed: %s", agent_name, e)
        return self

    def reset_timeout(self) -> None:
        self._per_call_timeout = None

    # ---------------------------------------------------------------
    # main public method
    # ---------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> ChatResult:
        """Run a chat completion.

        Parameters
        ----------
        messages : OpenAI-style list, e.g. [{"role":"system","content":"..."}, ...]
        temperature : sampling temperature.
        json_mode : if True, force `response_format={"type":"json_object"}` AND
                    retry parsing failures.
        max_tokens : optional cap.
        extra_body : appended to the request body.
        """
        if not messages:
            raise ValueError("messages must be non-empty")
        if messages[-1]["role"] != "user":
            raise ValueError("the last message must have role='user'")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if extra_body:
            body.update(extra_body)

        # Per-call timeout override
        timeout_s = self._per_call_timeout or self.timeout
        # We swap to a new client if timeout changed mid-call
        active_client = self._client
        if timeout_s != self.timeout:
            active_client = httpx.Client(timeout=timeout_s, trust_env=False)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = active_client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                if resp.status_code >= 500 or resp.status_code in (408, 429):
                    raise httpx.HTTPStatusError(
                        f"transient {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}) or {}
                tokens_in = int(usage.get("prompt_tokens", 0))
                tokens_out = int(usage.get("completion_tokens", 0))
                # Update lifetime stats
                self.total_tokens_in += tokens_in
                self.total_tokens_out += tokens_out
                self.total_cost_cny += (
                    tokens_in / 1000.0 * self.cost_per_1k_in
                    + tokens_out / 1000.0 * self.cost_per_1k_out
                )
                self.call_count += 1
                return ChatResult(
                    content=content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    raw=data,
                    elapsed_s=time.time() - t0,
                )
            except (httpx.HTTPError, httpx.HTTPStatusError, ValueError) as e:
                last_exc = e
                # 429 rate-limit: longer backoff (5s, 15s, 30s capped)
                # other transients: standard 1s, 2s, 4s
                is_rate_limit = (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response is not None
                    and e.response.status_code == 429
                )
                if is_rate_limit:
                    backoff = min(30.0, 5.0 * (2 ** (attempt - 1)))
                else:
                    backoff = min(8.0, 2 ** attempt * 0.5)
                log.warning(
                    "LLM call attempt %d/%d failed (%s%s); sleeping %.1fs",
                    attempt, self.max_retries, type(e).__name__,
                    " [rate-limited]" if is_rate_limit else "",
                    backoff,
                )
                time.sleep(backoff)
                continue
            except Exception as e:
                # Non-transient; bubble up
                raise
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_exc}")
        # cleanup temporary client if we made one
        if active_client is not self._client:
            active_client.close()

    # ---------------------------------------------------------------
    # convenience helpers
    # ---------------------------------------------------------------

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Convenience wrapper: ask for JSON, parse it, retry if parse fails.

        On parse failure we re-ask once, prefixing the user message with
        a parse-failure note + the last raw content.
        """
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = self.chat(
            msgs,
            temperature=temperature,
            json_mode=True,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            # retry once with a correction
            correction_user = (
                "你的上一次回复不是合法 JSON。原文：\n"
                "```\n" + result.content + "\n```\n"
                "请重新用**严格合法 JSON** 回复，无前后解释文字，且**使用双引号**。"
            )
            msgs2 = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": correction_user},
            ]
            result2 = self.chat(
                msgs2,
                temperature=0.0,
                json_mode=True,
                max_tokens=max_tokens,
            )
            return json.loads(result2.content)


# ---------------------------------------------------------------------------
# module-level singleton, lazy-initialised
# ---------------------------------------------------------------------------

_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
