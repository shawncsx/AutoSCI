"""AutoSCI — Model registry (v0.13).

Decouples LLMClient from any single provider. Users declare N named
"profiles" in YAML/JSON, then route each sub-agent to one profile.

Why? Different agents have different cost/latency trade-offs:
    - IdeaCloser  needs creativity → use a *larger* model
    - PaperWriter needs depth  → use the *best* model you can afford
    - IdeaMiner  needs volume  → use a *cheap* model (e.g. mini)
    - Reviewer   needs critique→ use a *reasoning* model
    - CitationGrounder is mechanical → use the *cheapest* model

Default behaviour (no profile file): fall back to the AGNES_* env vars
the user already had.

Supported providers
------------------
    "openai"     - OpenAI / Azure OpenAI / any OpenAI-compatible
    "anthropic"  - Anthropic Claude (messages API; we translate)
    "deepseek"   - DeepSeek (OpenAI-compatible; just point base_url)
    "qwen"       - Aliyun Qwen / DashScope (OpenAI-compatible)
    "glm"        - Zhipu GLM (OpenAI-compatible; z.ai)
    "custom"     - user-supplied base_url

The provider adapter is small (mostly the request body shape + auth
header). Everything else (retries, JSON-mode, cost accounting) is shared.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml  # bundled with pyyaml; AutoSCI only hard-requires pyyaml>=6


log = logging.getLogger("autosci.model_registry")


# ---------------------------------------------------------------------------
# Provider catalog — sensible defaults the user can override
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": {
            "fast":   "gpt-4o-mini",
            "smart":  "gpt-4o",
            "reason": "o1-preview",
        },
        "auth_header": "Authorization: Bearer {api_key}",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "models": {
            "fast":   "claude-3-5-haiku-20241022",
            "smart":  "claude-3-5-sonnet-20241022",
            "reason": "claude-3-5-sonnet-20241022",
        },
        "auth_header": "x-api-key: {api_key}",
        # Anthropic uses a /messages endpoint with a different body shape —
        # we always translate to the OpenAI-style /chat/completions for
        # shared handling; this is implemented in ``AnthropicAdapter``.
        "translate_to_openai": True,
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": {
            "fast":   "deepseek-chat",
            "smart":  "deepseek-chat",
            "reason": "deepseek-reasoner",
        },
        "auth_header": "Authorization: Bearer {api_key}",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": {
            "fast":   "qwen-turbo",
            "smart":  "qwen-plus",
            "reason": "qwen-max",
        },
        "auth_header": "Authorization: Bearer {api_key}",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": {
            "fast":   "glm-4-flash",
            "smart":  "glm-4-plus",
            "reason": "glm-4-plus",
        },
        "auth_header": "Authorization: Bearer {api_key}",
    },
    "custom": {
        # user must provide base_url + model
        "auth_header": "Authorization: Bearer {api_key}",
    },
}


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """A single named LLM profile that LLMClient can switch to."""
    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    translate_to_openai: bool = False
    extra_headers: dict = field(default_factory=dict)
    cost_per_1k_in: float = 0.002
    cost_per_1k_out: float = 0.006
    timeout: float = 60.0
    note: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_env_placeholders(value: str) -> str:
    """Replace ``${VAR}`` or ``$VAR`` with ``os.environ[VAR]``."""
    if not isinstance(value, str):
        return value
    # ${VAR}
    def _sub(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    value = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", _sub, value)
    # $VAR (whole word)
    value = re.sub(r"\$([A-Z_][A-Z0-9_]*)",
                   lambda m: os.environ.get(m.group(1), m.group(0)),
                   value)
    return value


def _profile_from_dict(name: str, raw: dict) -> ModelProfile:
    provider = raw.get("provider", "openai").lower()
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    if "api_key" in raw:
        api_key = _resolve_env_placeholders(str(raw["api_key"]))
    elif "api_key_env" in raw:
        api_key = os.environ.get(raw["api_key_env"], "")
    else:
        # fall back to legacy AGNES_* env vars
        api_key = os.environ.get("AGNES_API_KEY", "")
    base_url = raw.get("base_url") or os.environ.get("AGNES_BASE_URL") or defaults.get("base_url", "")
    model = raw.get("model") or defaults.get("models", {}).get("smart", "")
    return ModelProfile(
        name=name,
        provider=provider,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        translate_to_openai=bool(defaults.get("translate_to_openai", False)),
        extra_headers=dict(defaults.get("extra_headers", {})),
        cost_per_1k_in=float(raw.get("cost_per_1k_in", defaults.get("cost_per_1k_in", 0.002))),
        cost_per_1k_out=float(raw.get("cost_per_1k_out", defaults.get("cost_per_1k_out", 0.006))),
        timeout=float(raw.get("timeout", 60.0)),
        note=raw.get("note", ""),
    )


def load_profiles(path: Path | str = "data/model_profiles.yaml") -> dict[str, ModelProfile]:
    """Load the profile YAML. Returns an empty dict if the file is missing."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = _load_yaml(path)
    except Exception as e:
        log.warning("Failed to load profiles from %s: %s", path, e)
        return {}
    profiles = raw.get("profiles") or raw or {}
    if not isinstance(profiles, dict):
        return {}
    return {n: _profile_from_dict(n, p) for n, p in profiles.items()}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


# Default per-agent routing. Users can override in model_profiles.yaml.
DEFAULT_ROUTING: dict[str, str] = {
    "idea_closer":       "smart",
    "lit_surveyer":      "fast",
    "method_designer":    "smart",
    "exp_designer":       "smart",
    "paper_writer":       "smart",
    "citation_grounder":  "fast",
    "section_validator":  "fast",
    "code_sketcher":      "fast",
    "reviewer":           "reason",
    "refuter":            "reason",
    "idea_miner":         "fast",
    "rerank":             "fast",
}


def default_profile_from_env() -> ModelProfile:
    """Build a single profile from legacy AGNES_* env vars (for users who
    haven't migrated to the YAML yet)."""
    return _profile_from_dict(
        "default",
        {
            "provider": os.environ.get("AGNES_PROVIDER", "openai"),
            "model": os.environ.get("AGNES_MODEL", "agnes-3.0-flash"),
        },
    )


class ModelRouter:
    """Resolve an agent name → ModelProfile."""

    def __init__(
        self,
        profiles: dict[str, ModelProfile] | None = None,
        routing: dict[str, str] | None = None,
        tier_to_profile: dict[str, str] | None = None,
        default_tier: str = "smart",
    ) -> None:
        self.profiles = profiles or (load_profiles() or {})
        if not self.profiles:
            self.profiles = {"default": default_profile_from_env()}
        self.routing = dict(DEFAULT_ROUTING)
        if routing:
            self.routing.update(routing)
        # Tier -> profile-name mapping. v0.13: if caller passes
        # `tier_to_profile`, use that; otherwise auto-build from profile
        # names that match a tier.
        self.tier_to_profile: dict[str, str] = {}
        if tier_to_profile:
            self.tier_to_profile.update(tier_to_profile)
        else:
            for prof_name, prof in self.profiles.items():
                if prof_name in DEFAULT_ROUTING.values() or prof_name in ("fast", "smart", "reason"):
                    for tier in ("fast", "smart", "reason"):
                        self.tier_to_profile.setdefault(tier, prof_name)

    def resolve(self, agent_name: str) -> ModelProfile:
        tier = self.routing.get(agent_name, "smart")
        prof_name = self.tier_to_profile.get(tier)
        if prof_name is None:
            # Fallback: try the next-best tier. Order: same-tier → smart → fast → any.
            for fallback in ("smart", "fast", "reason"):
                if fallback in self.tier_to_profile:
                    prof_name = self.tier_to_profile[fallback]
                    break
            if prof_name is None:
                prof_name = next(iter(self.profiles))
        return self.profiles[prof_name]

    def all_profiles(self) -> dict[str, ModelProfile]:
        return dict(self.profiles)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="List configured model profiles")
    parser.add_argument("--profiles", default="data/model_profiles.yaml")
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    if not profiles:
        print("(no profiles — using AGNES_* env vars)")
        return 0
    for name, p in profiles.items():
        print(f"{name:24s}  {p.provider:8s}  {p.model:30s}  {p.base_url}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
