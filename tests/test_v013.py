"""Tests for v0.13 — model registry, LLMClient profile routing, MCP server imports."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autosci.llm import LLMClient
from autosci.model_registry import (
    DEFAULT_ROUTING,
    PROVIDER_DEFAULTS,
    ModelProfile,
    ModelRouter,
    _profile_from_dict,
    _resolve_env_placeholders,
    load_profiles,
)


# ---------------------------------------------------------------------------
# model_registry helpers
# ---------------------------------------------------------------------------


def test_resolve_env_placeholders_handles_known_and_unknown(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    assert _resolve_env_placeholders("hello ${FOO}") == "hello bar"
    assert _resolve_env_placeholders("hello $FOO world") == "hello bar world"
    assert _resolve_env_placeholders("hello ${NOT_SET_XYZ}") == "hello ${NOT_SET_XYZ}"


def test_profile_from_dict_resolves_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-123")
    p = _profile_from_dict("foo", {
        "provider": "openai",
        "api_key": "${MY_KEY}",
        "base_url": "https://x.example/v1",
        "model": "x-model",
    })
    assert p.api_key == "secret-123"
    assert p.base_url == "https://x.example/v1"
    assert p.model == "x-model"
    assert p.provider == "openai"


def test_load_profiles_missing_returns_empty(tmp_path: Path):
    assert load_profiles(tmp_path / "nope.yaml") == {}


def test_load_profiles_real_file(tmp_path: Path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "profiles:\n"
        "  fast:\n"
        "    provider: openai\n"
        "    base_url: http://x/v1\n"
        "    model: gpt-fast\n"
        "routing:\n"
        "  idea_closer: fast\n",
        encoding="utf-8",
    )
    profiles = load_profiles(cfg)
    assert "fast" in profiles
    assert profiles["fast"].model == "gpt-fast"


def test_provider_defaults_cover_all_supported():
    for p in ("openai", "anthropic", "deepseek", "qwen", "glm", "custom"):
        assert p in PROVIDER_DEFAULTS


def test_default_routing_has_all_agents():
    expected = {
        "idea_closer", "lit_surveyer", "method_designer", "exp_designer",
        "paper_writer", "citation_grounder", "section_validator", "code_sketcher",
        "reviewer", "refuter", "idea_miner", "rerank",
    }
    assert expected.issubset(set(DEFAULT_ROUTING.keys()))


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


def test_router_resolves_unknown_agent_to_default_tier():
    profs = {"fast": ModelProfile("fast", "openai", "http://x", "k", "fast-model"),
             "smart": ModelProfile("smart", "openai", "http://x", "k", "smart-model")}
    router = ModelRouter(profiles=profs, tier_to_profile={"fast": "fast", "smart": "smart"})
    p = router.resolve("unknown_agent")
    assert p.model == "smart-model"  # default tier


def test_router_resolves_known_agent_to_correct_tier():
    profs = {"fast": ModelProfile("fast", "openai", "http://x", "k", "fast-model"),
             "smart": ModelProfile("smart", "openai", "http://x", "k", "smart-model"),
             "reason": ModelProfile("reason", "openai", "http://x", "k", "reason-model")}
    router = ModelRouter(profiles=profs, tier_to_profile={"fast": "fast", "smart": "smart", "reason": "reason"})
    assert router.resolve("refuter").model == "reason-model"
    assert router.resolve("lit_surveyer").model == "fast-model"
    assert router.resolve("paper_writer").model == "smart-model"


def test_router_falls_back_when_tier_missing():
    """If a tier (e.g. 'reason') is requested but only fast/smart profiles exist,
    fall back to the next-available tier rather than crashing."""
    profs = {"fast": ModelProfile("fast", "openai", "http://x", "k", "fast-model"),
             "smart": ModelProfile("smart", "openai", "http://x", "k", "smart-model")}
    router = ModelRouter(profiles=profs, tier_to_profile={"fast": "fast", "smart": "smart"})
    # 'reason' is requested for refuter but no profile → fallback to 'smart'
    assert router.resolve("refuter").model == "smart-model"
    # also confirm fall-back
    assert router.resolve("lit_surveyer").model == "fast-model"


# ---------------------------------------------------------------------------
# LLMClient accepts profile
# ---------------------------------------------------------------------------


def test_llm_client_accepts_profile(monkeypatch):
    prof = ModelProfile("test", "openai", "https://x/v1", "abc", "test-model",
                         cost_per_1k_in=0.001, cost_per_1k_out=0.002, timeout=10.0)
    client = LLMClient(profile=prof)
    assert client.base_url == "https://x/v1"
    assert client.api_key == "abc"
    assert client.model == "test-model"
    assert client.cost_per_1k_in == 0.001
    assert client.cost_per_1k_out == 0.002


def test_llm_client_with_timeout_routes_via_router():
    fast = ModelProfile("fast", "openai", "https://fast/v1", "kf", "fast-model")
    smart = ModelProfile("smart", "openai", "https://smart/v1", "ks", "smart-model")
    router = ModelRouter(profiles={"fast": fast, "smart": smart},
                         tier_to_profile={"fast": "fast", "smart": "smart"})
    # Override the routing so 'refuter' → 'fast' (we have only fast/smart profiles)
    router.routing = {"refuter": "fast"}
    client = LLMClient(profile=fast, router=router)
    client.with_timeout("refuter")
    assert client.model == "fast-model"
    # Switch to a different agent — model should change to 'smart-model'
    router.routing = {"refuter": "smart", "lit_surveyer": "fast"}
    client.with_timeout("lit_surveyer")
    assert client.model == "fast-model"  # lit_surveyer → 'fast'
    client.with_timeout("refuter")
    assert client.model == "smart-model"
    client.reset_timeout()
