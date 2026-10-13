"""LLM Client integration smoke — requires AGNES_API_KEY in .env.

Run with:
    python -m pytest tests/test_llm.py -v
or:
    AUTOSCI_RUN_NET_TESTS=1 python -m pytest tests/test_llm.py -v
"""
from __future__ import annotations

import json
import os

import pytest

from autosci.llm import LLMClient


pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOSCI_RUN_NET_TESTS") != "1",
    reason="network tests skipped unless AUTOSCI_RUN_NET_TESTS=1",
)


@pytest.fixture(scope="module")
def client():
    api_key = os.environ.get("AGNES_API_KEY")
    if not api_key:
        pytest.skip("AGNES_API_KEY not set in env")
    return LLMClient()


def test_chat_returns_content(client):
    result = client.chat(
        [
            {"role": "system", "content": "你是一个简短的助手。"},
            {"role": "user", "content": "请用一个名词回答：DNA 的中文别名是什么？"},
        ],
        temperature=0.0,
        json_mode=False,
        max_tokens=20,
    )
    assert isinstance(result.content, str) and len(result.content) > 0
    print("\n[LLM sample content]", repr(result.content))


def test_chat_json_parses(client):
    payload = client.chat_json(
        system="你是 JSON 助手。",
        user='请返回一个 JSON 对象，字段为 `{"answer":"DNA","length":3}`。',
    )
    assert payload["answer"] == "DNA"
    assert payload["length"] == 3
