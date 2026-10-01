"""Anthropic bridge retry with exponential backoff on 429/5xx."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_http import FakeHTTPServer  # noqa: E402

from adaptagent.config import _load_env_file  # noqa: E402
from adaptagent.llm.base import LLMConfig  # noqa: E402
from adaptagent.llm.openai_compat import AnthropicLLM  # noqa: E402


def _claude_ok(text: str = "hello from claude") -> bytes:
    return json.dumps(
        {
            "id": "msg_1",
            "model": "claude-x",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    ).encode()


def test_anthropic_retries_on_429_and_5xx():
    calls = {"n": 0}

    def router(method, path, headers, body):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, "application/json", b'{"error":{"type":"rate_limit_error","message":"slow down"}}'
        if calls["n"] == 2:
            return 500, "application/json", b'{"error":"boom"}'
        return 200, "application/json", _claude_ok()

    with FakeHTTPServer(router) as srv:
        llm = AnthropicLLM(
            LLMConfig(provider="anthropic", model="claude-x", api_key="k",
                      base_url=srv.base_url, timeout=5)
        )
        resp = asyncio.run(llm.generate([{"role": "user", "content": "hi"}]))
        assert resp.text == "hello from claude"
        assert calls["n"] == 3  # 429 -> 500 -> 200


def test_anthropic_single_call_when_ok():
    calls = {"n": 0}

    def router(method, path, headers, body):  # noqa: ANN001
        calls["n"] += 1
        return 200, "application/json", _claude_ok()

    with FakeHTTPServer(router) as srv:
        llm = AnthropicLLM(
            LLMConfig(provider="anthropic", model="claude-x", api_key="k",
                      base_url=srv.base_url, timeout=5)
        )
        resp = asyncio.run(llm.generate([{"role": "user", "content": "hi"}]))
        assert resp.text == "hello from claude"
        assert calls["n"] == 1