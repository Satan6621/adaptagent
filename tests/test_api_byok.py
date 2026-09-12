"""Tests for the ASGI BYOK handler (api/asgi.py)."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.asgi import app  # noqa: E402


def call_asgi(path: str, body: dict) -> tuple[int, dict]:
    status = {}
    received = {"body": json.dumps(body).encode(), "more_body": False}

    async def receive():
        return received

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            status["body"] = message["body"].decode()

    asyncio.run(app({"type": "http", "path": path}, receive, send))
    return status["code"], json.loads(status["body"])


def test_missing_goal_returns_400():
    code, data = call_asgi("/api", {})
    assert code == 400
    assert "goal" in data["error"]


def test_byok_requires_key_for_non_server_provider():
    code, data = call_asgi("/api", {"goal": "x", "provider": "openai"})
    assert code == 400
    assert "api_key" in data["error"]


def test_models_endpoint_lists_openrouter():
    code, data = call_asgi("/api/models", {"provider": "openrouter"})
    assert code == 200
    assert any("openai/" in m for m in data["models"])


def test_byok_invalid_key_gives_provider_500_error():
    # reaches the provider and fails there -> 500 with the provider error text
    code, data = call_asgi("/api", {"goal": "x", "provider": "gemini", "api_key": "invalid", "model": "gemini-2.5-flash"})
    assert code == 500
    assert "error" in data
