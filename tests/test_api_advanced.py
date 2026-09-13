"""API tests: /api/mcp/test endpoint + advanced payload passthrough (tools/skills)."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import FakeLLM  # noqa: E402
from fake_http import FakeHTTPServer, echo_router, mcp_router  # noqa: E402

from api import asgi  # noqa: E402
from api.asgi import app  # noqa: E402


def call_api(path: str, body: dict) -> tuple[int, dict]:
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


def test_mcp_test_endpoint_requires_server():
    code, data = call_api("/api/mcp/test", {})
    assert code == 400


def test_mcp_test_endpoint_bad_server():
    code, data = call_api("/api/mcp/test", {"server": {"type": "http", "url": "http://127.0.0.1:1/mcp"}})
    assert code == 200
    assert data["ok"] is False
    assert data["error"]


def test_mcp_test_endpoint_good_server():
    with FakeHTTPServer(mcp_router) as server:
        code, data = call_api("/api/mcp/test", {"server": {"type": "http", "name": "fake", "url": server.base_url + "/mcp"}})
    assert code == 200
    assert data["ok"] is True
    assert data["count"] == 1
    assert "echo" in data["tools"]


def test_execute_with_custom_tools_and_skills(monkeypatch):
    llm = FakeLLM(["final answer"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {"goal": "g", "steps": [{"id": "s1", "name": "A", "instruction": "i", "inputs": [], "agent_hint": "writer"}]}
    with FakeHTTPServer(echo_router) as server:
        body = {
            "workflow": wf,
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "custom_tools": [{"name": "q", "description": "query", "method": "GET", "url": server.base_url + "/we?city={city}"}],
            "skills": [{"name": "S", "text": "Sé conciso.", "apply": ["writer"]}],
        }
        code, data = call_api("/api/execute", body)
    assert code == 200
    assert data["final_output"] == "final answer"
    assert data["meta"]["http_tools"] == 1
    assert data["meta"]["mcp_tools"] == 0
    assert data["meta"]["skills"][0]["applied_to"] == 1
    # the writer agent should have received the skill text in its system prompt
    call = llm.calls[0]
    assert "Sé conciso." in call["messages"][0]["content"]
