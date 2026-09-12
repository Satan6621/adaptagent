"""Tests for executor events, Anthropic role mapping, and SSE streaming endpoint."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import FakeLLM, make_graph  # noqa: E402

from adaptagent.agents import AgentManager  # noqa: E402
from adaptagent.workflow import Workflow  # noqa: E402
from api.asgi import app  # noqa: E402

# ---------- executor events ----------


def test_executor_emits_events_in_order():
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["a out", "b out"]))
    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM(), on_event=on_event)
    asyncio.run(wf.execute())
    kinds = [e["event"] for e in events]
    assert kinds == ["step_start", "step_done", "step_start", "step_done", "done"]
    assert events[-1]["final_output"] == "b out"
    assert events[-1]["elapsed"] >= 0


def test_executor_sync_callback_supported():
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["a", "b"]))
    events: list[dict] = []
    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM(), on_event=events.append)
    asyncio.run(wf.execute())
    assert len(events) == 5


# ---------- Anthropic role mapping (bug fix regression) ----------


def test_anthropic_role_mapping_preserves_assistant():
    from adaptagent.llm.base import LLMConfig
    from adaptagent.llm.openai_compat import AnthropicLLM

    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="test"))
    captured: dict = {}

    async def fake_post(path, json_payload=None):
        captured["payload"] = json_payload
        return fake_post

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}], "usage": {}}

    class _Client:
        async def post(self, path, json=None):
            captured["messages"] = json["messages"]
            return _Resp()

    llm.client = _Client()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q2"},
    ]
    asyncio.run(llm.generate(messages))
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user", "assistant", "user"]


# ---------- SSE streaming endpoint ----------


def call_sse(body: dict) -> tuple[int, list[dict]]:
    status = {}
    events: list[dict] = []
    received = {"body": json.dumps(body).encode(), "more_body": False}

    async def receive():
        return received

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"").decode()
            for line in chunk.splitlines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

    asyncio.run(app({"type": "http", "path": "/api/execute/stream"}, receive, send))
    return status["code"], events


WORKFLOW = {
    "goal": "echo",
    "steps": [{"id": "s1", "name": "A", "instruction": "say hi", "inputs": [], "agent_hint": "writer"}],
}


def test_sse_stream_missing_workflow_errors():
    code, events = call_sse({})
    assert code == 200
    assert events[0]["event"] == "error"


def test_sse_stream_invalid_workflow_errors():
    code, events = call_sse({"workflow": {"goal": "g", "steps": [{"bad": 1}]}})
    assert events[0]["event"] == "error"
    assert "invalid workflow" in events[0]["error"] or "error" in events[0]["error"]


def test_sse_stream_byok_missing_key_errors():
    code, events = call_sse({"workflow": WORKFLOW, "provider": "openai"})
    assert events[0]["event"] == "error"
    assert "api_key" in events[0]["error"]
