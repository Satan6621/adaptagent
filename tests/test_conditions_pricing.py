"""Tests for conditionals, repeat_until, pricing, usage tracking, /api/step."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import FakeLLM  # noqa: E402

from adaptagent.agents import AgentManager  # noqa: E402
from adaptagent.llm.pricing import estimate_cost, parse_usage  # noqa: E402
from adaptagent.workflow import Conditional, Workflow, WorkflowGraph, WorkflowStep  # noqa: E402
from api.asgi import app  # noqa: E402

# ---------- conditionals ----------


def test_conditional_rules():
    assert Conditional(source="s1", rule="contains", value="ok").evaluate("everything ok here")
    assert not Conditional(source="s1", rule="contains", value="ok").evaluate("nope")
    assert Conditional(source="s1", rule="not_contains", value="bad").evaluate("good")
    assert not Conditional(source="s1", rule="nonempty", value="").evaluate("   ")
    assert Conditional(source="s1", rule="regex", value=r"\d{3}").evaluate("code 404 found")
    assert not Conditional(source="s1", rule="regex", value=r"\d{3}").evaluate("no digits")
    assert Conditional(source="s1", rule="unknown_rule", value="x").evaluate("anything")  # safe default


def test_conditional_serialization_roundtrip():
    step = WorkflowStep(
        id="s2", name="B", instruction="i", inputs=["s1"],
        condition=Conditional(source="s1", rule="contains", value="go"),
    )
    graph = WorkflowGraph(goal="g", steps=[WorkflowStep(id="s1", name="A", instruction="i"), step])
    data = json.loads(graph.to_json())
    loaded = WorkflowGraph.from_json(json.dumps(data))
    assert loaded.steps["s2"].condition.rule == "contains"
    assert loaded.steps["s2"].condition.value == "go"


def test_executor_skips_failed_condition():
    graph = WorkflowGraph(
        goal="g",
        steps=[
            WorkflowStep(id="s1", name="A", instruction="i", agent_hint="x"),
            WorkflowStep(
                id="s2", name="B", instruction="i", inputs=["s1"], agent_hint="y",
                condition=Conditional(source="s1", rule="contains", value="MAGIC"),
            ),
        ],
    )
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["plain text, nothing relevant"]))
    events: list[dict] = []
    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM(), on_event=events.append)
    output = asyncio.run(wf.execute())
    assert output.results["s2"].skipped is True
    assert output.final_output == "plain text, nothing relevant"  # last non-skipped
    kinds = [e["event"] for e in events]
    assert "step_skipped" in kinds


def test_executor_runs_when_condition_passes():
    graph = WorkflowGraph(
        goal="g",
        steps=[
            WorkflowStep(id="s1", name="A", instruction="i", agent_hint="x"),
            WorkflowStep(
                id="s2", name="B", instruction="i", inputs=["s1"], agent_hint="y",
                condition=Conditional(source="s1", rule="contains", value="MAGIC"),
            ),
        ],
    )
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["text WITH MAGIC word", "second step output"]))
    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM())
    output = asyncio.run(wf.execute())
    assert output.results["s2"].skipped is False
    assert output.final_output == "second step output"


def test_repeat_until_loops_until_pass():
    graph = WorkflowGraph(
        goal="g",
        steps=[
            WorkflowStep(
                id="s1", name="A", instruction="i", agent_hint="x",
                repeat_until=Conditional(source="s1", rule="contains", value="DONE"),
                max_repeats=3,
            ),
        ],
    )
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    # agent returns "no" twice then "final DONE marker"
    manager.register_llm(FakeLLM(["attempt 1", "attempt 2", "attempt with DONE"]))
    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM())
    output = asyncio.run(wf.execute())
    assert "DONE" in output.final_output
    assert output.results["s1"].metadata.get("repeats") == 2


def test_repeat_until_respects_cap():
    graph = WorkflowGraph(
        goal="g",
        steps=[
            WorkflowStep(
                id="s1", name="A", instruction="i", agent_hint="x",
                repeat_until=Conditional(source="s1", rule="contains", value="NEVER_APPEARS"),
                max_repeats=2,
            ),
        ],
    )
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["always the same"]))
    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM())
    output = asyncio.run(wf.execute())
    assert output.results["s1"].metadata.get("repeats") == 2  # capped


# ---------- pricing ----------


def test_estimate_cost_known_model():
    # gpt-4o-mini: 0.15 in / 0.60 out per 1M
    cost = estimate_cost("openai/gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-9


def test_estimate_cost_prefix_match():
    cost = estimate_cost("gemini-2.0-flash-001", 1_000_000, 0)
    assert abs(cost - 0.10) < 1e-9


def test_estimate_cost_unknown_model_free():
    assert estimate_cost("mystery/model", 100, 100) == 0.0


def test_parse_usage_openai_and_anthropic():
    assert parse_usage({"prompt_tokens": 10, "completion_tokens": 5}) == (10, 5)
    assert parse_usage({"input_tokens": 7, "output_tokens": 3}, "anthropic") == (7, 3)
    assert parse_usage({}) == (0, 0)


def test_agent_tracks_usage():
    from adaptagent.agents import Agent
    from adaptagent.llm.base import LLMResponse

    llm = FakeLLM()
    llm.config.model = "openai/gpt-4o-mini"

    async def gen(messages, tools=None, **kw):
        return LLMResponse(text="hi", usage={"prompt_tokens": 100, "completion_tokens": 50})

    llm.generate = gen
    agent = Agent(name="a", role="r", instruction="", llm=llm)
    asyncio.run(agent.arun("q"))
    assert agent.last_usage == {"prompt_tokens": 100, "completion_tokens": 50}
    assert abs(agent.last_cost_usd - (100 * 0.15 + 50 * 0.60) / 1_000_000) < 1e-9


# ---------- /api/step endpoint ----------


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


WORKFLOW = {
    "goal": "echo",
    "steps": [{"id": "s1", "name": "A", "instruction": "say hi", "inputs": [], "agent_hint": "writer"}],
}


def test_step_endpoint_validation():
    code, data = call_api("/api/step", {})
    assert code == 400
    assert "required" in data["error"]
    code, data = call_api("/api/step", {"workflow": WORKFLOW, "step_id": "zzz"})
    assert code == 400
    assert "unknown step_id" in data["error"]


def test_step_endpoint_skips_condition():
    wf = {
        "goal": "g",
        "steps": [
            {"id": "s1", "name": "A", "instruction": "i", "inputs": []},
            {"id": "s2", "name": "B", "instruction": "i", "inputs": ["s1"],
             "condition": {"source": "s1", "rule": "contains", "value": "MAGIC"}},
        ],
    }
    code, data = call_api("/api/step", {"workflow": wf, "step_id": "s2", "step_outputs": {"s1": "nothing here"}})
    assert code == 200
    assert data["skipped"] is True
