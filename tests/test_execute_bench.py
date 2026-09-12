"""Tests for /api/execute and the Benchmark harness."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import FakeLLM  # noqa: E402

from adaptagent.agents import AgentManager  # noqa: E402
from adaptagent.evolution import Benchmark, BenchmarkCase  # noqa: E402
from adaptagent.workflow import WorkflowGraph, WorkflowStep  # noqa: E402
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


WORKFLOW = {
    "goal": "echo goal",
    "steps": [
        {"id": "s1", "name": "A", "instruction": "do a", "inputs": [], "agent_hint": "x"},
        {"id": "s2", "name": "B", "instruction": "do b", "inputs": ["s1"], "agent_hint": "y"},
    ],
}


def test_execute_missing_workflow_and_goal():
    code, data = call_asgi("/api/execute", {})
    assert code == 400
    assert "required" in data["error"]


def test_execute_invalid_workflow_json():
    code, data = call_asgi("/api/execute", {"workflow": {"goal": "g", "steps": [{"bad": True}]}})
    assert code == 400
    assert "invalid workflow" in data["error"]


def test_execute_byok_requires_key_for_non_server_provider():
    code, data = call_asgi("/api/execute", {"workflow": WORKFLOW, "provider": "openai"})
    assert code == 400
    assert "api_key" in data["error"]


# ---------- Benchmark ----------


def test_token_f1_metric():
    bench = Benchmark(cases=[BenchmarkCase(input="q", expected="the quick brown fox")], metric="f1")
    assert METRICS_f1(bench) > 0.5  # partial overlap counts


def METRICS_f1(bench):
    from adaptagent.evolution.benchmark import METRICS

    return METRICS["f1"]("the quick brown fox", "the quick fox")


def test_benchmark_runs_workflow_per_case():
    graph = WorkflowGraph(
        goal="goal",
        steps=[WorkflowStep(id="s1", name="A", instruction="echo", agent_hint="x")],
    )
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["the quick brown fox"]))
    bench = Benchmark(
        cases=[BenchmarkCase(input="q1", expected="the quick brown fox"), BenchmarkCase(input="q2", expected="zzz")],
        metric="em",
    )
    result = asyncio.run(bench.run(graph, manager, FakeLLM(), verbose=False))
    assert result.scores == [1.0, 0.0]
    assert abs(result.mean - 0.5) < 1e-9


def test_benchmark_from_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"input": "q1", "expected": "a1"}\n{"input": "q2", "expected": "a2"}\n', encoding="utf-8")
    bench = Benchmark.from_jsonl(str(path), metric="contains")
    assert len(bench.cases) == 2
    assert bench.cases[1].expected == "a2"
