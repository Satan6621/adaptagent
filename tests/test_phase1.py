"""Phase 1 tests: structured outputs, judge loops, termination, checkpoint/resume."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from conftest import FakeLLM  # noqa: E402

from adaptagent.checkpoint import Checkpoint, CheckpointStore, reset_store  # noqa: E402
from adaptagent.structured import extract_json, validate_step_output  # noqa: E402
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


@pytest.fixture(autouse=True)
def fresh_store():
    reset_store()


def _llm(monkeypatch, responses):
    llm = FakeLLM(responses)
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    return llm


# --- structured outputs ---

def test_extract_json_fenced_and_raw():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json("prefix\n{'b': 2}\n") is None  # single quotes not JSON
    assert extract_json('prefix {"c": 3} suffix') == {"c": 3}


def test_validate_schema_required_and_types():
    schema = {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}, "count": {"type": "integer"}}}
    parsed, errors = validate_step_output(schema, '{"title": "ok", "count": 3}')
    assert errors == [] and parsed["count"] == 3
    _, errors = validate_step_output(schema, '{"count": "x"}')
    assert any("title" in e for e in errors)
    _, errors = validate_step_output(schema, '{"title": "ok", "count": 1.5}')
    assert any("integer" in e for e in errors)


def test_validate_additional_properties():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}
    _, errors = validate_step_output(schema, '{"a": "x", "b": 1}')
    assert any("no permitido" in e for e in errors)


def test_output_schema_retries_until_valid(monkeypatch):
    llm = FakeLLM(["not json at all", '```json\n{"title": "final"}\n```'])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {
        "goal": "g",
        "steps": [
            {"id": "s1", "name": "Writer", "instruction": "Devuelve un JSON.", "inputs": [], "agent_hint": "writer",
             "output_schema": {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}},
             "max_repeats": 3},
        ],
    }
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash"})
    assert code == 200
    assert data["final_output"].strip() == '```json\n{"title": "final"}\n```'
    assert len(llm.calls) == 2  # invalid attempt + valid retry


def test_output_schema_gives_up_after_cap(monkeypatch):
    llm = FakeLLM(["junk", "junk2"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {
        "goal": "g",
        "steps": [
            {"id": "s1", "name": "Writer", "instruction": "JSON.", "inputs": [], "agent_hint": "writer",
             "output_schema": {"type": "object"}, "max_repeats": 2},
        ],
    }
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash"})
    assert code == 200
    assert len(llm.calls) == 2  # max_repeats caps total attempts (initial + retries)


def test_structured_metadata_in_result(monkeypatch):
    from adaptagent.workflow import WorkflowGraph  # noqa: E402
    from adaptagent.workflow.executor import Workflow  # noqa: E402

    llm = FakeLLM(['{"n": 7}'])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    from adaptagent import AgentManager  # noqa: E402

    wf = {
        "goal": "g",
        "steps": [{"id": "s1", "name": "W", "instruction": "json", "inputs": [], "agent_hint": "writer",
                   "output_schema": {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}}],
    }
    graph = WorkflowGraph.from_json(json.dumps(wf))
    manager = AgentManager()
    manager.register_llm(llm)
    manager.build_agents_from_workflow(graph)
    output = asyncio.run(Workflow(graph=graph, agent_manager=manager, llm=llm, max_parallel=1).execute())
    assert output["s1"].metadata["json"] == {"n": 7}


# --- judge-backed loops ---

def test_score_gte_loop_uses_judge(monkeypatch):
    # order: step attempt, judge score 40, step attempt, judge score 95 -> converge
    llm = FakeLLM(["attempt one", "SCORE: 40 | meh", "attempt two", "SCORE: 95 | good"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {
        "goal": "g",
        "steps": [
            {"id": "s1", "name": "Coder", "instruction": "Mejora la solución.", "inputs": [], "agent_hint": "coder",
             "repeat_until": {"rule": "score_gte", "value": "80", "source": ""}, "max_repeats": 3},
        ],
    }
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash"})
    assert code == 200
    assert data["final_output"] == "attempt two"
    assert len(llm.calls) == 4  # 2 attempts + 2 judge scores


# --- global termination ---

def test_stop_when_sentinel(monkeypatch):
    llm = FakeLLM(["first ok", "PARAR aquí", "third never"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {"goal": "g", "steps": [
        {"id": "s1", "name": "A", "instruction": "s1", "inputs": [], "agent_hint": "writer"},
        {"id": "s2", "name": "B", "instruction": "s2", "inputs": ["s1"], "agent_hint": "writer"},
        {"id": "s3", "name": "C", "instruction": "s3", "inputs": ["s2"], "agent_hint": "writer"},
    ]}
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash", "stop_when": {"contains": "PARAR"}})
    assert code == 200
    assert data["stopped"] and "PARAR" in data["stopped"]
    assert len(llm.calls) == 2
    assert data["steps"][2]["output"] == ""  # s3 skipped


def test_stop_when_max_steps(monkeypatch):
    llm = FakeLLM(["one"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {"goal": "g", "steps": [
        {"id": "s1", "name": "A", "instruction": "s1", "inputs": [], "agent_hint": "writer"},
        {"id": "s2", "name": "B", "instruction": "s2", "inputs": ["s1"], "agent_hint": "writer"},
    ]}
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash", "stop_when": {"max_steps": 1}})
    assert code == 200
    assert data["stopped"] == "max_steps (1) reached"
    assert len(llm.calls) == 1


# --- checkpointing / resume ---

def test_checkpoint_inline_resume(monkeypatch):
    llm = FakeLLM(["two", "three"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {"goal": "g", "steps": [
        {"id": "s1", "name": "A", "instruction": "s1", "inputs": [], "agent_hint": "writer"},
        {"id": "s2", "name": "B", "instruction": "s2", "inputs": ["s1"], "agent_hint": "writer"},
        {"id": "s3", "name": "C", "instruction": "s3", "inputs": ["s2"], "agent_hint": "writer"},
    ]}
    cp = {"run_id": "r1", "goal": "g", "step_outputs": {"s1": "one"}, "skipped": [], "created_at": ""}
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash", "checkpoint": cp})
    assert code == 200
    assert len(llm.calls) == 2  # s1 resumed without an LLM call
    by_id = {s["id"]: s for s in data["steps"]}
    assert by_id["s1"]["output"] == "one"
    assert by_id["s2"]["output"] == "two"
    assert by_id["s3"]["output"] == "three"
    assert data["run_id"] == "r1"


def test_checkpoint_store_roundtrip(tmp_path):
    store = CheckpointStore(runs_dir=str(tmp_path))
    cp = Checkpoint(run_id="abc", goal="g", step_outputs={"s1": "hello"})
    store.save(cp)
    store2 = CheckpointStore(runs_dir=str(tmp_path))
    assert store2.get("abc").step_outputs == {"s1": "hello"}
    assert any(h["run_id"] == "abc" for h in store2.history())


def test_checkpoints_endpoint_lists_runs(monkeypatch):
    _llm(monkeypatch, ["out"])
    wf = {"goal": "g", "steps": [{"id": "s1", "name": "A", "instruction": "s1", "inputs": [], "agent_hint": "writer"}]}
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash"})
    assert code == 200
    run_id = data["run_id"]
    assert data["checkpoint"]["step_outputs"]["s1"] == "out"
    code, list_data = call_api("/api/checkpoints", {})
    assert code == 200
    assert any(h["run_id"] == run_id for h in list_data["history"])
    code, run_data = call_api(f"/api/checkpoints/{run_id}", {})
    assert code == 200
    assert run_data["step_outputs"]["s1"] == "out"
