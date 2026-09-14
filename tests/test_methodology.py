"""Tests for the Superpowers-inspired methodology: contracts, skills kit, templates."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import FakeLLM  # noqa: E402

from adaptagent.agents import AgentManager  # noqa: E402
from adaptagent.skills import (  # noqa: E402
    SUPERPOWERS_SKILLS,
    apply_methodology,
    apply_skills,
)
from adaptagent.workflow import WorkflowGraph  # noqa: E402
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


WF = {
    "goal": "g",
    "steps": [
        {"id": "s1", "name": "Planificador", "instruction": "Plantea el plan.", "inputs": [], "agent_hint": "analyst"},
        {"id": "s2", "name": "Coder", "instruction": "Implementa.", "inputs": ["s1"], "agent_hint": "coder"},
    ],
}


def test_methodology_contract_injected_into_every_agent(monkeypatch):
    llm = FakeLLM(["f1", "f2"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    code, data = call_api("/api/execute", {"workflow": WF, "provider": "gemini", "model": "gemini-2.5-flash", "methodology": "superpowers"})
    assert code == 200
    assert data["meta"]["methodology"] == ["superpowers"]
    system_prompts = [c["messages"][0]["content"] for c in llm.calls]
    assert len(system_prompts) == 2
    for sp in system_prompts:
        assert "METODOLOGÍA ACTIVA" in sp
        assert "TDD" in sp or "test" in sp.lower()


def test_methodology_unknown_is_noop(monkeypatch):
    llm = FakeLLM(["f1"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    code, data = call_api("/api/execute", {"workflow": WF, "provider": "gemini", "model": "gemini-2.5-flash", "methodology": "bogus"})
    assert code == 200
    assert "methodology" not in data["meta"]
    assert "METODOLOGÍA ACTIVA" not in llm.calls[0]["messages"][0]["content"]


def test_apply_methodology_directly():
    manager = AgentManager()
    manager.build_agents_from_workflow(WorkflowGraph.from_json(json.dumps(WF)))
    applied = apply_methodology(manager, "superpowers")
    assert applied == ["superpowers"]
    assert all("METODOLOGÍA ACTIVA" in a.instruction for a in manager.agents.values())


def test_methodology_endpoint_serves_methods_and_kit():
    code, data = call_api("/api/methodology", {})
    assert code == 200
    assert "superpowers" in data["methods"]
    assert len(data["skills"]) >= 5
    for s in data["skills"]:
        assert s["name"] and s["text"]
        assert isinstance(s.get("apply"), list)


def test_superpowers_kit_items_apply():
    manager = AgentManager()
    manager.build_agents_from_workflow(WorkflowGraph.from_json(json.dumps(WF)))
    summary = apply_skills(manager, SUPERPOWERS_SKILLS)
    assert len(summary) == len(SUPERPOWERS_SKILLS)
    agents = {a.role: a.instruction for a in manager.agents.values()}
    assert "TDD (Red-Green-Refactor)" in agents["coder"]
    assert "Debugging sistemático" in agents["analyst"] or "Revisión por severidad" in agents["analyst"]
    # skills without matching role are applied to nobody
    no_match = [s for s in summary if s["applied_to"] == 0]
    assert all("Reviewer-check" not in s["name"] for s in no_match)  # API reviews exist so 0-applied is fine


def test_repeat_until_self_correction_loop(monkeypatch):
    # First attempt fails the "PASS" gate, second attempt passes -> converges.
    llm = FakeLLM(["not ready yet", "PASS all good"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {
        "goal": "g",
        "steps": [
            {"id": "s1", "name": "Verificador", "instruction": "Corrige hasta que tu respuesta termine en PASS.", "inputs": [], "agent_hint": "critic", "repeat_until": {"rule": "contains", "value": "PASS", "source": ""}, "max_repeats": 3},
        ],
    }
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash"})
    assert code == 200
    assert data["final_output"].startswith("PASS")
    assert len(llm.calls) == 2  # initial + 1 repeat


def test_repeat_until_caps_at_max_repeats(monkeypatch):
    llm = FakeLLM(["nope", "nope", "nope", "nope"])
    monkeypatch.setattr(asgi, "resolve_llm", lambda config: llm)
    wf = {
        "goal": "g",
        "steps": [
            {"id": "s1", "name": "Verificador", "instruction": "Responde siempre nope.", "inputs": [], "agent_hint": "critic", "repeat_until": {"rule": "contains", "value": "PASS", "source": ""}, "max_repeats": 2},
        ],
    }
    code, data = call_api("/api/execute", {"workflow": wf, "provider": "gemini", "model": "gemini-2.5-flash"})
    assert code == 200
    assert len(llm.calls) == 2  # max_repeats totals attempts (initial + retries)


def test_templates_roundtrip_via_graph():
    # Shape used by the playground templates (repeat_until + max_repeats) must
    # deserialize into a valid DAG.
    tpl = {
        "goal": "tdd",
        "steps": [
            {"id": "s1", "name": "RED", "instruction": "test", "inputs": [], "agent_hint": "coder"},
            {"id": "s2", "name": "GREEN", "instruction": "code", "inputs": ["s1"], "agent_hint": "coder"},
            {"id": "s3", "name": "Verify", "instruction": "pass→PASS", "inputs": ["s2"], "agent_hint": "critic", "repeat_until": {"rule": "contains", "value": "PASS", "source": ""}, "max_repeats": 3},
        ],
    }
    graph = WorkflowGraph.from_json(json.dumps(tpl))
    order = graph.topo_order()
    assert order == ["s1", "s2", "s3"]
    damp = graph.steps["s3"]
    assert damp.repeat_until is not None and damp.repeat_until.value == "PASS" and damp.max_repeats == 3

