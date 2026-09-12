"""Tests for AFlow structural evolution and the CLI."""

import asyncio
import json

import pytest
from conftest import FakeLLM, make_graph

from adaptagent.agents import AgentManager
from adaptagent.evolution import AFlowEngine, Evaluator

# ---------- AFlow ----------

def test_aflow_add_critic_fallback():
    """LLM gives invalid JSON -> deterministic fallback op mutates the graph."""
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    # executor LLM: each workflow step returns this text
    manager.register_llm(FakeLLM(["step output containing MAGIC"]))
    # aflow LLM: invalid proposal -> fallback; judge not used (custom evaluator)
    aflow_llm = FakeLLM(["not json at all"])
    evaluator = Evaluator(name="magic", fn=lambda g, o: 100.0 if "MAGIC" in o else 10.0)
    engine = AFlowEngine(graph=graph, agent_manager=manager, llm=aflow_llm, evaluator=evaluator)
    result = asyncio.run(engine.evolve(generations=2, verbose=False))
    assert result.best_score == 100.0
    # baseline eval + 2 mutation evals ran
    assert len(result.history) <= 2


def test_aflow_llm_add_step_proposal():
    """LLM proposes a valid 'add' mutation -> new step appears."""
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["plain output"]))

    proposal = json.dumps(
        {
            "type": "add",
            "step": {
                "id": "step3",
                "name": "Critic",
                "instruction": "Critique and improve",
                "inputs": ["step2"],
                "agent_hint": "critic",
            },
        }
    )
    # sequence: baseline output ("score text"), then mutation proposal, then evals
    aflow_llm = FakeLLM([proposal])
    evaluator = Evaluator(name="always", fn=lambda g, o: 42.0)
    engine = AFlowEngine(graph=graph, agent_manager=manager, llm=aflow_llm, evaluator=evaluator)
    result = asyncio.run(engine.evolve(generations=1, verbose=False))
    assert "step3" in result.best_graph.steps


def test_aflow_remove_step_proposal():
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["out"]))
    proposal = json.dumps({"type": "remove", "id": "step1"})
    aflow_llm = FakeLLM([proposal])
    evaluator = Evaluator(name="always", fn=lambda g, o: 50.0)
    engine = AFlowEngine(graph=graph, agent_manager=manager, llm=aflow_llm, evaluator=evaluator)
    result = asyncio.run(engine.evolve(generations=1, verbose=False))
    assert "step1" not in result.best_graph.steps
    # step2 no longer references the removed step
    assert result.best_graph.steps["step2"].inputs == []


def test_aflow_rejects_cyclic_reorder():
    graph = make_graph()
    engine = AFlowEngine(graph=graph, agent_manager=AgentManager(), llm=FakeLLM(), evaluator=Evaluator("x", lambda *a: 1))
    proposal = {"type": "reorder", "id": "step1", "new_inputs": ["step2"]}
    mutated = engine._apply_proposal(graph, proposal)
    assert mutated is not None
    assert mutated.steps["step1"].inputs == []  # cycle defused


# ---------- CLI ----------

def test_cli_version(capsys):
    from adaptagent.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "adaptagent" in capsys.readouterr().out


def test_cli_show(tmp_path, capsys):
    from adaptagent.cli import main

    path = tmp_path / "wf.json"
    make_graph().save(str(path))
    code = main(["show", str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Research" in out
    assert "Write" in out


def test_cli_resolve_tools():
    from adaptagent.cli import _resolve_tools

    tools = _resolve_tools(["wiki", "python"])
    assert {t.name for t in tools} == {"wikipedia_search", "python_repl"}
    assert _resolve_tools(["bogus"]) == []


def test_cli_run_with_fake_llm(monkeypatch, capsys):
    """End-to-end CLI run with the LLM factory patched."""
    from adaptagent import cli

    goal_json = json.dumps(
        {
            "goal": "g",
            "steps": [
                {"id": "s1", "name": "A", "instruction": "do a", "inputs": [], "agent_hint": "x"},
                {"id": "s2", "name": "B", "instruction": "do b", "inputs": ["s1"], "agent_hint": "y"},
            ],
        }
    )
    # generator call -> goal_json; step executions -> simple texts
    fake = FakeLLM([goal_json, "result a", "result b"])
    monkeypatch.setattr(cli, "_build_llm", lambda: fake)
    code = cli.main(["run", "test goal"])
    out = capsys.readouterr().out
    assert code == 0
    assert "FINAL OUTPUT" in out
    assert "result b" in out
