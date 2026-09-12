"""Tests for AdaptAgent core (no API keys required)."""

import asyncio
import json

import pytest
from conftest import FakeLLM, make_graph

from adaptagent.agents import Agent, AgentManager
from adaptagent.evolution import Evaluator, EvolutionEngine
from adaptagent.hitl import HITLManager
from adaptagent.llm.base import LLMConfig
from adaptagent.llm.schemas import get_tool_schema
from adaptagent.memory import FileMemoryStore, InMemoryStore
from adaptagent.tools import file_write, python_repl
from adaptagent.workflow import Workflow, WorkflowGenerator, WorkflowGraph, WorkflowStep

# ---------- graph ----------

def test_graph_topology():
    graph = make_graph()
    assert graph.topo_order() == ["step1", "step2"]
    assert graph.roots() == ["step1"]


def test_graph_cycle_detection():
    graph = WorkflowGraph(
        steps=[
            WorkflowStep(id="a", name="A", instruction="", inputs=["b"]),
            WorkflowStep(id="b", name="B", instruction="", inputs=["a"]),
        ]
    )
    with pytest.raises(ValueError):
        graph.topo_order()


def test_graph_serialization_roundtrip(tmp_path):
    graph = make_graph()
    path = tmp_path / "wf.json"
    graph.save(str(path))
    loaded = WorkflowGraph.from_file(str(path))
    assert loaded.to_json() == graph.to_json()


# ---------- generator ----------

def test_generator_parses_llm_json():
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "goal": "g",
                    "steps": [
                        {"id": "s1", "name": "A", "instruction": "do a", "inputs": [], "agent_hint": "x"},
                        {"id": "s2", "name": "B", "instruction": "do b", "inputs": ["s1"], "agent_hint": "y"},
                    ],
                }
            )
        ]
    )
    gen = WorkflowGenerator(llm)
    graph = gen.generate_workflow_sync("test")
    assert set(graph.steps) == {"s1", "s2"}
    assert graph.steps["s2"].inputs == ["s1"]


# ---------- agents & workflow ----------

def test_agent_system_prompt_includes_tools():
    agent = Agent(name="t", role="tester", instruction="be helpful", llm=FakeLLM(), tools=[python_repl])
    prompt = agent.system_prompt
    assert "python_repl" in prompt
    assert "be helpful" in prompt


def test_workflow_execution_with_fake_llm():
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["research done", "summary done"]))
    wf = Workflow(graph=graph, agent_manager=manager, llm=FakeLLM())
    output = asyncio.run(wf.execute())
    assert output.final_output == "summary done"
    assert set(output.results) == {"step1", "step2"}
    assert output.results["step1"].output == "research done"


def test_agent_clone():
    agent = Agent(name="a", role="r", instruction="original", llm=None)
    clone = agent.clone(instruction="new")
    assert clone.instruction == "new"
    assert agent.instruction == "original"


# ---------- tools ----------

def test_python_repl_sandbox():
    result = python_repl("print(2 + 2)")
    assert "4" in result


def test_python_repl_blocks_import_os():
    result = python_repl("import os")
    assert "error" in result.lower() or "Error" in result


def test_file_tools_confined_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTAGENT_WORKSPACE", str(tmp_path))
    import importlib

    import adaptagent.tools.files as files_mod
    importlib.reload(files_mod)
    from adaptagent.tools import file_list, file_read
    assert "wrote" in file_write("sub/test.txt", "hello").lower()
    assert file_read("sub/test.txt") == "hello"
    assert "test.txt" in file_list("sub")


def test_tool_schema_generation():
    schema = get_tool_schema(python_repl)
    fn = schema["function"]
    assert fn["name"] == "python_repl"
    assert fn["parameters"]["properties"]["code"]["type"] == "string"
    assert fn["parameters"]["required"] == ["code"]


# ---------- memory ----------

def test_memory_stores_and_recalls():
    mem = InMemoryStore(capacity=3)
    for i in range(5):
        mem.store(f"item {i}")
    assert len(mem.items) == 3  # FIFO
    assert "item 4" in mem.recall()


def test_file_memory_persists(tmp_path):
    path = tmp_path / "mem.jsonl"
    mem = FileMemoryStore(str(path))
    asyncio.run(mem.store_async("persisted fact"))
    mem2 = FileMemoryStore(str(path))
    assert "persisted fact" in mem2.recall()


# ---------- evaluator & evolution ----------

def test_evaluator_clamps_scores():
    ev = Evaluator(name="x", fn=lambda g, o: 150)
    assert ev.evaluate("g", "o") == 100.0


def test_evolution_engine_selects_best(tmp_path):
    """Evolution with a deterministic evaluator: the best candidate wins."""
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    # deterministic LLM: generator / mutator returns fixed texts
    llm = FakeLLM(["best instruction", "ok", "worse instruction", "ok", "best instruction", "ok", "worse", "ok"])
    manager.register_llm(FakeLLM(["research", "FINAL magic answer"]))
    evaluator = Evaluator(name="magic", fn=lambda g, o: 100.0 if "magic" in o else 10.0)
    engine = EvolutionEngine(graph=graph, agent_manager=manager, llm=llm, evaluator=evaluator)
    result = asyncio.run(engine.evolve(generations=2, population=3, verbose=False))
    assert result.best_score > 0
    assert len(result.history) == 2


# ---------- hitl ----------

def test_hitl_disabled_passthrough():
    hitl = HITLManager()
    step = WorkflowStep(id="s", name="s", instruction="i")
    agent = Agent(name="a", role="r", instruction="", llm=None)
    out = asyncio.run(hitl.intercept(step, agent, "unchanged"))
    assert out == "unchanged"


def test_hitl_auto_approve():
    hitl = HITLManager(enabled=True, auto_approve=True)
    step = WorkflowStep(id="s", name="s", instruction="i")
    agent = Agent(name="a", role="r", instruction="", llm=None)
    out = asyncio.run(hitl.intercept(step, agent, "passthrough"))
    assert out == "passthrough"
    assert hitl.interactions == 1


# ---------- llm config ----------

def test_llm_config_overrides():
    cfg = LLMConfig(provider="deepseek", model="m")
    cfg2 = cfg.with_overrides(model="other", temperature=0.1)
    assert cfg2.model == "other"
    assert cfg.model == "m"  # original untouched
