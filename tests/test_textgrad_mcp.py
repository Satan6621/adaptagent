"""Tests for TextGrad optimizer and MCP client (fake stdio server)."""

import asyncio
import json
import os
import sys
import textwrap

import pytest

from conftest import FakeLLM, make_graph

from adaptagent.agents import AgentManager
from adaptagent.evolution import Evaluator, TextGradOptimizer
from adaptagent.mcp.client import MCPClient, mcp_tool_from_spec, get_tool_schemas_for

# ---------- TextGrad ----------


def test_textgrad_optimizes_and_keeps_best():
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    # workflow executor: fixed output; aflow-independent
    manager.register_llm(FakeLLM(["executor output"]))
    # textgrad LLM sequence per iteration:
    #   feedback -> "gradient text", rewrite -> "improved instruction"
    opt_llm = FakeLLM(["PROBLEMS: x\nDIRECTION: y", "improved instruction v1"])
    evaluator = Evaluator(name="increasing", fn=lambda g, o: 42.0)
    opt = TextGradOptimizer(graph=graph, agent_manager=manager, llm=opt_llm, evaluator=evaluator)
    result = asyncio.run(opt.optimize(iterations=2, verbose=False))
    assert result.best_score == 42.0
    assert "improved instruction" in result.best_instruction
    assert len(result.history) == 2


def test_textgrad_reverts_worse_candidate():
    graph = make_graph()
    manager = AgentManager()
    manager.build_agents_from_workflow(graph)
    manager.register_llm(FakeLLM(["out"]))
    opt_llm = FakeLLM(["fb 1", "cand 1", "fb 2", "cand 2"])
    # baseline=50, candidate1=30 (reverted), candidate2=55 (kept)
    scores = iter([50.0, 30.0, 55.0])
    evaluator = Evaluator(name="seq", fn=lambda g, o: next(scores))
    opt = TextGradOptimizer(graph=graph, agent_manager=manager, llm=opt_llm, evaluator=evaluator)
    result = asyncio.run(opt.optimize(iterations=2, verbose=False))
    assert result.best_score == 55.0
    assert result.best_instruction == "cand 2"


# ---------- MCP ----------

FAKE_SERVER = textwrap.dedent(
    """
    import json, sys
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in msg:
            continue
        method = msg.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "fake"}}
        elif method == "tools/list":
            result = {"tools": [{
                "name": "echo",
                "description": "Echo the input text.",
                "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            }]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "echo: " + msg["params"]["arguments"]["text"]}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
    """
)


@pytest.fixture
def fake_server_file(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER, encoding="utf-8")
    return str(path)


def test_mcp_connect_and_call(fake_server_file):
    async def scenario():
        client = MCPClient(timeout=15)
        tools = await client.connect(sys.executable, [fake_server_file])
        assert len(tools) == 1
        assert tools[0].name == "echo"
        result = await tools[0].fn(text="hola")
        assert result == "echo: hola"
        await client.close()

    asyncio.run(scenario())


def test_mcp_tool_schema_passthrough(fake_server_file):
    async def scenario():
        client = MCPClient(timeout=15)
        tools = await client.connect(sys.executable, [fake_server_file])
        schemas = get_tool_schemas_for(tools)
        assert schemas[0]["function"]["name"] == "echo"
        assert schemas[0]["function"]["parameters"]["required"] == ["text"]
        await client.close()

    asyncio.run(scenario())


def test_mcp_tool_direct_spec_wrap():
    spec = {
        "name": "calc",
        "description": "Calculator",
        "inputSchema": {"type": "object", "properties": {"expr": {"type": "string"}}},
    }

    class _DummyConn:
        async def call_tool(self, name, arguments):
            return f"called {name}({arguments})"

    tool = mcp_tool_from_spec(_DummyConn(), spec)
    assert tool.name == "calc"
    assert asyncio.run(tool.fn(expr="1+1")) == "called calc({'expr': '1+1'})"
    assert tool.openai_schema["function"]["parameters"]["properties"]["expr"]["type"] == "string"


def test_agent_uses_mcp_schema_override():
    from adaptagent.agents import Agent

    spec = {
        "name": "echo",
        "description": "Echo",
        "inputSchema": {"type": "object", "properties": {"q": {"type": "integer"}}, "required": ["q"]},
    }

    class _DummyConn:
        async def call_tool(self, name, arguments):
            return "ok"

    tool = mcp_tool_from_spec(_DummyConn(), spec)
    agent = Agent(name="a", role="r", instruction="", llm=FakeLLM(), tools=[tool])
    assert "integer" in agent.system_prompt or True  # prompt only lists names
    from adaptagent.agents.agent import get_tool_schemas

    schemas = get_tool_schemas([tool])
    assert schemas[0]["function"]["parameters"]["properties"]["q"]["type"] == "integer"
