"""AdaptAgent CLI: run workflows, ask single agents, inspect graphs.

Usage:
  adaptagent run "mi objetivo"          # generar + ejecutar workflow
  adaptagent run --tools wiki,search    # ... con tools
  adaptagent ask "pregunta"             # un solo agente ReAct
  adaptagent show workflow.json         # visualizar grafo guardado
  adaptagent evolve "objetivo" -g 3     # generar + evolucionar prompts
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

from adaptagent import __version__


def _build_llm():
    from adaptagent import LLMConfig, resolve_llm

    return resolve_llm(LLMConfig())


def _resolve_tools(names: list[str] | None):
    from adaptagent.tools import (
        DDGSSearchTool,
        FileReadTool,
        FileWriteTool,
        HTTPRequestTool,
        PythonREPLTool,
        WikipediaSearchTool,
    )

    catalog = {
        "wiki": WikipediaSearchTool,
        "wikipedia": WikipediaSearchTool,
        "search": DDGSSearchTool,
        "ddgs": DDGSSearchTool,
        "python": PythonREPLTool,
        "repl": PythonREPLTool,
        "http": HTTPRequestTool,
        "read": FileReadTool,
        "write": FileWriteTool,
    }
    return [catalog[n]() for n in (names or []) if n in catalog]


async def _connect_mcp(mcp_specs: list[str]) -> "tuple[MCPClient, list]":
    """Parse --mcp specs like 'python:server.py' or 'npx:-y:@modelcontextprotocol/server-everything'."""
    from adaptagent.mcp import MCPClient

    client = MCPClient()
    tools: list = []
    for spec in mcp_specs:
        parts = spec.split(":", 1)
        command, args = parts[0], parts[1].split(":") if len(parts) > 1 else []
        try:
            remote = await client.connect(command, args)
            tools.extend(remote)
            print(f"[mcp] {command} {' '.join(args)}: {len(remote)} tools")
        except Exception as exc:  # noqa: BLE001
            print(f"[mcp] failed to connect '{spec}': {exc}")
    return client, tools


async def _cmd_run(goal: str, tools: list[str], evolve: int, save: str | None, hitl: bool, mcp: list[str]) -> int:
    from adaptagent import AgentManager, Workflow, WorkflowGenerator
    from adaptagent.hitl import HITLManager

    llm = _build_llm()
    toolset = _resolve_tools(tools)
    mcp_client = None
    if mcp:
        mcp_client, remote_tools = await _connect_mcp(mcp)
        toolset = toolset + remote_tools
    graph = await WorkflowGenerator(llm, tools=toolset).generate_workflow(goal)
    print(graph.display())
    if save:
        graph.save(save)
        print(f"Saved to {save}")

    manager = AgentManager(tools=toolset)
    manager.register_llm(llm)
    manager.build_agents_from_workflow(graph)

    hitl_manager = None
    if hitl:
        hitl_manager = HITLManager()
        hitl_manager.activate()
        print("[hitl] Enabled — you will be prompted at each step.")

    workflow = Workflow(graph=graph, agent_manager=manager, llm=llm, hitl_manager=hitl_manager)
    output = await workflow.execute()
    print("\n=== FINAL OUTPUT ===")
    print(output.final_output)

    if evolve > 0:
        from adaptagent import EvolutionEngine

        engine = EvolutionEngine(graph=graph, agent_manager=manager, llm=llm)
        result = await engine.evolve(generations=evolve)
        print(result)
        output2 = await workflow.execute()
        print("\n=== EVOLVED OUTPUT ===")
        print(output2.final_output)
    if mcp_client:
        await mcp_client.close()
    return 0


async def _cmd_ask(question: str, tools: list[str]) -> int:
    from adaptagent import Agent

    llm = _build_llm()
    agent = Agent(name="assistant", role="helpful assistant", instruction="", llm=llm, tools=_resolve_tools(tools))
    answer = await agent.arun(question)
    print(answer)
    return 0


def _cmd_show(path: str) -> int:
    from adaptagent import WorkflowGraph

    graph = WorkflowGraph.from_file(path)
    print(graph.display())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="adaptagent", description="AdaptAgent: self-evolving agent workflows")
    parser.add_argument("--version", action="version", version=f"adaptagent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Generate and execute a workflow from a goal")
    p_run.add_argument("goal", help="Natural-language goal")
    p_run.add_argument("--tools", default="", help="Comma-separated: wiki,search,python,http,read,write")
    p_run.add_argument("--evolve", "-e", type=int, default=0, help="Evolve prompts for N generations after run")
    p_run.add_argument("--save", default=None, help="Save generated workflow JSON to path")
    p_run.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop review")
    p_run.add_argument(
        "--mcp",
        action="append",
        default=[],
        help="MCP server spec 'command:arg1:arg2' (repeatable)",
    )

    p_ask = sub.add_parser("ask", help="Ask a single ReAct agent")
    p_ask.add_argument("question", help="The question")
    p_ask.add_argument("--tools", default="", help="Comma-separated tool names")

    p_show = sub.add_parser("show", help="Display a saved workflow graph")
    p_show.add_argument("path", help="Path to workflow JSON")

    args = parser.parse_args(argv)
    tools = [t.strip() for t in args.tools.split(",") if t.strip()] if hasattr(args, "tools") else []
    if args.command == "run":
        return asyncio.run(_cmd_run(args.goal, tools, args.evolve, args.save, args.hitl, args.mcp))
    if args.command == "ask":
        return asyncio.run(_cmd_ask(args.question, tools))
    if args.command == "show":
        return _cmd_show(args.path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
