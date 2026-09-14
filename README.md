# AdaptAgent

[![CI](https://github.com/Satan6621/adaptagent/actions/workflows/ci.yml/badge.svg)](https://github.com/Satan6621/adaptagent/actions)
[![Live playground](https://img.shields.io/badge/playground-adaptagent.vercel.app-58a6ff)](https://adaptagent.vercel.app)

A lightweight, strongly-typed framework for building, evaluating and **self-evolving** LLM agent workflows.

**Live playground**: https://adaptagent.vercel.app — generate, edit and **execute** workflows visually (server-side OpenRouter or BYOK).

## Why AdaptAgent (vs. EvoAgentX)

| Dimension | AdaptAgent | EvoAgentX |
|---|---|---|
| Core dependencies | 1 (httpx) | many |
| Typing | strict dataclasses/Protocols everywhere | loose |
| Evolution | 3 engines: EvoPrompt, AFlow, TextGrad | separate optimizer layer |
| MCP | built-in stdio client, zero SDK | tutorial-level |
| Web playground | visual editor deployable to Vercel | none |
| Providers | any OpenAI-compatible endpoint + Anthropic bridge | via SDKs/LiteLLM |
| Sandbox | process-isolated Python REPL | interpreter class |

## Install

```bash
pip install -e .            # core (httpx only)
pip install -e ".[search]"  # + web search (ddgs)
```

## Quickstart

```python
import asyncio
from adaptagent import AgentManager, LLMConfig, Workflow, WorkflowGenerator, resolve_llm

llm = resolve_llm(LLMConfig())  # env: OPENAI_API_KEY / ADAPTAGENT_*

graph = asyncio.run(WorkflowGenerator(llm).generate_workflow("Write a haiku about the sea"))
manager = AgentManager()
manager.register_llm(llm)
manager.build_agents_from_workflow(graph)

output = asyncio.run(Workflow(graph, manager, llm).execute())
print(output.final_output)
```

## Self-evolution (the differentiator)

Three complementary optimization engines:

```python
from adaptagent import EvolutionEngine, AFlowEngine, TextGradOptimizer

# 1. EvoPrompt: evolutionary search on prompts (population, mutation, crossover)
result = await EvolutionEngine(graph, manager, llm).evolve(generations=3, population=4)

# 2. AFlow: structural evolution of the DAG itself (add/remove/modify/reorder steps)
result = await AFlowEngine(graph, manager, llm).evolve(generations=5)

# 3. TextGrad: textual gradients — cheaper, 1 evaluation per iteration
result = await TextGradOptimizer(graph, manager, llm).optimize(iterations=4)
```

## MCP (Model Context Protocol)

```python
from adaptagent.mcp import MCPClient

client = MCPClient()
tools = await client.connect("npx", ["-y", "@modelcontextprotocol/server-everything"])
agent = Agent(name="a", role="r", instruction="", llm=llm, tools=tools)
# ... use the agent; remote tools behave like local ones
await client.close()
```

## CLI

```bash
adaptagent run "analiza X y escribe un resumen" --tools wiki,search --evolve 3 --save wf.json
adaptagent run "objetivo" --mcp "python:server.py" --mcp "npx:-y:@some/server" --hitl
adaptagent ask "¿qué es la computación cuántica?" --tools wiki,python
adaptagent show wf.json
```

## Benchmarks

```python
from adaptagent import Benchmark

# JSONL: {"input": ..., "expected": ...} per line
bench = Benchmark.from_jsonl("data.jsonl", metric="f1")  # or "em", "contains"
result = await bench.run(graph, agent_manager, llm, judge=False)
print(result)  # Benchmark [f1]: mean=0.7231 over 50 cases

# combine with evolution: benchmark -> evolve -> re-benchmark
from adaptagent import TextGradOptimizer
await TextGradOptimizer(graph, manager, llm).optimize(iterations=4)
result2 = await bench.run(graph, manager, llm)
```

## Web playground (Vercel)

```bash
vercel dev     # local: http://localhost:3000
vercel         # deploy (set OPENAI_API_KEY in the project env)
```

Visual editor: generate workflows from a goal, drag nodes, double-click to edit
instructions/roles/dependencies, export JSON.

Playground includes **methodology templates** adapted from the Superpowers
methodology (plan→spec→design, TDD Red-Green-Refactor with self-correcting
loops, systematic debugging, sprint with severity-graded review), a one-click
skills kit and an optional **methodology mode** that injects a process contract
(design-before-code, TDD, evidence over claims, root-cause debugging) into every
agent. See `adaptagent/skills.py`.

## Providers

`resolve_llm` auto-configures from env vars:

| Provider | Env key | Base URL |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | api.openai.com/v1 |
| DeepSeek | `DEEPSEEK_API_KEY` | api.deepseek.com/v1 |
| OpenRouter | `OPENROUTER_API_KEY` | openrouter.ai/api/v1 |
| SiliconFlow | `SILICONFLOW_API_KEY` | api.siliconflow.com/v1 |
| Ollama (local) | — | localhost:11434/v1 |
| vLLM (local) | — | localhost:8000/v1 |
| LM Studio (local) | — | localhost:1234/v1 |
| Anthropic | `ANTHROPIC_API_KEY` | native bridge |

Override anything: `LLMConfig(provider="ollama", model="llama3.1", base_url="http://localhost:11434/v1")`.

## Tools

- `PythonREPLTool` — process-isolated sandbox (no network/fs, allow-listed modules)
- `DDGSSearchTool` — web search (optional `ddgs`)
- `WikipediaSearchTool` — no key required
- `FileReadTool` / `FileWriteTool` / `FileListTool` — confined to `ADAPTAGENT_WORKSPACE`
- `HTTPRequestTool` — GET/POST/PUT/DELETE

Custom tool: decorate any function.

```python
from adaptagent import tool

@tool
def shout(text: str) -> str:
    """Uppercase the text.
    text: Text to shout.
    """
    return text.upper()
```

## HITL (Human-in-the-Loop)

```python
from adaptagent import HITLManager

hitl = HITLManager()
hitl.activate()  # console prompts: approve / edit / reject
workflow = Workflow(graph, manager, llm, hitl_manager=hitl)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Status

v0.2.0 — core workflow engine, 3 evolution engines (EvoPrompt, AFlow, TextGrad),
MCP client, tools, memory, HITL, CLI, web playground.
