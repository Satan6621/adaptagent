"""Vercel Python serverless function: generate, execute and stream workflows.

Endpoints (POST):
  /api                -> generate a workflow JSON from a goal
  /api/models         -> list models for a provider
  /api/execute        -> execute a workflow (full JSON response)
  /api/execute/stream -> execute with server-sent events streaming per step

BYOK: the client may supply its own provider/model/api_key (OpenRouter,
OpenAI, Gemini, ...). If absent, falls back to server keys. Keys are used
per-request only — never stored.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from adaptagent import AgentManager, LLMConfig, Workflow, WorkflowGenerator, resolve_llm
from adaptagent.workflow import WorkflowGraph

SERVER_PROVIDERS = {"gemini", "google", "openrouter"}
MAX_EXECUTION_SECONDS = 50.0  # Vercel hobby function limit


def _json(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": status, "body": json.dumps(payload)}


def _resolve_client_config(payload: dict[str, Any]) -> LLMConfig:
    provider = (payload.get("provider") or "").strip().lower() or "openrouter"
    default_model = "openai/gpt-4o-mini" if provider == "openrouter" else "gemini-2.5-flash"
    model = (payload.get("model") or "").strip() or default_model
    api_key = (payload.get("api_key") or "").strip()
    config = LLMConfig(provider=provider, model=model)
    if api_key:
        config = config.with_overrides(api_key=api_key)
    return config


def _check_key(config: LLMConfig) -> str | None:
    if not config.api_key and config.provider not in SERVER_PROVIDERS:
        return f"api_key is required for provider '{config.provider}'"
    return None


async def _llm_from_payload(payload: dict[str, Any]) -> tuple[Any, LLMConfig | None, str | None]:
    """Returns (llm, config, error). config None when payload has no LLM request."""
    config = _resolve_client_config(payload)
    error = _check_key(config)
    if error:
        return None, None, error
    return resolve_llm(config), config, None


def _server_label(spec: dict[str, Any]) -> str:
    return str(
        spec.get("name") or spec.get("url") or spec.get("command") or spec.get("type") or "MCP"
    )


async def _augment_manager(manager: AgentManager, graph: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Assign MCP + custom HTTP tools to agents and apply personal skills.

    Returns a stat dict: {"mcp_tools", "http_tools", "skills": [...], "mcp_errors": [...]}
    Individual MCP server failures are collected in mcp_errors (never abort the run).
    """
    stat: dict[str, Any] = {"mcp_tools": 0, "http_tools": 0, "skills": [], "mcp_errors": []}
    servers = payload.get("mcp_servers") or []
    custom = payload.get("custom_tools") or []
    skills = payload.get("skills") or []
    tools: list[Any] = []
    mcp_client: Any = None
    try:
        if servers:
            from adaptagent.mcp.client import MCPClient

            mcp_client = MCPClient(timeout=12.0)
            for spec in servers:
                try:
                    conn_tools = await asyncio.wait_for(mcp_client.connect_spec(spec), timeout=12.0)
                except Exception as exc:  # noqa: BLE001
                    stat["mcp_errors"].append(f"{_server_label(spec)}: {str(exc)[:300]}")
                else:
                    tools.extend(conn_tools)
                    stat["mcp_tools"] += len(conn_tools)
        if custom:
            from adaptagent.tools.custom import make_custom_http_tool

            for spec in custom:
                try:
                    tools.append(make_custom_http_tool(spec))
                    stat["http_tools"] += 1
                except Exception as exc:  # noqa: BLE001
                    stat["mcp_errors"].append(f"custom tool: {str(exc)[:300]}")
        manager.tools = tools
        manager.build_agents_from_workflow(graph, llm_config=None, assign_tools=bool(tools))
        if skills:
            from adaptagent.skills import apply_skills

            stat["skills"] = apply_skills(manager, skills)
    finally:
        if mcp_client is not None:
            await mcp_client.close()
    return stat


async def _mcp_test(payload: dict[str, Any]) -> dict[str, Any]:
    """Test one MCP server config and report the tools it exposes."""
    server = payload.get("server")
    if not isinstance(server, dict):
        return _json(400, {"error": "server config object is required"})
    from adaptagent.mcp.client import MCPClient

    client = MCPClient(timeout=8.0)
    try:
        tools = await asyncio.wait_for(client.connect_spec(server), timeout=8.0)
        names = sorted({t.name for t in tools})
        return _json(200, {"ok": True, "tools": names, "count": len(names)})
    except asyncio.TimeoutError:
        return _json(200, {"ok": False, "error": "timeout after 8s"})
    except Exception as exc:  # noqa: BLE001
        return _json(200, {"ok": False, "error": str(exc)[:500]})
    finally:
        await client.close()


async def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    goal = (payload.get("goal") or "").strip()
    if not goal:
        return _json(400, {"error": "goal is required"})
    llm, _, error = await _llm_from_payload(payload)
    if error:
        return _json(400, {"error": error})
    tools = []
    for spec in payload.get("custom_tools") or []:
        try:
            from adaptagent.tools.custom import make_custom_http_tool

            tools.append(make_custom_http_tool(spec))
        except Exception:  # noqa: BLE001
            continue
    generator = WorkflowGenerator(llm, tools=tools)
    graph = await generator.generate_workflow(goal)
    return _json(200, json.loads(graph.to_json()))


async def _models(payload: dict[str, Any]) -> dict[str, Any]:
    from adaptagent.llm.base import DEFAULT_MODELS

    provider = (payload.get("provider") or "").strip().lower()
    return _json(200, {"models": DEFAULT_MODELS.get(provider, [])})


async def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a full workflow (existing JSON or generated from goal)."""
    wf_data = payload.get("workflow")
    goal = (payload.get("goal") or (wf_data or {}).get("goal") or "").strip()
    if not wf_data:
        if not goal:
            return _json(400, {"error": "workflow or goal is required"})
        llm, _, error = await _llm_from_payload(payload)
        if error:
            return _json(400, {"error": error})
        wf_data = json.loads((await WorkflowGenerator(llm).generate_workflow(goal)).to_json())
    try:
        graph = WorkflowGraph.from_json(json.dumps(wf_data))
    except Exception as exc:  # noqa: BLE001
        return _json(400, {"error": f"invalid workflow: {exc}"})

    llm, _, error = await _llm_from_payload(payload)
    if error:
        return _json(400, {"error": error})
    manager = AgentManager()
    manager.register_llm(llm)
    advanced = await _augment_manager(manager, graph, payload)

    start = time.time()
    workflow = Workflow(graph=graph, agent_manager=manager, llm=llm, max_parallel=2)
    try:
        output = await asyncio.wait_for(workflow.execute(), timeout=MAX_EXECUTION_SECONDS)
    except asyncio.TimeoutError:
        return _json(504, {"error": f"workflow exceeded {MAX_EXECUTION_SECONDS:.0f}s serverless limit"})
    elapsed = round(time.time() - start, 1)
    steps = [
        {
            "id": sid,
            "name": graph.steps[sid].name,
            "output": result.output[:2000],
            "duration": round(result.duration, 1),
        }
        for sid, result in output.results.items()
    ]
    return _json(
        200,
        {
            "goal": graph.goal,
            "steps": steps,
            "final_output": output.final_output[:6000],
            "elapsed_seconds": elapsed,
            "cost_usd": round(output.total_cost_usd, 6),
            "usage": output.total_usage,
            "meta": advanced,
        },
    )


async def _execute_step(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute ONE step (client-driven orchestration: no serverless time limit).

    payload: {workflow, step_id, step_outputs: {id: output}, provider, model, api_key}
    -> {output, duration, usage, cost_usd}
    """
    wf_data = payload.get("workflow")
    step_id = (payload.get("step_id") or "").strip()
    step_outputs = payload.get("step_outputs") or {}
    if not wf_data or not step_id:
        return _json(400, {"error": "workflow and step_id are required"})
    try:
        graph = WorkflowGraph.from_json(json.dumps(wf_data))
        graph.topo_order()
    except KeyError as exc:
        return _json(400, {"error": f"invalid workflow: missing field {exc}"})
    except Exception as exc:  # noqa: BLE001
        return _json(400, {"error": f"invalid workflow: {exc}"})
    if step_id not in graph.steps:
        return _json(400, {"error": f"unknown step_id '{step_id}'"})
    llm, _, error = await _llm_from_payload(payload)
    if error:
        return _json(400, {"error": error})

    # validate condition if present
    step = graph.steps[step_id]
    cond = step.condition
    if cond is not None:
        source_out = step_outputs.get(cond.source, "")
        if not cond.evaluate(source_out):
            return _json(200, {"skipped": True, "output": "", "usage": {}, "cost_usd": 0.0})

    # execute single step (with personal tools/skills if provided)
    manager = AgentManager()
    manager.register_llm(llm)
    advanced = await _augment_manager(manager, graph, payload)
    from adaptagent.workflow.executor import StepResult

    workflow = Workflow(graph=graph, agent_manager=manager, llm=llm, max_parallel=1)
    fake_results = {sid: StepResult(step_id=sid, output=out) for sid, out in step_outputs.items()}
    start = time.time()
    try:
        result = await asyncio.wait_for(workflow._run_step(step_id, fake_results, {}), timeout=50.0)
    except asyncio.TimeoutError:
        return _json(504, {"error": "step exceeded 50s limit"})
    elapsed = round(time.time() - start, 1)
    return _json(
        200,
        {
            "output": result.output[:6000],
            "duration": elapsed,
            "usage": result.usage,
            "cost_usd": round(result.cost_usd, 6),
            "skipped": False,
            "meta": advanced,
        },
    )


class _App:
    """ASGI entrypoint for the Vercel Python runtime (JSON + SSE)."""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        path = scope.get("path", "/api")
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            response = _json(400, {"error": "invalid JSON"})
        else:
            try:
                if path.endswith("/execute/stream"):
                    return await self._stream_execute(payload, send)
                if path.endswith("/mcp/test"):
                    response = await _mcp_test(payload)
                elif path.endswith("/step"):
                    response = await _execute_step(payload)
                elif path.endswith("/models"):
                    response = await _models(payload)
                elif path.endswith("/execute"):
                    response = await _execute(payload)
                else:
                    response = await _generate(payload)
            except Exception as exc:  # noqa: BLE001
                response = _json(500, {"error": str(exc)})
        await send(
            {
                "type": "http.response.start",
                "status": response["statusCode"],
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": response["body"].encode()})

    async def _stream_execute(self, payload: dict[str, Any], send: Any) -> None:
        """SSE: streams step_start/step_done/done/error events as they happen."""
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")],
            }
        )

        async def sse(event: dict) -> None:
            await send({"type": "http.response.body", "body": (f"data: {json.dumps(event)}\n\n").encode(), "more_body": True})

        try:
            wf_data = payload.get("workflow")
            if not wf_data:
                await sse({"event": "error", "error": "workflow is required"})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            graph = WorkflowGraph.from_json(json.dumps(wf_data))
            graph.topo_order()  # validate DAG / non-empty
            llm, _, error = await _llm_from_payload(payload)
            if error:
                await sse({"event": "error", "error": error})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            manager = AgentManager()
            manager.register_llm(llm)
            advanced = await _augment_manager(manager, graph, payload)
            await sse({"event": "meta", "meta": advanced})
        except KeyError as exc:
            await sse({"event": "error", "error": f"invalid workflow: missing field {exc}"})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        except Exception as exc:  # noqa: BLE001
            await sse({"event": "error", "error": str(exc)})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        workflow = Workflow(
            graph=graph,
            agent_manager=manager,
            llm=llm,
            max_parallel=2,
            on_event=lambda e: sse(e),
        )
        try:
            await asyncio.wait_for(workflow.execute(), timeout=MAX_EXECUTION_SECONDS)
        except asyncio.TimeoutError:
            await sse({"event": "error", "error": f"workflow exceeded {MAX_EXECUTION_SECONDS:.0f}s limit"})
        except Exception as exc:  # noqa: BLE001
            await sse({"event": "error", "error": str(exc)})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


app = _App()
