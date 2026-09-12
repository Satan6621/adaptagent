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


async def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    goal = (payload.get("goal") or "").strip()
    if not goal:
        return _json(400, {"error": "goal is required"})
    llm, _, error = await _llm_from_payload(payload)
    if error:
        return _json(400, {"error": error})
    generator = WorkflowGenerator(llm)
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
    manager.build_agents_from_workflow(graph, assign_tools=False)

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
                if path.endswith("/models"):
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
            manager.build_agents_from_workflow(graph, assign_tools=False)
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
