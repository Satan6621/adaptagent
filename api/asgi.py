"""Vercel Python serverless function: generate a workflow from a goal.

POST /api  {"goal": "...", "provider": "...", "model": "...", "api_key": "..."}

BYOK: the client may supply its own provider/model/api_key (OpenRouter,
OpenAI, Gemini, DeepSeek, ...). If absent, falls back to the server-side
GEMINI_API_KEY. Keys are used per-request only — never stored.
"""

from __future__ import annotations

import json
from typing import Any

from adaptagent import LLMConfig, WorkflowGenerator, resolve_llm

SERVER_PROVIDERS = {"gemini", "google", "openrouter"}


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


async def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    goal = (payload.get("goal") or "").strip()
    if not goal:
        return _json(400, {"error": "goal is required"})
    config = _resolve_client_config(payload)
    if not config.api_key and config.provider not in SERVER_PROVIDERS:
        return _json(400, {"error": f"api_key is required for provider '{config.provider}'"})
    llm = resolve_llm(config)
    generator = WorkflowGenerator(llm)
    graph = await generator.generate_workflow(goal)
    return _json(200, json.loads(graph.to_json()))


async def _models(payload: dict[str, Any]) -> dict[str, Any]:
    from adaptagent.llm.base import DEFAULT_MODELS

    provider = (payload.get("provider") or "").strip().lower()
    return _json(200, {"models": DEFAULT_MODELS.get(provider, [])})


class _App:
    """ASGI entrypoint for the Vercel Python runtime."""

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
                if path.endswith("/models"):
                    response = await _models(payload)
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


app = _App()
