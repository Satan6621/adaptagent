"""Vercel Python serverless function: generate a workflow from a goal.

POST /api/generate  {"goal": "...", "tools": ["wiki","search"]}
-> {"goal": ..., "steps": [...]}

Set OPENAI_API_KEY (or ADAPTAGENT_* vars) in the Vercel project environment.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from adaptagent import LLMConfig, WorkflowGenerator, resolve_llm


def _json_response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(payload)}


async def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    goal = (payload.get("goal") or "").strip()
    if not goal:
        return _json_response(400, {"error": "goal is required"})
    llm = resolve_llm(LLMConfig())
    generator = WorkflowGenerator(llm)
    graph = await generator.generate_workflow(goal)
    return _json_response(200, json.loads(graph.to_json()))


class _App:
    """ASGI-style entrypoint required by the Vercel Python runtime."""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            response = _json_response(400, {"error": "invalid JSON"})
        else:
            try:
                response = await _generate(payload)
            except Exception as exc:  # noqa: BLE001
                response = _json_response(500, {"error": str(exc)})
        await send(
            {
                "type": "http.response.start",
                "status": response["statusCode"],
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": response["body"].encode()})


app = _App()
