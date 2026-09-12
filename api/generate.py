"""Vercel Python serverless function: generate a workflow from a goal.

POST /api/generate  {"goal": "...", "tools": ["wiki","search"]}
-> {"workflow": {goal, steps[]}}

Uses the ADAPTAGENT_* / OPENAI_API_KEY env vars configured in Vercel.
"""

from __future__ import annotations

import json
from typing import Any

from httpx import HTTPError

from adaptagent import LLMConfig, WorkflowGenerator, resolve_llm
from adaptagent.workflow.generator import _extract_json


def handler(request: Any) -> dict[str, Any]:
    if request.method != "POST":
        return {"statusCode": 405, "body": json.dumps({"error": "POST only"})}
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": json.dumps({"error": "invalid JSON"})}
    goal = (payload.get("goal") or "").strip()
    if not goal:
        return {"statusCode": 400, "body": json.dumps({"error": "goal is required"})}

    try:
        llm = resolve_llm(LLMConfig())
        generator = WorkflowGenerator(llm)
        graph = __import__("asyncio").run(generator.generate_workflow(goal))
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": graph.to_json(),
        }
    except HTTPError as exc:
        return {"statusCode": 502, "body": json.dumps({"error": f"LLM provider error: {exc}"})}
    except Exception as exc:  # noqa: BLE001
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
