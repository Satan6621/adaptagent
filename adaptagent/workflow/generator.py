"""Generates a typed workflow DAG from a natural-language goal via LLM."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from ..llm.base import LLM
from .graph import WorkflowGraph, WorkflowStep

if TYPE_CHECKING:
    from ..tools.base import Tool

GENERATION_PROMPT = """You are a workflow architect. Decompose the user's goal into a minimal set of \
coordinated steps (2-6 steps). Each step is executed by one agent.

Rules:
- Output ONLY valid JSON, no markdown fences, no commentary.
- Structure: {{"goal": "...", "steps": [{{"id": "step1", "name": "...", "instruction": "...", \
"inputs": ["step1"], "output": "output", "agent_hint": "researcher|writer|coder|analyst|critic|reviewer"}}]}}
- ids: step1, step2, ... in topological order. "inputs" may only reference earlier step ids (or be empty).
- "instruction" is a complete, self-contained instruction for the agent executing that step.
- agent_hint: suggested role for the agent.

Goal: {goal}"""

TOOL_SECTION = """

You may assign tools to steps. Available tools:
{tool_lines}

Add a "tools" field to steps that need it: "tools": ["tool_name", ...]."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM output:\n{text[:500]}")
    return json.loads(match.group(0))


class WorkflowGenerator:
    """Builds a WorkflowGraph from a goal, optionally considering tools."""

    def __init__(self, llm: LLM, tools: list[Tool] | None = None) -> None:
        self.llm = llm
        self.tools = tools or []

    def _tool_lines(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools)

    async def generate_workflow(self, goal: str) -> WorkflowGraph:
        prompt = GENERATION_PROMPT.format(goal=goal)
        if self.tools:
            prompt += TOOL_SECTION.format(tool_lines=self._tool_lines())
        response = await self.llm.generate(
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        data = _extract_json(response.text)
        steps: list[WorkflowStep] = []
        for raw in data.get("steps", []):
            if not raw.get("id") or not raw.get("instruction"):
                continue
            step = WorkflowStep(
                id=raw["id"],
                name=raw.get("name", raw["id"]),
                instruction=raw["instruction"],
                inputs=[i for i in raw.get("inputs", []) if i != raw["id"]],
                output=raw.get("output", "output"),
                agent_hint=raw.get("agent_hint"),
            )
            steps.append(step)
        graph = WorkflowGraph(goal=data.get("goal", goal), steps=steps)
        graph.topo_order()  # validate: raises on cycle
        return graph

    def generate_workflow_sync(self, goal: str) -> WorkflowGraph:
        import asyncio

        return asyncio.run(self.generate_workflow(goal))
