"""Typed workflow DAG: nodes are agents, edges are execution dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MAX_NODE_NAME_ITER = 100


@dataclass
class WorkflowStep:
    """A single step in the workflow."""

    id: str
    name: str
    instruction: str
    inputs: list[str] = field(default_factory=list)  # upstream node ids
    output: str = "output"
    agent_hint: str | None = None  # e.g. "researcher", "writer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "instruction": self.instruction,
            "inputs": self.inputs,
            "output": self.output,
            "agent_hint": self.agent_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        return cls(
            id=data["id"],
            name=data["name"],
            instruction=data["instruction"],
            inputs=list(data.get("inputs", [])),
            output=data.get("output", "output"),
            agent_hint=data.get("agent_hint"),
        )


class WorkflowGraph:
    """A DAG of steps. Execution order is computed by topological sort."""

    def __init__(self, goal: str = "", steps: list[WorkflowStep] | None = None) -> None:
        self.goal = goal
        self.steps: dict[str, WorkflowStep] = {s.id: s for s in (steps or [])}

    def add_step(self, step: WorkflowStep) -> None:
        if step.id in self.steps:
            raise ValueError(f"Duplicate step id: {step.id}")
        self.steps[step.id] = step

    def edges(self) -> list[tuple[str, str]]:
        return [(src, step.id) for step in self.steps.values() for src in step.inputs if src in self.steps]

    def roots(self) -> list[str]:
        targets = {t for _, t in self.edges()}
        return [sid for sid in self.steps if sid not in targets]

    def topo_order(self) -> list[str]:
        indeg = {sid: 0 for sid in self.steps}
        for _, dst in self.edges():
            indeg[dst] += 1
        queue = [sid for sid, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            sid = queue.pop()
            order.append(sid)
            for src, dst in self.edges():
                if src == sid:
                    indeg[dst] -= 1
                    if indeg[dst] == 0:
                        queue.append(dst)
        if len(order) != len(self.steps):
            raise ValueError("Cycle detected in workflow graph")
        return order

    def to_json(self) -> str:
        return json.dumps(
            {"goal": self.goal, "steps": [s.to_dict() for s in self.steps.values()]}, indent=2, ensure_ascii=False
        )

    @classmethod
    def from_json(cls, text: str) -> "WorkflowGraph":
        data = json.loads(text)
        return cls(goal=data.get("goal", ""), steps=[WorkflowStep.from_dict(s) for s in data.get("steps", [])])

    def display(self) -> str:
        lines = [f"Goal: {self.goal}", ""]
        for sid in self.topo_order():
            step = self.steps[sid]
            deps = ", ".join(step.inputs) if step.inputs else "(root)"
            lines.append(f"[{sid}] {step.name} <- {deps}")
        lines.append("")
        lines.append(f"Total: {len(self.steps)} steps")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_file(cls, path: str) -> "WorkflowGraph":
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())
