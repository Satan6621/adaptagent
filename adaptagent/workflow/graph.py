"""Typed workflow DAG: nodes are agents, edges are execution dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MAX_NODE_NAME_ITER = 100


@dataclass
class Conditional:
    """Edge condition: route to the step only if the rule passes.

    Rules: contains | not_contains | nonempty | regex | score_gte
    score_gte is evaluated asynchronously in the executor using an LLM judge
    (out of scope here: evaluate() returns True for it).
    """

    source: str  # step id whose output is evaluated
    rule: str  # contains|not_contains|nonempty|regex|score_gte
    value: str = ""  # needle, regex pattern, or minimum score (for score_gte)

    def evaluate(self, source_output: str) -> bool:
        text = source_output or ""
        if self.rule == "contains":
            return self.value.lower() in text.lower()
        if self.rule == "not_contains":
            return self.value.lower() not in text.lower()
        if self.rule == "nonempty":
            return len(text.strip()) > 0
        if self.rule == "regex":
            import re

            return re.search(self.value, text) is not None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "rule": self.rule, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conditional":
        return cls(source=data["source"], rule=data.get("rule", "nonempty"), value=data.get("value", ""))


@dataclass
class CodeSpec:
    """Optional code-execution for a step: runs `source` in the existing
    sandboxed python_repl (Process + timeout + restricted builtins).
    NO LLM, NO tokens, zero cost -- serverless-safe port of the
    smolagents CodeAgent exec-node (python_repl tool)."""
    source: str = ""
    timeout_s: int = 20
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "timeout_s": self.timeout_s, "language": self.language}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeSpec":
        return cls(
            source=str(data.get("source", "")),
            timeout_s=int(data.get("timeout_s", 20)),
            language=str(data.get("language", "python")),
        )


@dataclass
class RetrievalSpec:
    """Optional declarative retrieval for a step: runs BM25 over the
    RetrievalStore and returns docs as 'output' -- NO LLM, NO cost.
    Mirrors smolagents RetrieverTool + LlamaIndex retrieve-node."""
    index: str = "kb"
    query: str = ""
    k: int = 3
    min_score: float = 0.0

    def to_dict(self) -> dict:
        return {"index": self.index, "query": self.query, "k": self.k, "min_score": self.min_score}

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalSpec":
        return cls(index=data.get("index", "kb"), query=data.get("query", ""),
                   k=int(data.get("k", 3)), min_score=float(data.get("min_score", 0.0)))


@dataclass
class WorkflowStep:
    """A single step in the workflow."""

    id: str
    name: str
    instruction: str
    inputs: list[str] = field(default_factory=list)  # upstream node ids
    output: str = "output"
    agent_hint: str | None = None  # e.g. "researcher", "writer"
    condition: Conditional | None = None  # skip step if the incoming condition fails
    repeat_until: Conditional | None = None  # loop: re-run step until this passes
    max_repeats: int = 3  # safety cap for repeat_until loops
    output_schema: dict[str, Any] | None = None  # optional JSON schema; validate+retry
    retrieval: RetrievalSpec | None = None
    code: CodeSpec | None = None  # if set, step runs sandboxed python (no LLM, no cost)  # if set, step is a BM25 retrieval (no LLM, no cost)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "instruction": self.instruction,
            "inputs": self.inputs,
            "output": self.output,
            "agent_hint": self.agent_hint,
        }
        if self.condition:
            data["condition"] = self.condition.to_dict()
        if self.repeat_until:
            data["repeat_until"] = self.repeat_until.to_dict()
            data["max_repeats"] = self.max_repeats
        if self.output_schema:
            data["output_schema"] = self.output_schema
        if self.retrieval is not None:
            data["retrieval"] = self.retrieval.to_dict()
        if self.code is not None:
            data["code"] = self.code.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        condition = data.get("condition")
        repeat = data.get("repeat_until")
        return cls(
            id=data["id"],
            name=data["name"],
            instruction=data["instruction"],
            inputs=list(data.get("inputs", [])),
            output=data.get("output", "output"),
            agent_hint=data.get("agent_hint"),
            condition=Conditional.from_dict(condition) if condition else None,
            repeat_until=Conditional.from_dict(repeat) if repeat else None,
            max_repeats=int(data.get("max_repeats", 3)),
            output_schema=data.get("output_schema"),
            retrieval=RetrievalSpec.from_dict(data["retrieval"]) if data.get("retrieval") else None,
            code=CodeSpec.from_dict(data["code"]) if data.get("code") else None,
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
