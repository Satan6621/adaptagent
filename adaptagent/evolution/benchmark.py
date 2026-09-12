"""Benchmark harness: evaluate and optimize workflows against datasets.

Datasets: JSONL files with {"input": ..., "expected": ...} per line.
Metrics come from task presets or custom callables. Compatible with the
EvolutionEngine so you can benchmark -> evolve -> re-benchmark.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..agents.manager import AgentManager
    from ..llm.base import LLM
    from ..workflow.graph import WorkflowGraph

METRICS: dict[str, Callable[[str, str], float]] = {
    "em": lambda expected, got: 1.0 if expected.strip() == got.strip() else 0.0,
    "f1": lambda expected, got: _token_f1(expected, got),
    "contains": lambda expected, got: 1.0 if expected.lower() in got.lower() else 0.0,
}


def _token_f1(expected: str, got: str) -> float:
    """SQuAD-style token F1."""
    import string

    def tokens(text: str) -> list[str]:
        return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

    exp, out = tokens(expected), tokens(got)
    if not exp or not out:
        return 1.0 if not exp and not out else 0.0
    common: dict[str, int] = {}
    for t in exp:
        common[t] = min(exp.count(t), out.count(t))
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(out)
    recall = n_common / len(exp)
    return 2 * precision * recall / (precision + recall)


def _parse_score(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0


@dataclass
class BenchmarkCase:
    input: str
    expected: str


@dataclass
class BenchmarkResult:
    metric: str
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def __str__(self) -> str:
        return f"Benchmark [{self.metric}]: mean={self.mean:.4f} over {len(self.scores)} cases"


class Benchmark:
    """Runs a workflow per benchmark case and scores the outputs.

    Usage:
        bench = Benchmark.from_jsonl("data.jsonl", metric="f1")
        result = await bench.run(graph, agent_manager, llm)
    """

    def __init__(self, cases: list[BenchmarkCase], metric: str = "em", name: str = "benchmark") -> None:
        self.cases = cases
        self.metric = metric
        self.name = name

    @classmethod
    def from_jsonl(cls, path: str, metric: str = "em", name: str | None = None) -> "Benchmark":
        cases = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(BenchmarkCase(input=data["input"], expected=str(data.get("expected", ""))))
        return cls(cases=cases, metric=metric, name=name or Path(path).stem)

    async def run(
        self,
        graph: "WorkflowGraph",
        agent_manager: "AgentManager",
        llm: "LLM",
        judge: bool = False,
        limit: int | None = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Execute the workflow per case; score via metric (or LLM judge)."""
        from ..workflow.executor import Workflow

        result = BenchmarkResult(metric=self.metric if not judge else f"{self.metric}+judge")
        for i, case in enumerate(self.cases[: limit or len(self.cases)]):
            workflow = Workflow(graph=graph, agent_manager=agent_manager, llm=llm)
            output = await workflow.execute(inputs={"input": case.input})
            got = output.final_output
            if judge:
                score = await self._judge(case.expected, got, llm)
            else:
                score = METRICS[self.metric](case.expected, got)
            result.scores.append(float(score))
            if verbose:
                print(f"[bench {self.name}] case {i + 1}/{len(self.cases)}: {score:.2f}")
        return result

    async def _judge(self, expected: str, got: str, llm: "LLM") -> float:
        prompt = (
            f"Expected answer: {expected}\n\nCandidate answer:\n{got[:2000]}\n\n"
            "Score the candidate 0-100 on how well it matches the expected substance. "
            "Respond with ONLY the number."
        )
        response = await llm.generate(messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return min(1.0, _parse_score(response.text) / 100.0)
