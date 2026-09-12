"""Evaluators: score workflow outputs (LLM-as-judge or custom functions)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..llm.base import LLM


@dataclass
class Evaluator:
    """Scores a workflow output against the goal. Higher is better (0-100)."""

    name: str
    fn: Callable[[str, str], float]  # (goal, output) -> score

    def evaluate(self, goal: str, output: str) -> float:
        score = self.fn(goal, output)
        return max(0.0, min(100.0, float(score)))


def _parse_score(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0


class LLMJudgeEvaluator(Evaluator):
    """LLM-as-judge: asks the model to score output quality 0-100."""

    JUDGE_PROMPT = """You are a strict evaluator. Score the following output against the goal.

Goal: {goal}

Output to evaluate:
{output}

Evaluate: correctness, completeness, relevance, quality. Respond with a single integer 0-100 \
and a one-line justification, in this exact format:
SCORE: <number> | {justification}"""

    def __init__(self, llm: "LLM", name: str = "llm_judge") -> None:
        super().__init__(name=name, fn=self._judge)
        self.llm = llm

    async def evaluate_async(self, goal: str, output: str) -> float:
        prompt = self.JUDGE_PROMPT.format(goal=goal, output=output[:4000])
        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return _parse_score(response.text)

    def _judge(self, goal: str, output: str) -> float:
        raise NotImplementedError("Use evaluate_async for LLMJudgeEvaluator")


class ExactMatchEvaluator(Evaluator):
    """For tests: score 100 if the target substring is present."""

    def __init__(self, target: str) -> None:
        super().__init__(name="exact_match", fn=lambda goal, output: 100.0 if target in output else 0.0)
