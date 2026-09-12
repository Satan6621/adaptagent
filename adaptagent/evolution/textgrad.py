"""TextGrad: prompt optimization via textual gradients.

Recipe (Madaan et al., Nature 2025, adapted):
1. Forward pass: run the workflow, get output.
2. Feedback: LLM criticizes output -> "textual gradient" of the loss.
3. Backward pass: LLM rewrites the prompt applying the gradient.
4. Momentum: accumulate past gradients to stabilize updates.

Cheaper than evolutionary search: 1 evaluation per iteration (vs population).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.manager import AgentManager
    from ..llm.base import LLM
    from ..workflow.graph import WorkflowGraph

FEEDBACK_PROMPT = """You are a strict evaluator. Given this goal and the output, provide concrete \
"textual gradient" feedback: what exactly is wrong or missing, and what should change.

Goal: {goal}

Output:
{output}

Format:
PROBLEMS: <numbered list of specific problems>
DIRECTION: <one-sentence instruction on how to improve the output>"""

BACKWARD_PROMPT = """You are optimizing an agent instruction via textual gradients.

Current instruction:
{instruction}

Goal: {goal}

Gradients (accumulated feedback from previous iterations, most recent last):
{gradients}

Latest output produced with the current instruction:
{output}

Rewrite the instruction to address the feedback. Keep what works, fix what fails. \
Output ONLY the new instruction as plain text, no quotes, no commentary."""


@dataclass
class TextGradResult:
    best_score: float
    best_instruction: str
    best_output: str
    history: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"TextGrad finished. Best score: {self.best_score:.1f}/100"]
        for h in self.history:
            lines.append(f"  iter {h['iter']}: score={h['score']:.1f} improved={h['improved']}")
        return "\n".join(lines)


class TextGradOptimizer:
    """Optimizes the terminal-step agent's prompt with textual gradients.

    Usage:
        opt = TextGradOptimizer(graph, agent_manager, llm, evaluator)
        result = await opt.optimize(iterations=4)
    """

    MOMENTUM_WINDOW = 3  # gradients kept for momentum

    def __init__(
        self,
        graph: "WorkflowGraph",
        agent_manager: "AgentManager",
        llm: "LLM",
        evaluator: "Evaluator | None" = None,
    ) -> None:
        self.graph = graph
        self.agent_manager = agent_manager
        self.llm = llm
        from .evaluator import Evaluator, LLMJudgeEvaluator

        self.evaluator = evaluator or LLMJudgeEvaluator(llm)

    async def _score(self, goal: str, output: str) -> float:
        from .evaluator import LLMJudgeEvaluator

        if isinstance(self.evaluator, LLMJudgeEvaluator):
            return await self.evaluator.evaluate_async(goal, output)
        import asyncio

        return await asyncio.to_thread(self.evaluator.evaluate, goal, output)

    async def _run_workflow(self, goal: str) -> str:
        from ..workflow.executor import Workflow

        workflow = Workflow(graph=self.graph, agent_manager=self.agent_manager, llm=self.llm)
        output = await workflow.execute()
        return output.final_output

    async def optimize(self, iterations: int = 4, verbose: bool = True) -> TextGradResult:
        order = self.graph.topo_order()
        if not order:
            raise ValueError("Empty workflow")
        target_step = self.graph.steps[order[-1]]
        agent = self.agent_manager.get_agent_for_step(target_step)
        goal = self.graph.goal

        best_score = await self._score(goal, await self._run_workflow(goal))
        best_instruction = agent.instruction
        best_output = ""
        gradients: list[str] = []
        history: list[dict] = []

        for it in range(iterations):
            output = await self._run_workflow(goal)
            # 1. feedback = textual gradient
            feedback = await self.llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": FEEDBACK_PROMPT.format(goal=goal, output=output[:3000]),
                    }
                ],
                temperature=0.2,
            )
            gradients.append(feedback.text.strip())
            gradients = gradients[-self.MOMENTUM_WINDOW :]
            # 2. backward pass: apply gradient to the instruction
            rewrite = await self.llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": BACKWARD_PROMPT.format(
                            instruction=agent.instruction,
                            goal=goal,
                            gradients="\n".join(f"- {g}" for g in gradients),
                            output=output[:1500],
                        ),
                    }
                ],
                temperature=0.7,
            )
            candidate = rewrite.text.strip().strip('"')
            # 3. evaluate candidate
            saved = agent.instruction
            agent.instruction = candidate
            try:
                new_output = await self._run_workflow(goal)
                new_score = await self._score(goal, new_output)
            finally:
                agent.instruction = saved
            improved = new_score >= best_score
            history.append({"iter": it, "score": new_score, "improved": improved})
            if verbose:
                print(f"[textgrad] iter {it}: score={new_score:.1f} {'✓ kept' if improved else '✗ reverted'}")
            if improved:
                best_score, best_instruction, best_output = new_score, candidate, new_output
                agent.instruction = candidate
            else:
                gradients.pop()  # unhelpful gradient: drop it

        return TextGradResult(
            best_score=best_score,
            best_instruction=best_instruction,
            best_output=best_output,
            history=history,
        )
