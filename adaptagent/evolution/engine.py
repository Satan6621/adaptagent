"""Evolution Engine: optimize agent prompts via evolutionary search.

Loop: evaluate candidates -> select top -> LLM-driven mutation/crossover ->
evaluate offspring -> keep the best. This is the EvoPrompt recipe adapted to
typed multi-agent workflows, integrated as a first-class workflow operation.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..agents.agent import Agent
from ..workflow.executor import Workflow
from ..workflow.generator import _extract_json
from .evaluator import Evaluator, LLMJudgeEvaluator

if TYPE_CHECKING:
    from ..agents.manager import AgentManager
    from ..llm.base import LLM
    from ..workflow.graph import WorkflowGraph

MUTATION_PROMPT = """You are an expert prompt engineer. Improve the following agent instruction \
to make the agent more effective.

Current instruction:
{instruction}

Performance feedback (score 0-100, higher is better): {score}/100 on the goal: {goal}

Recent output produced with this instruction (may be truncated):
{output}

Output ONLY the improved instruction as plain text, no quotes, no commentary."""

CROSSOVER_PROMPT = """You are an expert prompt engineer. Combine the best elements of these two \
agent instructions into a single, stronger instruction.

Instruction A:
{a}

Instruction B:
{b}

Goal the agents must achieve: {goal}

Output ONLY the combined instruction as plain text, no quotes, no commentary."""


@dataclass
class EvolutionResult:
    best_score: float
    best_output: str
    history: list[dict] = field(default_factory=list)  # [{gen, best, mean}]

    def __str__(self) -> str:
        lines = [f"Evolution finished. Best score: {self.best_score:.1f}/100"]
        for h in self.history:
            lines.append(f"  gen {h['gen']}: best={h['best']:.1f} mean={h['mean']:.1f}")
        return "\n".join(lines)


@dataclass
class _Candidate:
    agent: Agent
    score: float = 0.0
    output: str = ""


class EvolutionEngine:
    """Evolves the prompts of a workflow's agents.

    Usage:
        engine = EvolutionEngine(graph, agent_manager, llm, evaluator)
        result = await engine.evolve(generations=3, population=4)
    """

    def __init__(
        self,
        graph: "WorkflowGraph",
        agent_manager: "AgentManager",
        llm: "LLM",
        evaluator: Evaluator | None = None,
        seed_examples: list[str] | None = None,
    ) -> None:
        self.graph = graph
        self.agent_manager = agent_manager
        self.llm = llm
        self.evaluator = evaluator or LLMJudgeEvaluator(llm)
        self.seed_examples = seed_examples or []

    async def evaluate_workflow(self, goal: str | None = None) -> tuple[float, str]:
        goal = goal or self.graph.goal
        workflow = Workflow(graph=self.graph, agent_manager=self.agent_manager, llm=self.llm)
        output = await workflow.execute()
        score = await self._score(goal, output.final_output)
        return score, output.final_output

    async def _score(self, goal: str, output: str) -> float:
        if isinstance(self.evaluator, LLMJudgeEvaluator):
            return await self.evaluator.evaluate_async(goal, output)
        return await asyncio.to_thread(self.evaluator.evaluate, goal, output)

    async def evolve(self, generations: int = 3, population: int = 4, verbose: bool = True) -> EvolutionResult:
        """Evolutionary loop over the *target agent* (the terminal step's agent)."""
        # 1. identify target: agent of the last step in topological order
        order = self.graph.topo_order()
        if not order:
            raise ValueError("Empty workflow")
        target_step = self.graph.steps[order[-1]]
        base_agent = self.agent_manager.get_agent_for_step(target_step)
        goal = self.graph.goal

        # 2. initial population: base instruction + LLM mutations
        population_list: list[_Candidate] = [_Candidate(agent=base_agent.clone())]
        for _ in range(population - 1):
            mutated = await self._mutate(base_agent, goal, initial=True)
            population_list.append(_Candidate(agent=mutated))

        history: list[dict] = []
        best = population_list[0]
        for gen in range(generations):
            # evaluate all candidates
            await asyncio.gather(
                *(self._evaluate_candidate(c, goal, target_step) for c in population_list if c.score == 0)
            )
            population_list.sort(key=lambda c: c.score, reverse=True)
            best = population_list[0]
            mean = sum(c.score for c in population_list) / len(population_list)
            history.append({"gen": gen, "best": best.score, "mean": mean})
            if verbose:
                print(f"[evolve] gen {gen}: best={best.score:.1f} mean={mean:.1f}")

            # evolve: keep elite + offspring via mutation/crossover
            if gen < generations - 1:
                elite = population_list[: max(1, len(population_list) // 2)]
                offspring: list[_Candidate] = [_Candidate(agent=e.agent.clone()) for e in elite]
                while len(offspring) < population:
                    if len(elite) >= 2 and random.random() < 0.3:
                        a, b = random.sample(elite, 2)
                        child = await self._crossover(a.agent, b.agent, goal)
                    else:
                        parent = random.choice(elite)
                        child = await self._mutate(parent.agent, goal, score=parent.score, output=parent.output)
                    offspring.append(_Candidate(agent=child))
                population_list = offspring

        # 3. commit the best instruction back to the real agent
        base_agent.instruction = best.agent.instruction
        return EvolutionResult(best_score=best.score, best_output=best.output, history=history)

    async def _evaluate_candidate(self, cand: _Candidate, goal: str, target_step) -> None:
        """Run the workflow with the candidate's instruction and score it."""
        original = self.agent_manager.agents.get(self._key(target_step))
        key = self._key(target_step)
        real_agent = self.agent_manager.agents.get(key)
        if real_agent is None:
            real_agent = self.agent_manager.get_agent_for_step(target_step)
        saved_instruction = real_agent.instruction
        try:
            real_agent.instruction = cand.agent.instruction
            workflow = Workflow(graph=self.graph, agent_manager=self.agent_manager, llm=self.llm)
            output = await workflow.execute()
            cand.score = await self._score(goal, output.final_output)
            cand.output = output.final_output
        finally:
            real_agent.instruction = saved_instruction

    def _key(self, step) -> str:
        return f"{step.name} ({step.id})"

    async def _mutate(self, agent: Agent, goal: str, score: float = 50.0, output: str = "", initial: bool = False) -> Agent:
        prompt = MUTATION_PROMPT.format(
            instruction=agent.instruction,
            score=f"{score:.0f} (unknown)" if initial else f"{score:.0f}",
            goal=goal,
            output=output[:1500] or "(not available yet)",
        )
        resp = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=1.0
        )
        new_instruction = resp.text.strip().strip('"')
        return agent.clone(instruction=new_instruction)

    async def _crossover(self, a: Agent, b: Agent, goal: str) -> Agent:
        prompt = CROSSOVER_PROMPT.format(a=a.instruction, b=b.instruction, goal=goal)
        resp = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        return a.clone(instruction=resp.text.strip().strip('"'))
