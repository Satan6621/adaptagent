"""AFlow-style structural evolution: mutate the workflow DAG itself.

While the EvolutionEngine optimizes *prompts* (instructions), AFlowEngine
optimizes the *structure*: adds/removes/reorders/reconnects steps. Each
mutation is proposed by an LLM; if the proposal is invalid, deterministic
fallback operators are applied so evolution never stalls.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..workflow.generator import _extract_json
from ..workflow.graph import WorkflowGraph, WorkflowStep

if TYPE_CHECKING:
    from ..agents.manager import AgentManager
    from ..llm.base import LLM
    from .evaluator import Evaluator

MUTATION_PROMPT = """You are a workflow architect improving a multi-agent pipeline.

Current workflow (steps in topological order):
{workflow}

Goal: {goal}

Performance of the current structure: {score:.0f}/100.
Recent final output (may be truncated):
{output}

Propose ONE structural mutation to improve the workflow. Choose exactly one type:
- add: a new step that fills a missing capability
- remove: delete a low-value step (never the only step)
- modify: change instruction/inputs/role of one step
- reorder: change dependencies between steps (keep it a valid DAG)

Respond ONLY with JSON: {{"type": "add|remove|modify|reorder", "step": {{"id": "...", "name": "...", \
"instruction": "...", "inputs": ["step1"], "agent_hint": "researcher|writer|coder|analyst|critic"}}}}
For "remove": {{"type": "remove", "id": "step_id"}}. For "reorder": {{"type": "reorder", "id": "step_id", \
"new_inputs": ["step1"]}}. For "modify": use the full step object (same id)."""

ROLE_POOL = ["researcher", "writer", "coder", "analyst", "critic", "reviewer"]


@dataclass
class AFlowResult:
    best_score: float
    best_graph: WorkflowGraph
    best_output: str = ""
    history: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"AFlow finished. Best score: {self.best_score:.1f}/100"]
        for h in self.history:
            lines.append(f"  gen {h['gen']}: score={h['score']:.1f} mutation={h['mutation']}")
        return "\n".join(lines)


class AFlowEngine:
    """Structural evolution of a WorkflowGraph.

    Usage:
        engine = AFlowEngine(graph, agent_manager, llm, evaluator)
        result = await engine.evolve(generations=5)
        best_graph = result.best_graph  # optimized structure
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        agent_manager: "AgentManager",
        llm: "LLM",
        evaluator: "Evaluator | None" = None,
    ) -> None:
        self.graph = graph
        self.agent_manager = agent_manager
        self.llm = llm
        from .evaluator import LLMJudgeEvaluator

        self.evaluator = evaluator or LLMJudgeEvaluator(llm)

    async def _score(self, goal: str, output: str) -> float:
        from .evaluator import LLMJudgeEvaluator

        if isinstance(self.evaluator, LLMJudgeEvaluator):
            return await self.evaluator.evaluate_async(goal, output)
        import asyncio

        return await asyncio.to_thread(self.evaluator.evaluate, goal, output)

    def _clone_manager(self) -> "AgentManager":
        """Shallow-clone the manager sharing LLM instances (httpx clients are
        not deep-copyable); agent instructions are isolated per build."""
        from ..agents.manager import AgentManager

        clone = AgentManager(tools=list(self.agent_manager.tools))
        executor_llm = getattr(self.agent_manager, "_default_llm", None) or self.llm
        clone.register_llm(executor_llm)
        return clone

    async def _evaluate_graph(self, graph: WorkflowGraph) -> tuple[float, str]:
        from ..workflow.executor import Workflow

        manager = self._clone_manager()
        manager.build_agents_from_workflow(graph, assign_tools=bool(self.agent_manager.tools))
        workflow = Workflow(graph=graph, agent_manager=manager, llm=self.llm)
        output = await workflow.execute()
        score = await self._score(graph.goal, output.final_output)
        return score, output.final_output

    async def evolve(self, generations: int = 5, verbose: bool = True) -> AFlowResult:
        goal = self.graph.goal
        best_graph = copy.deepcopy(self.graph)
        best_score, best_output = await self._evaluate_graph(best_graph)
        history: list[dict] = []
        for gen in range(generations):
            candidate, mutation_desc = await self._mutate_structure(copy.deepcopy(best_graph), goal, best_score, best_output)
            if candidate is None:
                continue
            try:
                score, output = await self._evaluate_graph(candidate)
            except Exception:  # noqa: BLE001
                continue
            accepted = score >= best_score
            if accepted:
                best_graph, best_score, best_output = candidate, score, output
            history.append({"gen": gen, "score": score, "mutation": mutation_desc, "accepted": accepted})
            if verbose:
                flag = "ACCEPTED" if accepted else "rejected"
                print(f"[aflow] gen {gen}: score={score:.1f} ({flag}) <- {mutation_desc}")
        return AFlowResult(best_score=best_score, best_graph=best_graph, best_output=best_output, history=history)

    # ---------- mutation ----------

    async def _mutate_structure(
        self, graph: WorkflowGraph, goal: str, score: float, output: str
    ) -> tuple[WorkflowGraph | None, str]:
        """LLM proposal first; deterministic fallback on invalid proposals."""
        prompt = MUTATION_PROMPT.format(
            workflow=graph.display(), goal=goal, score=score, output=output[:1500] or "(none yet)"
        )
        try:
            resp = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}], temperature=0.8
            )
            proposal = _extract_json(resp.text)
            mutated = self._apply_proposal(graph, proposal)
            if mutated is not None:
                return mutated, f"llm:{proposal.get('type', '?')}"
        except Exception:  # noqa: BLE001
            pass
        # fallback: deterministic operator
        mutated = self._fallback_mutation(graph)
        return mutated, "fallback"

    def _apply_proposal(self, graph: WorkflowGraph, proposal: dict) -> WorkflowGraph | None:
        mtype = proposal.get("type")
        if mtype == "add":
            step = self._step_from(proposal.get("step", {}))
            if step is None or step.id in graph.steps:
                return None
            candidates = list(graph.steps)
            step.inputs = step.inputs[:1] or [random.choice(candidates)] if candidates else []
            graph.add_step(step)
            return graph
        if mtype == "remove":
            sid = proposal.get("id")
            if sid not in graph.steps or len(graph.steps) <= 1:
                return None
            graph.steps.pop(sid)
            for s in graph.steps.values():
                s.inputs = [i for i in s.inputs if i != sid]
            return graph
        if mtype == "modify":
            sid = proposal.get("step", {}).get("id")
            if sid not in graph.steps:
                return None
            current = graph.steps[sid]
            new = self._step_from(proposal.get("step", {}))
            if new is None:
                return None
            current.instruction = new.instruction or current.instruction
            current.agent_hint = new.agent_hint or current.agent_hint
            return graph
        if mtype == "reorder":
            sid = proposal.get("id")
            new_inputs = proposal.get("new_inputs", [])
            if sid not in graph.steps:
                return None
            valid = [i for i in new_inputs if i in graph.steps and i != sid]
            graph.steps[sid].inputs = valid
            try:
                graph.topo_order()  # validate DAG
                return graph
            except ValueError:
                graph.steps[sid].inputs = []
                return graph
        return None

    def _step_from(self, raw: dict) -> WorkflowStep | None:
        sid = raw.get("id")
        if not sid:
            return None
        return WorkflowStep(
            id=sid,
            name=raw.get("name", sid),
            instruction=raw.get("instruction", ""),
            inputs=[i for i in raw.get("inputs", []) if i != sid],
            agent_hint=raw.get("agent_hint") or random.choice(ROLE_POOL),
        )

    # ---------- deterministic fallback operators ----------

    def _fallback_mutation(self, graph: WorkflowGraph) -> WorkflowGraph | None:
        ops = [
            self._op_add_critic,
            self._op_remove_leaf,
            self._op_swap_role,
            self._op_extend_inputs,
        ]
        random.shuffle(ops)
        for op in ops:
            result = op(graph)
            if result is not None:
                return result
        return None

    def _op_add_critic(self, graph: WorkflowGraph) -> WorkflowGraph | None:
        if len(graph.steps) >= 6:
            return None
        sid = f"step{len(graph.steps) + 1}"
        last = graph.topo_order()[-1]
        step = WorkflowStep(
            id=sid,
            name="Critic",
            instruction="Review the previous output. Identify weaknesses and produce an improved final version.",
            inputs=[last],
            agent_hint="critic",
        )
        graph.add_step(step)
        return graph

    def _op_remove_leaf(self, graph: WorkflowGraph) -> WorkflowGraph | None:
        if len(graph.steps) <= 2:
            return None
        order = graph.topo_order()
        victims = [sid for sid in order[1:] if sid not in {i for s in graph.steps.values() for i in s.inputs}]
        if not victims:
            return None
        sid = random.choice(victims)
        graph.steps.pop(sid)
        for s in graph.steps.values():
            s.inputs = [i for i in s.inputs if i != sid]
        return graph

    def _op_swap_role(self, graph: WorkflowGraph) -> WorkflowGraph | None:
        step = random.choice(list(graph.steps.values()))
        new_role = random.choice(ROLE_POOL)
        if step.agent_hint == new_role:
            return None
        step.agent_hint = new_role
        return graph

    def _op_extend_inputs(self, graph: WorkflowGraph) -> WorkflowGraph | None:
        order = graph.topo_order()
        if len(order) < 3:
            return None
        target = random.choice(order[1:])
        candidates = [s for s in order[: order.index(target)] if s not in graph.steps[target].inputs]
        if not candidates:
            return None
        graph.steps[target].inputs.append(random.choice(candidates))
        return graph
