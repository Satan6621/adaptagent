"""Executes a WorkflowGraph step-by-step using agents from an AgentManager."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .graph import WorkflowGraph

if TYPE_CHECKING:
    from ..agents.manager import AgentManager
    from ..hitl.manager import HITLManager
    from ..llm.base import LLM


@dataclass
class StepResult:
    step_id: str
    output: str
    duration: float = 0.0
    skipped: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowOutput:
    goal: str
    results: dict[str, StepResult] = field(default_factory=dict)
    final_output: str = ""
    total_cost_usd: float = 0.0
    total_usage: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, step_id: str) -> StepResult:
        return self.results[step_id]


class Workflow:
    """Orchestrates execution of a WorkflowGraph with AgentManager agents.

    on_event: optional async callback receiving execution events:
        {"event": "step_start", "step_id", "name"}
        {"event": "step_done", "step_id", "name", "duration", "output", "usage", "cost_usd"}
        {"event": "step_skipped", "step_id", "name"}
        {"event": "done", "final_output", "elapsed", "cost_usd", "usage"}
        {"event": "error", "error"}
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        agent_manager: "AgentManager",
        llm: "LLM",
        hitl_manager: "HITLManager | None" = None,
        max_parallel: int = 4,
        on_event: Callable[[dict], Any] | None = None,
    ) -> None:
        self.graph = graph
        self.agent_manager = agent_manager
        self.llm = llm
        self.hitl_manager = hitl_manager
        self.max_parallel = max_parallel
        self.on_event = on_event
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def _emit(self, event: dict) -> None:
        if self.on_event is not None:
            result = self.on_event(event)
            if asyncio.iscoroutine(result):
                await result

    def _condition_passes(self, step: Any, outputs: dict[str, StepResult]) -> bool:
        """Evaluate the step's incoming condition against its source output."""
        cond = step.condition
        if cond is None:
            return True
        source_result = outputs.get(cond.source)
        source_output = source_result.output if source_result else ""
        return cond.evaluate(source_output)

    async def execute(self, inputs: dict[str, str] | None = None) -> WorkflowOutput:
        inputs = inputs or {}
        start_all = time.perf_counter()
        outputs: dict[str, StepResult] = {}
        totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        total_cost = 0.0
        order = self.graph.topo_order()
        remaining = list(order)
        pending: dict[str, asyncio.Task] = {}
        while remaining or pending:
            launchable = []
            for sid in list(remaining):
                step = self.graph.steps[sid]
                if all(dep in outputs for dep in step.inputs):
                    if not self._condition_passes(step, outputs):
                        remaining.remove(sid)
                        outputs[sid] = StepResult(step_id=sid, output="", skipped=True)
                        await self._emit(
                            {"event": "step_skipped", "step_id": sid, "name": step.name}
                        )
                    else:
                        launchable.append(sid)
            for sid in launchable:
                remaining.remove(sid)
                pending[sid] = asyncio.create_task(self._run_step(sid, outputs, inputs))
            if not pending:
                break
            done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                for sid, t in list(pending.items()):
                    if t is task:
                        outputs[sid] = task.result()
                        del pending[sid]
                        if not outputs[sid].skipped:
                            for k, v in outputs[sid].usage.items():
                                totals[k] = totals.get(k, 0) + v
                            total_cost += outputs[sid].cost_usd
                        await self._emit(
                            {
                                "event": "step_done",
                                "step_id": sid,
                                "name": self.graph.steps[sid].name,
                                "duration": round(outputs[sid].duration, 1),
                                "output": outputs[sid].output,
                                "usage": outputs[sid].usage,
                                "cost_usd": round(outputs[sid].cost_usd, 6),
                            }
                        )
                        break
        elapsed = time.perf_counter() - start_all
        # final output: last non-skipped step in topological order
        final = ""
        for sid in order:
            r = outputs.get(sid)
            if r and not r.skipped and r.output:
                final = r.output
        await self._emit(
            {
                "event": "done",
                "final_output": final,
                "elapsed": round(elapsed, 1),
                "cost_usd": round(total_cost, 6),
                "usage": totals,
            }
        )
        return WorkflowOutput(
            goal=self.graph.goal,
            results=outputs,
            final_output=final,
            total_cost_usd=total_cost,
            total_usage=totals,
        )

    async def _run_step(
        self, step_id: str, outputs: dict[str, StepResult], inputs: dict[str, str]
    ) -> StepResult:
        step = self.graph.steps[step_id]
        await self._emit({"event": "step_start", "step_id": step_id, "name": step.name})
        async with self._semaphore:
            start = time.perf_counter()
            agent = self.agent_manager.get_agent_for_step(step)
            usage: dict[str, int] = {}
            cost = 0.0
            output = ""
            repeats = 0
            while True:
                prompt = self._build_prompt(step, outputs, inputs)
                output = await agent.arun(prompt)
                usage = dict(agent.last_usage)
                cost += agent.last_cost_usd
                if step.repeat_until is None or step.repeat_until.evaluate(output):
                    break
                repeats += 1
                if repeats >= step.max_repeats:
                    break
            duration = time.perf_counter() - start
            if self.hitl_manager:
                output = await self.hitl_manager.intercept(step, agent, output)
            return StepResult(
                step_id=step_id,
                output=output,
                duration=duration,
                usage=usage,
                cost_usd=cost,
                metadata={"repeats": repeats} if repeats else {},
            )

    def _build_prompt(
        self, step: Any, outputs: dict[str, StepResult], inputs: dict[str, str]
    ) -> str:
        context_parts = [f"Overall goal: {self.graph.goal}"]
        if step.inputs:
            for dep in step.inputs:
                context_parts.append(
                    f"Output of step '{self.graph.steps[dep].name}' ({dep}):\n{outputs[dep].output}"
                )
        for key, value in inputs.items():
            context_parts.append(f"Input '{key}': {value}")
        prompt = f"{context_parts[0]}\n\nTask for you ({step.name}): {step.instruction}"
        for part in context_parts[1:]:
            prompt += f"\n\n{part}"
        return prompt
