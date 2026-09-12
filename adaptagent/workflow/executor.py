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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowOutput:
    goal: str
    results: dict[str, StepResult] = field(default_factory=dict)
    final_output: str = ""

    def __getitem__(self, step_id: str) -> StepResult:
        return self.results[step_id]


class Workflow:
    """Orchestrates execution of a WorkflowGraph with AgentManager agents.

    on_event: optional async callback receiving execution events:
        {"event": "step_start", "step_id", "name"}
        {"event": "step_done", "step_id", "name", "duration", "output"}
        {"event": "done", "final_output", "elapsed"}
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

    async def execute(self, inputs: dict[str, str] | None = None) -> WorkflowOutput:
        inputs = inputs or {}
        start_all = time.perf_counter()
        outputs: dict[str, StepResult] = {}
        order = self.graph.topo_order()
        remaining = list(order)
        pending: dict[str, asyncio.Task] = {}
        while remaining or pending:
            # launch all steps whose dependencies are satisfied
            launchable = [
                sid for sid in remaining if all(dep in outputs for dep in self.graph.steps[sid].inputs)
            ]
            for sid in launchable:
                remaining.remove(sid)
                pending[sid] = asyncio.create_task(self._run_step(sid, outputs, inputs))
            if not pending:
                break
            done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # find which step finished
                for sid, t in list(pending.items()):
                    if t is task:
                        outputs[sid] = task.result()
                        del pending[sid]
                        await self._emit(
                            {
                                "event": "step_done",
                                "step_id": sid,
                                "name": self.graph.steps[sid].name,
                                "duration": round(outputs[sid].duration, 1),
                                "output": outputs[sid].output,
                            }
                        )
                        break
        elapsed = time.perf_counter() - start_all
        final_sid = order[-1] if order else None
        final = outputs.get(final_sid).output if final_sid and final_sid in outputs else ""
        await self._emit({"event": "done", "final_output": final, "elapsed": round(elapsed, 1)})
        return WorkflowOutput(goal=self.graph.goal, results=outputs, final_output=final)

    async def _run_step(
        self, step_id: str, outputs: dict[str, StepResult], inputs: dict[str, str]
    ) -> StepResult:
        step = self.graph.steps[step_id]
        await self._emit({"event": "step_start", "step_id": step_id, "name": step.name})
        async with self._semaphore:
            start = time.perf_counter()
            context_parts = [f"Overall goal: {self.graph.goal}"]
            if step.inputs:
                for dep in step.inputs:
                    context_parts.append(
                        f"Output of step '{self.graph.steps[dep].name}' ({dep}):\n{outputs[dep].output}"
                    )
            for key, value in inputs.items():
                context_parts.append(f"Input '{key}': {value}")
            prompt = f"{context_parts[0]}\n\nYou are executing step '{step.name}'.\nInstruction: {step.instruction}"
            for part in context_parts[1:]:
                prompt += f"\n\n{part}"
            agent = self.agent_manager.get_agent_for_step(step)
            output = await agent.arun(prompt)
            duration = time.perf_counter() - start
            if self.hitl_manager:
                output = await self.hitl_manager.intercept(step, agent, output)
            return StepResult(step_id=step_id, output=output, duration=duration)
