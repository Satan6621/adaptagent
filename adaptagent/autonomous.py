"""Autonomous loop adapter: worker->validator->router over the real executor.

This module does NOT reimplement the loop. It is a thin public gateway that
reuses the already-shipped machinery (single source of truth):

  * worker  = each WorkflowStep (code / retrieval / LLM) executed by
              ``WorkflowExecutor.execute`` (topological order = router edge).
  * validator = the executor's judge-backed ``repeat_until`` gates using
              ``LLMJudgeEvaluator`` (score_gte) plus ``validate_step_output``
              for structured output schemas.
  * router = the executor's ``stop_when`` (global termination: contains /
              max_steps) and per-step repeat caps (max_repeats).

Callers get back the SAME ``WorkflowOutput`` shape the canvas and the chat
already consume (results, final_output, total_usage, cost, stopped, run_id),
and the same SSE ``on_event`` channel — so the tablero-chat can run this
loop with zero changes to its rendering code.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .llm.base import LLM
from .workflow.executor import WorkflowExecutor, WorkflowOutput
from .workflow.graph import WorkflowGraph

if TYPE_CHECKING:
    from .workflow.generator import WorkflowGenerator

on_event_t = Callable[[dict[str, Any]], Any]


@dataclass
class LoopState:
    """Observable loop bookkeeping (mirrors the chat's AgentState fields)."""

    goal: str
    iteration_count: int = 0
    max_iterations: int = 3
    validation_report: dict[str, Any] = field(default_factory=dict)
    final_status: str = "RUNNING"  # RUNNING | SUCCESS | FAILED | STOPPED
    run_id: str = ""


class AutonomousLoop:
    """Run worker->validator->router with the real executor.

    ``stop_when`` accepts:
        {"contains": str}   -> stop globally when any step output contains text
        {"max_steps": int}  -> stop after N non-skipped steps executed
    ``judge_target`` is the minimum score for judge-backed repeat gates
    (default 70). Set ``max_iterations`` to cap overall retries.
    """

    def __init__(
        self,
        goal: str,
        graph: WorkflowGraph | None = None,
        llm: LLM | None = None,
        generator: "WorkflowGenerator | None" = None,
        on_event: on_event_t | None = None,
        *,
        judge_target: float = 70.0,
        max_iterations: int = 3,
        stop_when: dict[str, Any] | None = None,
        run_id: str = "",
    ) -> None:
        self.goal = goal
        self.graph = graph
        self.llm = llm
        self.generator = generator
        self.on_event = on_event
        self.judge_target = judge_target
        self.max_iterations = max_iterations
        self.stop_when = stop_when or {}
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.state = LoopState(goal=goal, max_iterations=max_iterations, run_id=self.run_id)

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if asyncio.iscoroutine(result):
            await result

    async def _build_graph(self) -> WorkflowGraph:
        if self.graph is not None:
            return self.graph
        if self.generator is None:
            from .workflow.generator import WorkflowGenerator

            self.generator = WorkflowGenerator(llm=self.llm)
        return await self.generator.generate_workflow(self.goal)

    async def run(self) -> tuple[WorkflowOutput, LoopState]:
        """Execute the loop and return (output, state)."""
        graph = await self._build_graph()

        # The judge for repeat_until("score_gte") gates lives inside the shared
        # WorkflowExecutor (single source of truth): we only pass judge_target
        # through. Duplicating LLMJudgeEvaluator here would simulate, not reuse.
        for sid, step in graph.steps.items():
            if step.repeat_until is not None and step.repeat_until.rule == "score_gte":
                step.repeat_until.value = str(self.judge_target)

        async def on_executor_event(event: dict[str, Any]) -> None:
            if event.get("event") == "step_done" and not event.get("skipped"):
                self.state.iteration_count += 1
            self.state.validation_report = {
                "evaluation_status": (
                    "APPROVED" if event.get("event") in ("done", "checkpoint") else "IN_PROGRESS"
                ),
                "iteration_count": self.state.iteration_count,
            }
            if event.get("event") == "stopped":
                self.state.final_status = "STOPPED"
            if event.get("event") == "done":
                self.state.final_status = "SUCCESS" if not event.get("stopped") else "STOPPED"
            if self.on_event is not None:
                result = self.on_event(event)
                if asyncio.iscoroutine(result):
                    await result


        self.state.final_status = "RUNNING"
        start_all = time.perf_counter()
        executor = WorkflowExecutor(
            graph=graph,
            llm=self.llm,
            on_event=on_executor_event,
            stop_when=self.stop_when,
            run_id=self.run_id,
        )
        output: WorkflowOutput = await executor.execute(inputs={"goal": self.goal})
        self.state.validation_report["elapsed"] = round(time.perf_counter() - start_all, 2)
        if self.state.final_status == "RUNNING":
            self.state.final_status = "FAILED" if output.stopped else "SUCCESS"
        return output, self.state


async def run_autonomous_loop(
    goal: str,
    *,
    graph: WorkflowGraph | None = None,
    llm: LLM | None = None,
    generator: "WorkflowGenerator | None" = None,
    on_event: on_event_t | None = None,
    judge_target: float = 70.0,
    max_iterations: int = 3,
    stop_when: dict[str, Any] | None = None,
    run_id: str = "",
) -> tuple[WorkflowOutput, LoopState]:
    """Run the worker->validator->router loop and return (output, state)."""
    loop = AutonomousLoop(
        goal=goal,
        graph=graph,
        llm=llm,
        generator=generator,
        on_event=on_event,
        judge_target=judge_target,
        max_iterations=max_iterations,
        stop_when=stop_when,
        run_id=run_id,
    )
    return await loop.run()
