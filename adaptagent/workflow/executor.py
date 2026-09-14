"""Executes a WorkflowGraph step-by-step using agents from an AgentManager.

Supports structured outputs (output_schema validation with bounded retry),
judge-backed loops (repeat_until rule "score_gte"), global termination
conditions (stop_when) and checkpointing/resume.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..retrieval import docs_to_context, get_store
from ..structured import schema_hint, validate_step_output
from ..tools.python_repl import python_repl
from .graph import WorkflowGraph

if TYPE_CHECKING:
    from ..agents.manager import AgentManager
    from ..checkpoint import Checkpoint, CheckpointStore
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
    stopped: str | None = None
    run_id: str = ""

    def __getitem__(self, step_id: str) -> StepResult:
        return self.results[step_id]


class Workflow:
    """Orchestrates execution of a WorkflowGraph with AgentManager agents.

    on_event: optional async callback receiving execution events:
        {"event": "step_start", "step_id", "name"}
        {"event": "step_done", "step_id", "name", "duration", "output", "usage", "cost_usd"}
        {"event": "checkpoint", "checkpoint": {...}}
        {"event": "step_skipped", "step_id", "name"}
        {"event": "stopped", "reason"}
        {"event": "done", "final_output", "elapsed", "cost_usd", "usage", "stopped", "run_id"}
        {"event": "error", "error"}

    stop_when: optional dict with global termination conditions:
        {"contains": "text"}  -> stop when any step output contains text
        {"max_steps": int}    -> stop after N non-skipped steps executed
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        agent_manager: "AgentManager",
        llm: "LLM",
        hitl_manager: "HITLManager | None" = None,
        max_parallel: int = 4,
        on_event: Callable[[dict], Any] | None = None,
        stop_when: dict[str, Any] | None = None,
        checkpoint: "Checkpoint | None" = None,
        checkpoint_store: "CheckpointStore | None" = None,
        run_id: str = "",
    ) -> None:
        self.graph = graph
        self.agent_manager = agent_manager
        self.llm = llm
        self.hitl_manager = hitl_manager
        self.max_parallel = max_parallel
        self.on_event = on_event
        self.stop_when = stop_when or {}
        self.checkpoint = checkpoint
        self.checkpoint_store = checkpoint_store
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.stopped_reason: str | None = None
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

    def _check_stop(self, outputs: dict[str, StepResult]) -> str | None:
        """Global termination: return a stop reason or None."""
        if self.stopped_reason:
            return self.stopped_reason
        contains = self.stop_when.get("contains")
        if contains:
            needle = str(contains).lower()
            if any(needle in (r.output or "").lower() for r in outputs.values() if not r.skipped):
                self.stopped_reason = f"sentinel '{contains}' found in a step output"
                return self.stopped_reason
        max_steps = self.stop_when.get("max_steps")
        if max_steps:
            executed = sum(
                1 for r in outputs.values() if not r.skipped and not r.metadata.get("resumed")
            )
            if executed >= int(max_steps):
                self.stopped_reason = f"max_steps ({max_steps}) reached"
                return self.stopped_reason
        return None

    async def _emit_checkpoint(self, order: list[str], outputs: dict[str, StepResult]) -> None:
        if self.checkpoint_store is None:
            return
        from ..checkpoint import checkpoint_from

        cp = checkpoint_from(self.run_id, self.graph.goal, order, outputs)
        self.checkpoint_store.save(cp)
        await self._emit({"event": "checkpoint", "checkpoint": cp.to_dict()})

    async def execute(self, inputs: dict[str, str] | None = None) -> WorkflowOutput:
        inputs = inputs or {}
        start_all = time.perf_counter()
        totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        total_cost = 0.0
        order = self.graph.topo_order()
        outputs: dict[str, StepResult] = {}
        if self.checkpoint is not None:
            for sid in order:
                if sid in self.checkpoint.step_outputs:
                    outputs[sid] = StepResult(
                        step_id=sid, output=self.checkpoint.step_outputs[sid], metadata={"resumed": True}
                    )
                elif sid in self.checkpoint.skipped:
                    outputs[sid] = StepResult(step_id=sid, output="", skipped=True, metadata={"resumed": True})
            if any(r.metadata.get("resumed") for r in outputs.values()):
                await self._emit({"event": "resumed", "steps": sorted(outputs)})
        remaining = [sid for sid in order if sid not in outputs]
        pending: dict[str, asyncio.Task] = {}
        while (remaining or pending) and not self.stopped_reason:
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
                        await self._emit_checkpoint(order, outputs)
                        if self._check_stop(outputs):
                            break
                        break
            if self.stopped_reason:
                break
        if self.stopped_reason:
            for sid in order:
                if sid not in outputs:
                    outputs[sid] = StepResult(
                        step_id=sid, output="", skipped=True, metadata={"stopped": self.stopped_reason}
                    )
            await self._emit({"event": "stopped", "reason": self.stopped_reason})
        for task in pending.values():  # cancel any in-flight steps after a global stop
            task.cancel()
        if pending:
            await asyncio.gather(*pending.values(), return_exceptions=True)
        elapsed = time.perf_counter() - start_all
        # final output: last non-skipped step in topological order
        final = ""
        for sid in order:
            r = outputs.get(sid)
            if r and not r.skipped and r.output:
                final = r.output
        await self._emit_checkpoint(order, outputs)
        await self._emit(
            {
                "event": "done",
                "final_output": final,
                "elapsed": round(elapsed, 1),
                "cost_usd": round(total_cost, 6),
                "usage": totals,
                "stopped": self.stopped_reason,
                "run_id": self.run_id,
            }
        )
        return WorkflowOutput(
            goal=self.graph.goal,
            results=outputs,
            final_output=final,
            total_cost_usd=total_cost,
            total_usage=totals,
            stopped=self.stopped_reason,
            run_id=self.run_id,
        )

    async def _judge_repeat(self, condition: Any, output: str) -> tuple[bool, str]:
        """Judge-backed repeat gate (rule score_gte): score the output with an LLM."""
        target = float(condition.value or "70")
        from ..evolution.evaluator import LLMJudgeEvaluator

        judge = LLMJudgeEvaluator(self.llm)
        score = await judge.evaluate_async(self.graph.goal, output)
        ok = score >= target
        feedback = (
            f"Tu último intento obtuvo {score:.0f}/100 (se requiere ≥ {target:g}). "
            "Mejora el resultado y vuélvelo a intentar."
        )
        return ok, feedback

    async def _check_repeat(self, step: Any, output: str) -> tuple[bool, str | None]:
        if step.repeat_until is None:
            return True, None
        if step.repeat_until.rule == "score_gte":
            return await self._judge_repeat(step.repeat_until, output)
        ok = step.repeat_until.evaluate(output)
        if ok:
            return True, None
        return False, (
            f"Repite: tu última respuesta no cumplió la condición del bucle "
            f"(regla={step.repeat_until.rule}, valor={step.repeat_until.value!r})."
        )

    async def _run_code_step(self, step: Any, step_id: str) -> StepResult:
        """Run sandboxed code as a first-class workflow step.

        Port of the smolagents CodeAgent exec-node: run `code.source` in
        the shared sandboxed python_repl. No agent arun, no LLM, no
        tokens, no cost -- serverless-safe (mirror of _run_retrieval_step).
        """
        start = time.perf_counter()
        spec: Any = step.code
        import asyncio

        output = await asyncio.to_thread(python_repl, spec.source)
        elapsed = time.perf_counter() - start
        return StepResult(
            step_id=step_id,
            output=output,
            duration=elapsed,
            usage={},
            cost_usd=0.0,
            metadata={"code": True, "language": spec.language, "timeout_s": spec.timeout_s},
        )


    async def _run_retrieval_step(self, step: Any, step_id: str) -> StepResult:
        """Grounded BM25 retrieval as a first-class workflow step.

        Port of smolagents RetrieverTool + LlamaIndex retrieve-node:
        query the shared BM25 index, render top-k via docs_to_context.
        No agent arun, no LLM, no tokens, zero cost. Serverless-safe.
        """
        start = time.perf_counter()
        spec: Any = step.retrieval
        idx: Any = get_store().get(spec.index)
        docs = idx.search(spec.query, k=spec.k, min_score=spec.min_score) if idx is not None else []
        elapsed = time.perf_counter() - start
        return StepResult(
            step_id=step_id,
            output=docs_to_context(docs),
            duration=elapsed,
            usage={},
            cost_usd=0.0,
            metadata={"retrieval": True, "index": spec.index, "hits": len(docs)},
        )


    async def _run_step(
        self, step_id: str, outputs: dict[str, StepResult], inputs: dict[str, str]
    ) -> StepResult:
        step = self.graph.steps[step_id]
        await self._emit({"event": "step_start", "step_id": step_id, "name": step.name})
        async with self._semaphore:
            start = time.perf_counter()
            agent = self.agent_manager.get_agent_for_step(step)
            # STEP_RETRIEVAL_DISPATCH: grounded BM25 step -- no LLM, no cost
            if step.retrieval is not None:
                return await self._run_retrieval_step(step, step_id)
            if step.code is not None:
                return await self._run_code_step(step, step_id)
                return await self._run_retrieval_step(step, step_id)
            usage: dict[str, int] = {}
            cost = 0.0
            output = ""
            repeats = 0
            feedback = ""
            parsed = None
            validation_errors: list[str] = []
            while True:
                prompt = self._build_prompt(step, outputs, inputs, feedback=feedback)
                output = await agent.arun(prompt)
                usage = dict(agent.last_usage)
                cost += agent.last_cost_usd
                parsed, validation_errors = validate_step_output(step.output_schema, output) if step.output_schema else (None, [])
                repeat_ok = True
                if not validation_errors and step.repeat_until is not None:
                    repeat_ok, repeat_feedback = await self._check_repeat(step, output)
                if validation_errors:
                    feedback = (
                        "Tu último intento NO pasó la validación de salida:\n- "
                        + "\n- ".join(validation_errors)
                        + "\nCorrige y responde de nuevo siguiendo el esquema."
                    )
                elif step.repeat_until is None or repeat_ok:
                    break
                else:
                    feedback = repeat_feedback or ""
                repeats += 1
                if repeats >= step.max_repeats:
                    break
            duration = time.perf_counter() - start
            if self.hitl_manager:
                output = await self.hitl_manager.intercept(step, agent, output)
            metadata: dict[str, Any] = {}
            if repeats:
                metadata["repeats"] = repeats
            if validation_errors:
                metadata["validation_errors"] = validation_errors
            if parsed is not None:
                metadata["json"] = parsed
            return StepResult(
                step_id=step_id,
                output=output,
                duration=duration,
                usage=usage,
                cost_usd=cost,
                metadata=metadata,
            )

    def _build_prompt(
        self,
        step: Any,
        outputs: dict[str, StepResult],
        inputs: dict[str, str],
        feedback: str = "",
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
        if step.output_schema:
            prompt += f"\n\n{schema_hint(step.output_schema)}"
        if step.repeat_until is not None and step.repeat_until.rule == "score_gte":
            prompt += (
                f"\n\nTu respuesta será evaluada por un juez (0-100). "
                f"Debe alcanzar como mínimo {float(step.repeat_until.value or '70'):g}."
            )
        if feedback:
            prompt += f"\n\n--- FEEDBACK DEL INTENTO ANTERIOR ---\n{feedback}"
        return prompt
