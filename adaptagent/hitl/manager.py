"""HITL: human-in-the-loop interception of agent outputs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.agent import Agent
    from ..workflow.graph import WorkflowStep


class HITLMode(str, Enum):
    PRE_EXECUTION = "pre"
    POST_EXECUTION = "post"


class HITLInteractionType(str, Enum):
    APPROVE_REJECT = "approve_reject"
    EDIT = "edit"
    INPUT = "input"


@dataclass
class HITLDecision:
    approved: bool
    edited_output: str | None = None
    feedback: str | None = None


class HITLManager:
    """Central manager for human interactions. Disabled by default."""

    def __init__(self, enabled: bool = False, auto_approve: bool = False) -> None:
        self.enabled = enabled
        self.auto_approve = auto_approve  # for tests / CI
        self.interactions: int = 0

    def activate(self) -> None:
        self.enabled = True

    def deactivate(self) -> None:
        self.enabled = False

    async def intercept(self, step: "WorkflowStep", agent: "Agent", output: str) -> str:
        """Optionally pause for human review of a step's output."""
        if not self.enabled:
            return output
        self.interactions += 1
        if self.auto_approve:
            print(f"[hitl] auto-approved step '{step.name}'")
            return output
        print(f"\n[hitl] Review step '{step.name}' (agent '{agent.name}'):")
        print("-" * 60)
        print(output[:1000])
        print("-" * 60)
        decision = await self._ask_console()
        if decision.approved and decision.edited_output:
            return decision.edited_output
        if decision.approved:
            return output
        return f"[rejected by human] {decision.feedback or 'no feedback provided'}"

    async def _ask_console(self) -> HITLDecision:
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None, lambda: input("[a]pprove / [e]dit / [r]eject: ").strip().lower()
        )
        if answer.startswith("a"):
            return HITLDecision(approved=True)
        if answer.startswith("e"):
            edited = await loop.run_in_executor(None, lambda: input("Edited output: "))
            return HITLDecision(approved=True, edited_output=edited)
        feedback = await loop.run_in_executor(None, lambda: input("Rejection reason: "))
        return HITLDecision(approved=False, feedback=feedback)
