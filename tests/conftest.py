"""Shared test fixtures: FakeLLM and helpers."""

import pytest

from adaptagent.llm.base import LLMConfig, LLMResponse
from adaptagent.workflow import WorkflowGraph, WorkflowStep


class FakeLLM:
    """Deterministic fake LLM: pops responses in order, repeats last one."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["ok"]
        self._i = 0
        self.config = LLMConfig()
        self.calls: list[dict] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        text = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return LLMResponse(text=text)

    def tool_prompt(self, schemas):
        return {}

    async def close(self):
        pass


def make_graph() -> WorkflowGraph:
    return WorkflowGraph(
        goal="test goal",
        steps=[
            WorkflowStep(id="step1", name="Research", instruction="Do research", inputs=[], agent_hint="researcher"),
            WorkflowStep(
                id="step2", name="Write", instruction="Write summary", inputs=["step1"], agent_hint="writer"
            ),
        ],
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
