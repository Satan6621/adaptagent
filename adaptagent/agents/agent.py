"""A ReAct-style agent with a typed tool loop."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..llm.schemas import get_tool_schemas as _default_schemas  # noqa: F401 (re-exported via mcp)
from ..mcp.client import get_tool_schemas_for as get_tool_schemas

if TYPE_CHECKING:
    from ..llm.base import LLM
    from ..memory.base import Memory
    from ..tools.base import Tool

SYSTEM_TEMPLATE = """You are {name}, {role}.

{instructions}"""

TOOL_RULES = """
You can use tools. Available tools:
{tool_docs}

Tool usage protocol:
1. Reason about what to do next (Thought).
2. If a tool helps, output a tool call; otherwise answer directly.
3. Tool results appear as user messages prefixed with "Observation:".
4. Iterate until you can produce the FINAL ANSWER, then output it (plain text, no prefix)."""

MAX_ITERATIONS = 12


@dataclass
class Agent:
    """A single LLM-driven agent with tools, memory and an evolution-aware prompt."""

    name: str
    role: str
    instruction: str = ""
    llm: "LLM | None" = None
    tools: list["Tool"] = field(default_factory=list)
    memory: "Memory | None" = None
    max_iterations: int = MAX_ITERATIONS

    def __post_init__(self) -> None:
        self._conversations: dict[str, list[dict[str, str]]] = {}

    @property
    def system_prompt(self) -> str:
        prompt = SYSTEM_TEMPLATE.format(name=self.name, role=self.role, instructions=self.instruction)
        if self.tools:
            tool_docs = "\n".join(
                f"- {t.name}: {t.description}" for t in self.tools
            )
            prompt += TOOL_RULES.format(tool_docs=tool_docs)
        if self.memory:
            memories = self.memory.recall()
            if memories:
                prompt += f"\n\nRelevant long-term memories:\n{memories}"
        return prompt

    def clone(self, instruction: str | None = None) -> "Agent":
        """Create a copy (used by the evolution engine to try prompt variants)."""
        return Agent(
            name=self.name,
            role=self.role,
            instruction=instruction if instruction is not None else self.instruction,
            llm=self.llm,
            tools=list(self.tools),
            memory=self.memory,
            max_iterations=self.max_iterations,
        )

    async def arun(self, prompt: str, session_id: str = "default") -> str:
        if not self.llm:
            raise ValueError(f"Agent '{self.name}' has no LLM configured")
        messages = self._conversations.setdefault(session_id, [])
        messages.append({"role": "user", "content": prompt})
        schemas = get_tool_schemas(self.tools) if self.tools else None
        tool_map = {t.name: t for t in self.tools}
        iterations = 0
        while True:
            iterations += 1
            if iterations > self.max_iterations:
                messages.append({"role": "assistant", "content": "FINAL ANSWER: max iterations reached."})
                return "max iterations reached"
            response = await self.llm.generate(
                messages=self._system_messages() + messages, tools=schemas
            )
            if not response.has_tool_calls:
                messages.append({"role": "assistant", "content": response.text})
                if self.memory:
                    await self._maybe_store(prompt, response.text)
                return response.text
            for call in response.tool_calls:
                observation = await self._execute_tool(call, tool_map)
                messages.append(
                    {"role": "assistant", "content": f"[tool call {call.name}({call.arguments})]"}
                )
                messages.append({"role": "user", "content": f"Observation: {observation}"})

    def run(self, prompt: str, session_id: str = "default") -> str:
        return asyncio.run(self.arun(prompt, session_id))

    async def _execute_tool(self, call: Any, tool_map: dict[str, "Tool"]) -> str:
        tool = tool_map.get(call.name)
        if tool is None:
            return f"Error: unknown tool '{call.name}'"
        try:
            result = tool.fn(**call.arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Error calling tool '{call.name}': {exc}"

    async def _maybe_store(self, prompt: str, response: str) -> None:
        """Store significant exchanges into long-term memory (simple heuristic)."""
        try:
            if not self.memory:
                return
            if len(response) > 100:
                await self.memory.store_async(f"Q: {prompt[:200]}\nA: {response[:500]}")
        except Exception:
            pass

    def _system_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}]
