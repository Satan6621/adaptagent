"""AgentManager: registry of agents + assignment of steps to agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..llm.base import LLMConfig
from ..llm.registry import resolve_llm
from ..workflow.graph import WorkflowStep
from .agent import Agent

if TYPE_CHECKING:
    from ..llm.base import LLM
    from ..tools.base import Tool


class AgentManager:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.agents: dict[str, Agent] = {}
        self.tools: list[Tool] = tools or []
        self._default_llm: LLM | None = None
        self._llm_config: LLMConfig | None = None

    def add_agent(self, agent: Agent, make_default_llm: bool = True) -> None:
        if agent.llm is None and make_default_llm and self._default_llm:
            agent.llm = self._default_llm
        self.agents[agent.name] = agent

    def register_llm(self, llm: "LLM", config: LLMConfig | None = None) -> None:
        self._default_llm = llm
        self._llm_config = config
        for agent in self.agents.values():
            if agent.llm is None:
                agent.llm = llm

    def build_agents_from_workflow(
        self, graph, llm_config: LLMConfig | None = None, assign_tools: bool = True
    ) -> None:
        """Create one agent per step using agent_hint as role."""
        config = llm_config or self._llm_config
        for step in graph.steps.values():
            role = step.agent_hint or "assistant"
            name = f"{step.name} ({step.id})"
            agent = Agent(name=name, role=role, instruction=step.instruction)
            if config is not None:
                agent.llm = resolve_llm(config)
            else:
                agent.llm = self._default_llm
            if assign_tools and self.tools:
                agent.tools = list(self.tools)
            self.add_agent(agent)

    def get_agent_for_step(self, step: WorkflowStep) -> Agent:
        name = f"{step.name} ({step.id})"
        if name in self.agents:
            return self.agents[name]
        # fallback: nearest role match
        if step.agent_hint:
            for agent in self.agents.values():
                if agent.role == step.agent_hint:
                    return agent
        if self.agents:
            return next(iter(self.agents.values()))
        raise ValueError("No agents registered; call build_agents_from_workflow or add_agent first")

    def display(self) -> str:
        lines = [f"AgentManager: {len(self.agents)} agents"]
        for agent in self.agents.values():
            tool_names = [t.name for t in agent.tools]
            lines.append(f"  - {agent.name} | role={agent.role} | tools={tool_names}")
        return "\n".join(lines)
