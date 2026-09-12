"""Quickstart: generate and run a workflow from a natural-language goal.

Usage: python examples/quickstart.py
Requires: OPENAI_API_KEY in env or .env (or ADAPTAGENT_BASE_URL for local models).
"""

import asyncio

from adaptagent import AgentManager, LLMConfig, Workflow, WorkflowGenerator, resolve_llm


async def main() -> None:
    llm = resolve_llm(LLMConfig())  # provider=openai, model=gpt-4o-mini by default

    goal = "Research the latest trends in AI agents and write a one-paragraph executive summary."
    graph = await WorkflowGenerator(llm).generate_workflow(goal)
    print(graph.display())

    manager = AgentManager()
    manager.register_llm(llm)
    manager.build_agents_from_workflow(graph, assign_tools=False)
    print(manager.display())

    workflow = Workflow(graph=graph, agent_manager=manager, llm=llm)
    output = await workflow.execute()
    print("\n=== FINAL OUTPUT ===")
    print(output.final_output)


if __name__ == "__main__":
    asyncio.run(main())
