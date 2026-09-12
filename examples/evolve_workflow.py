"""Full demo: goal -> workflow with tools -> execution -> self-evolution.

Usage: python examples/evolve_workflow.py
"""

import asyncio

from adaptagent import (
    AgentManager,
    EvolutionEngine,
    LLMConfig,
    Workflow,
    WorkflowGenerator,
    resolve_llm,
)
from adaptagent.tools import DDGSSearchTool, WikipediaSearchTool


async def main() -> None:
    llm = resolve_llm(LLMConfig())
    tools = [WikipediaSearchTool(), DDGSSearchTool()]

    goal = "Explain what quantum computing is, with a real-world application example."
    graph = await WorkflowGenerator(llm, tools=tools).generate_workflow(goal)
    print(graph.display())
    graph.save("workflow.json")

    manager = AgentManager(tools=tools)
    manager.register_llm(llm)
    manager.build_agents_from_workflow(graph)
    print(manager.display())

    # 1. baseline run
    workflow = Workflow(graph=graph, agent_manager=manager, llm=llm)
    output = await workflow.execute()
    print(f"\nBaseline final output:\n{output.final_output[:500]}\n")

    # 2. evolve
    engine = EvolutionEngine(graph=graph, agent_manager=manager, llm=llm)
    result = await engine.evolve(generations=3, population=4)
    print(result)

    # 3. rerun with evolved prompts
    output2 = await workflow.execute()
    print(f"\nEvolved final output:\n{output2.final_output[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
