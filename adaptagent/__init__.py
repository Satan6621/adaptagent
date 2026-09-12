"""AdaptAgent: a lightweight, strongly-typed framework for building, evaluating
and self-evolving LLM agent workflows."""

from .config import _load_env_file

_load_env_file()

__version__ = "0.2.0"

from .agents import Agent, AgentManager  # noqa: E402
from .hitl import HITLManager  # noqa: E402
from .llm import LLM, LLMConfig, LLMResponse, resolve_llm  # noqa: E402
from .memory import InMemoryStore, FileMemoryStore  # noqa: E402
from .tools import Tool, tool  # noqa: E402
from .workflow import Workflow, WorkflowGenerator, WorkflowGraph  # noqa: E402
from .evolution import EvolutionEngine, Evaluator, LLMJudgeEvaluator  # noqa: E402

__all__ = [
    "Agent",
    "AgentManager",
    "HITLManager",
    "LLM",
    "LLMConfig",
    "LLMResponse",
    "resolve_llm",
    "InMemoryStore",
    "FileMemoryStore",
    "Tool",
    "tool",
    "Workflow",
    "WorkflowGenerator",
    "WorkflowGraph",
    "EvolutionEngine",
    "Evaluator",
    "LLMJudgeEvaluator",
    "__version__",
]
