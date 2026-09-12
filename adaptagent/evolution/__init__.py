"""⭐ Evolution Engine: the AdaptAgent differentiator.

Evaluates workflows and evolves agent prompts with an evolutionary algorithm
(EvoPrompt-style: mutation + crossover driven by an LLM, selection by score).
Evolution is a first-class citizen: any Workflow can be optimized in place.
"""

from .aflow import AFlowEngine, AFlowResult
from .engine import EvolutionEngine, EvolutionResult
from .evaluator import Evaluator, LLMJudgeEvaluator
from .textgrad import TextGradOptimizer, TextGradResult

__all__ = [
    "AFlowEngine",
    "AFlowResult",
    "EvolutionEngine",
    "EvolutionResult",
    "Evaluator",
    "LLMJudgeEvaluator",
    "TextGradOptimizer",
    "TextGradResult",
]
