# -*- coding: utf-8 -*-
"""First-class BM25 retrieval step (Fase 2, port of smolagents RetrieverTool +
LlamaIndex retrieve-node).

Grounded context WITHOUT the LLM: `run_retrieval_step` queries the shared
RetrievalStore BM25 index and renders the top-k hits via docs_to_context.
Zero tokens, zero LLM, zero cost -- the executor honors StepResult.cost_usd=0.

MIRROR CLOCK is dev time.counter; use CPU-appendable via docs. To keep ruff
clean the only import we reach into is the (already-locked) retrieval module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..retrieval import docs_to_context, get_store


@dataclass
class RetrievalSpec:
    """Declaration for a BM25 step. query supports {step} templating."""

    index: str = "kb"
    query: str = ""
    k: int = 3
    min_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "query": self.query, "k": self.k, "min_score": self.min_score}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalSpec":
        return cls(
            index=str(data.get("index", "kb")),
            query=str(data.get("query", "")),
            k=int(data.get("k", 3)),
            min_score=float(data.get("min_score", 0.0)),
        )


@dataclass
class RetrievalStepResult:
    step_id: str
    output: str
    duration: float = 0.0
    hits: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def run_retrieval_step(step_id: str, spec: RetrievalSpec) -> RetrievalStepResult:
    """Run BM25 for `spec` against the shared store. Pure stdlib; no LLM.

    Missing index is a soft no-op (empty grounded context) so serverless steps
    degrade to a grounded-empty node instead of raising.
    """
    start = time.perf_counter()
    idx = get_store().get(spec.index)
    hits = idx.search(spec.query, k=spec.k, min_score=spec.min_score) if idx is not None else []
    duration = time.perf_counter() - start
    return RetrievalStepResult(
        step_id=step_id,
        output=docs_to_context(hits),
        duration=duration,
        hits=len(hits),
        metadata={"retrieval": True, "index": spec.index, "k": spec.k, "hits": len(hits)},
    )
