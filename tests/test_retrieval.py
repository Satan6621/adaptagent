# -*- coding: utf-8 -*-
"""Fase 2a - grounded BM25 retrieval as a first-class workflow step.

Contract ported byte-by-byte from the REAL public API read off disk this
session (adaptagent/retrieval.py + workflow/graph.py):

  from adaptagent.retrieval import docs_to_context, get_store, reset_store
  from adaptagent.workflow.graph import RetrievalSpec

This mirrors smolagents RetrieverTool + LlamaIndex retrieve-node: grounded
BM25 over the shared store, docs rendered via docs_to_context as step output.
No LLM, no tokens, no cost -- serverless-safe port (the grounding contract).
"""
from __future__ import annotations

from adaptagent.retrieval import docs_to_context, get_store, reset_store
from adaptagent.workflow.graph import RetrievalSpec


def test_retrieval_spec_roundtrip() -> None:
    spec = RetrievalSpec(index="kb", query="bm25 serverless", k=2, min_score=0.1)
    data = spec.to_dict()
    again = RetrievalSpec.from_dict(data)
    assert again.to_dict() == data
    assert again.index == "kb"
    assert again.query == "bm25 serverless"
    assert again.k == 2


def test_store_add_search_grounds_docs() -> None:
    reset_store()
    store = get_store()
    store.add(
        "kb",
        [
            {"id": "a", "text": "AdaptAgent runs grounded BM25 retrieval with zero LLM cost"},
            {"id": "b", "text": "The executor dispatches retrieval steps server-side"},
        ],
    )
    idx = store.get("kb")
    assert idx is not None
    hits = idx.search("grounded retrieval", k=2)
    assert hits
    assert len(hits) >= 2
    reset_store()


def test_docs_to_context_renders_plain_text_with_directory() -> None:
    reset_store()
    store = get_store()
    store.add("kb", [{"id": "x", "text": "docs to context renders grounded text"}])
    idx = store.get("kb")
    assert idx is not None
    ctx = docs_to_context(idx.search("grounded", k=1))
    assert isinstance(ctx, str)
    assert "grounded" in ctx
    reset_store()


def test_reset_store_clears_all() -> None:
    get_store().add("tmp", [{"id": "z", "text": "temp doc"}])
    assert get_store().names()
    reset_store()
    assert get_store().get("tmp") is None  # empty store -> get() returns None
