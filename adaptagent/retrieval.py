"""Okapi BM25 retrieval store for AdaptAgent - stdlib only, serverless-safe.

Pattern from Fase-1 analysis:
- smolagents RetrieverTool (src/smolagents/tools.py): a first-class agent tool
  whose input is a query string and whose output is the matching documents;
  retrieval runs explicitly inside the agent loop.
- LlamaIndex Retriever + Workflows RetrievedEvent (llamaindex/rag.ipynb):
  `index.as_retriever(similarity_top_k=k)` -> `retriever.retrieve(query)` ->
  nodes travel as context (`ctx.store.set("retrieve.nodes", nodes)`) into the
  next step's inputs. The agent answers grounded on those nodes.

Port: a pure-stdlib Okapi BM25 index over named documents. A workflow step may
be typed `retrieval`: its output is a query string; the executor runs that
query against a named index in the retrieval store, returning the top-k docs
as the step's output. The next step reads that output through its normal
inputs edges - the same RetrievedEvent -> next inputs hand-off.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)
_STOPWORDS = frozenset('about all also an and any are as at be because been before being both but by can could did do does doing during for from had has have having he her him his how i if in into is it its just me more most my no nor not of off on once only or other our out over own same she should so some such than that the their them there these they this those through to too under until up very was we were what when where which while who whom why will with would you your'.split())



def tokenize(text: str | None) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "text": self.text, "score": round(self.score, 4)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetrievedDoc":
        return cls(str(d.get("doc_id", "")), str(d.get("text", "")), float(d.get("score", 0.0)))


class BM25Index:
    """Okapi BM25 over named documents - pure stdlib, serverless-safe."""

    K1 = 1.5
    B = 0.75

    def __init__(self, name: str) -> None:
        self.name = name
        self._docs: dict[str, str] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self._docs[doc_id] = text
        self._tf[doc_id] = Counter(toks)
        for term in set(toks):
            self._df[term] += 1
        self._avgdl = self._recompute_avgdl()

    def add_many(self, documents: list[dict[str, Any]]) -> None:
        for d in documents or []:
            doc_id = str(d.get("id") or d.get("doc_id") or "")
            text = str(d.get("text") or "")
            if doc_id and text:
                self.add(doc_id, text)
        self._avgdl = self._recompute_avgdl()

    def _recompute_avgdl(self) -> float:
        lengths = [sum(v.values()) for v in self._tf.values()]
        return sum(lengths) / max(1, len(lengths))

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, k: int = 3, min_score: float = 0.0) -> list[RetrievedDoc]:
        qtoks = tokenize(query)
        if not qtoks or not self._docs:
            return []
        qtf = Counter(qtoks)
        avgdl = self._avgdl or 1e-9
        scored: list[RetrievedDoc] = []
        for doc_id, tf in self._tf.items():
            dl = sum(tf.values())
            s = 0.0
            for term, q in qtf.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.B + self.B * dl / avgdl
                s += self._idf(term) * (f * (self.K1 + 1)) / (f + self.K1 * norm) * q
            if s >= min_score:
                scored.append(RetrievedDoc(doc_id, self._docs[doc_id], s))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._docs)


class RetrievalStore:
    """Thread-safe registry of named BM25 indexes (mirrors CheckpointStore)."""

    def __init__(self) -> None:
        self._indexes: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def _cidx(self, name: str) -> BM25Index:
        with self._lock:
            idx = self._indexes.get(name)
        if idx is None:
            idx = BM25Index(name)
            with self._lock:
                self._indexes[name] = idx
        return idx

    def upsert(self, name: str, documents: list[dict[str, Any]]) -> BM25Index:
        idx = BM25Index(name)
        idx.add_many(documents or [])
        with self._lock:
            self._indexes[name] = idx
        return idx

    def add(self, name: str, documents: list[dict[str, Any]]) -> BM25Index:
        idx = self._cidx(name)
        idx.add_many(documents or [])
        with self._lock:
            self._indexes[name] = idx
        return idx

    def add_doc(self, name: str, doc_id: str, text: str) -> BM25Index:
        idx = self._cidx(name)
        idx.add(doc_id, text)
        with self._lock:
            self._indexes[name] = idx
        return idx

    def get(self, name: str) -> BM25Index | None:
        with self._lock:
            return self._indexes.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._indexes)

    def reset(self) -> None:
        with self._lock:
            self._indexes.clear()


_store: RetrievalStore | None = None
_store_lock = threading.Lock()


def get_store() -> RetrievalStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RetrievalStore()
    return _store


def reset_store() -> None:
    global _store
    with _store_lock:
        _store = None


def docs_to_context(docs: list[RetrievedDoc], *, max_chars: int = 4000) -> str:
    """Render retrieved docs as a grounded context block for the next step."""
    if not docs:
        return ""
    budget = max_chars // max(1, len(docs))
    pieces: list[str] = []
    for i, d in enumerate(docs, 1):
        text = d.text if len(d.text) <= budget else d.text[:budget] + "..."
        pieces.append(f"[{i}] (id {d.doc_id}, score {d.score:.3f})\n{text}")
    return "\n\n".join(pieces)
