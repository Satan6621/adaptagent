"""BM25 retrieval for AdaptAgent — stdlib only, serverless-safe.

Pattern sources (both ported as the "retrieval node"):
- smolagents `RetrieverTool` (src/smolagents/tools.py 664/668): a tool whose
  `__call__(query) -> documents` runs a search query over an index and returns
  matching documents; the agent invokes it like any other tool.
- LlamaIndex Retriever (llama-index-core/retrievers + workflows rag.ipynb):
  `as_retriever(similarity_top_k=k)` then `RetrievedEvent(retrieved_nodes=...)`
  — the workflow step "retrieves" on the user query and hands the nodes to the
  next step as context.

Port: a BM25 (Okapi) index over named documents, exposed as a **retrieval step**
in the DAG. The LLM writes the step output as a `query` (or the step reads a
`query_field` input); the executor queries the named index and returns the
top-k documents as the step output, which later steps consume through regular
`inputs` edges. No heavy deps, no vector DB, no network calls at query time —
pure python math, safe to run inside a Vercel serverless function.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her his i if in is it "
    "its just me my no not of on or our out over said she so some such than that "
    "the their them then there these they this those through to too under up was we "
    "were what when where which who why will with would you your"
).split()


def _tokenize(text: str) -> list[str]:
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
    """Okapi BM25 over a set of named documents (stdlib-only)."""

    K1 = 1.5
    B = 0.75

    def __init__(self, name: str) -> None:
        self.name = name
        self._docs: dict[str, str] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    # -- ingestion ---------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        toks = _tokenize(text)
        self._docs[doc_id] = text
        self._tf[doc_id] = Counter(toks)
        for term in set(toks):
            self._df[term] += 1
        n = len(self._docs)
        self._avgdl = sum(sum(t.values()) for t in self._tf.values()) / max(1, n)

    def add_many(self, documents: list[dict[str, Any]]) -> None:
        for d in documents or []:
            did = str(d.get("doc_id") or d.get("id") or "")
            text = str(d.get("text") or d.get("content") or "")
            if did and text:
                self.add(did, text)

    # -- querying ----------------------------------------------------------
    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, k: int = 3, min_score: float = 0.0) -> list[RetrievedDoc]:
        qtoks = _tokenize(query)
        if not qtoks or not self._docs:
            return []
        scored: list[RetrievedDoc] = []
        for doc_id, text in self._docs.items():
            tf = self._tf[doc_id]
            dl = sum(tf.values())
            s = 0.0
            qtf = Counter(qtoks)
            for term, qn in qtf.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.K1 * (1 - self.B + self.B * dl / max(self._avgdl, 1e-9))
                s += self._idf(term) * (f * (self.K1 + 1)) / denom * qn
            if s >= min_score:
                scored.append(RetrievedDoc(doc_id, text, s))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "avgdl": round(self._avgdl, 3),
            "n_docs": len(self._docs),
        }

    def __len__(self) -> int:
        return len(self._docs)


class RetrievalStore:
    """Thread-safe registry of named BM25 indexes (memory) — mirrors CheckpointStore."""

    def __init__(self) -> None:
        self._indexes: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def upsert(self, name: str, documents: list[dict[str, Any]]) -> BM25Index:
        idx = self._indexes.get(name) or BM25Index(name)
        idx.add_many(documents or [])
        with self._lock:
            self._indexes[name] = idx
        return idx

    def add(self, name: str, doc_id: str, text: str) -> None:
        idx = self._indexes.get(name) or BM25Index(name)
        idx.add(doc_id, text)
        with self._lock:
            self._indexes[name] = idx

    def get(self, name: str) -> BM25Index | None:
        return self._indexes.get(name)

    def names(self) -> list[str]:
        return sorted(self._indexes)

    def reset(self) -> None:
        with self._lock:
            self._indexes.clear()


_store: RetrievalStore | None = None


def get_store() -> RetrievalStore:
    global _store
    if _store is None:
        _store = RetrievalStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None
