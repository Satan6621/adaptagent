"""Memory protocol + simple implementations (short-term & long-term)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """Interface for long-term memory stores."""

    def recall(self, query: str = "", limit: int = 5) -> str: ...

    async def store_async(self, item: str) -> None: ...


class InMemoryStore:
    """Ephemeral memory: list of strings, FIFO beyond capacity."""

    def __init__(self, capacity: int = 100) -> None:
        self.items: list[str] = []
        self.capacity = capacity

    def recall(self, query: str = "", limit: int = 5) -> str:
        # simple relevance: substring match, else most recent
        matches = [i for i in self.items if query.lower() in i.lower()] if query else []
        chosen = matches[-limit:] if matches else self.items[-limit:]
        return "\n---\n".join(reversed(chosen))

    async def store_async(self, item: str) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def store(self, item: str) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)


class FileMemoryStore(InMemoryStore):
    """Persistent memory: one JSON-lines file (append + load on start)."""

    def __init__(self, path: str = "memory.jsonl", capacity: int = 1000) -> None:
        super().__init__(capacity=capacity)
        self.path = Path(path)
        if self.path.exists():
            self.items = [line.strip() for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    async def store_async(self, item: str) -> None:
        await super().store_async(item)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(item.replace("\n", " ") + "\n")
