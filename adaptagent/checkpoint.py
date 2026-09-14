"""Checkpointing: snapshot workflow state to resume interrupted runs.

- `checkpoint_from` builds a checkpoint from partial execution results.
- `CheckpointStore` keeps run history in memory and, when a runs dir is
  configured (env ADAPTAGENT_RUNS_DIR), on disk (locally / self-hosted).
  On serverless it degrades to in-memory; for full serverless resume the
  client passes the checkpoint blob back in the execute payload.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workflow.executor import StepResult


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Checkpoint:
    """Snapshot of an execution: which steps produced what output."""

    run_id: str
    goal: str
    step_outputs: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        self.created_at = self.created_at or _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            run_id=str(data.get("run_id") or uuid.uuid4().hex[:8]),
            goal=str(data.get("goal") or ""),
            step_outputs=dict(data.get("step_outputs") or {}),
            skipped=list(data.get("skipped") or []),
            created_at=str(data.get("created_at") or ""),
        )


def checkpoint_from(
    run_id: str, goal: str, order: list[str], results: dict[str, "StepResult"]
) -> Checkpoint:
    """Build a checkpoint from executed step results."""
    returns = {
        sid: results[sid].output
        for sid in order
        if sid in results and not results[sid].skipped and results[sid].output
    }
    skipped = [sid for sid in order if sid in results and results[sid].skipped]
    return Checkpoint(run_id=run_id, goal=goal, step_outputs=returns, skipped=skipped)


def _runs_dir() -> str | None:
    return os.environ.get("ADAPTAGENT_RUNS_DIR") or os.environ.get("ADAPTAGENT_RUNS")


class CheckpointStore:
    """In-memory run history (optionally persisted as JSON files)."""

    def __init__(self, runs_dir: str | None = None) -> None:
        self._memory: dict[str, Checkpoint] = {}
        self.runs_dir = runs_dir or _runs_dir()

    def _path(self, run_id: str) -> Path:
        return Path(self.runs_dir) / f"{run_id}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        self._memory[checkpoint.run_id] = checkpoint
        if self.runs_dir:
            try:
                path = self._path(checkpoint.run_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass  # keep in-memory on write failure

    def get(self, run_id: str) -> Checkpoint | None:
        checkpoint = self._memory.get(run_id)
        if checkpoint is not None:
            return checkpoint
        if self.runs_dir:
            try:
                path = self._path(run_id)
                if path.exists():
                    return Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except OSError:
                pass
        return None

    def history(self) -> list[dict[str, Any]]:
        items = []
        for run_id, cp in self._memory.items():
            items.append(
                {
                    "run_id": run_id,
                    "goal": cp.goal,
                    "created_at": cp.created_at,
                    "steps": len(cp.step_outputs),
                }
            )
        if self.runs_dir:
            try:
                for path in Path(self.runs_dir).glob("*.json"):
                    if path.stem not in self._memory:
                        cp = Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
                        items.append(
                            {
                                "run_id": cp.run_id,
                                "goal": cp.goal,
                                "created_at": cp.created_at,
                                "steps": len(cp.step_outputs),
                            }
                        )
            except OSError:
                pass
        return sorted(items, key=lambda i: i["created_at"], reverse=True)[:50]

    def latest(self) -> Checkpoint | None:
        history = self.history()
        if not history:
            return None
        return self.get(history[0]["run_id"])


_CLIENT_STORE: CheckpointStore | None = None


def get_store() -> CheckpointStore:
    """Process-wide + disk-backed store singleton."""
    global _CLIENT_STORE
    if _CLIENT_STORE is None:
        _CLIENT_STORE = CheckpointStore()
    return _CLIENT_STORE


def reset_store() -> None:
    """For tests: clear the singleton."""
    global _CLIENT_STORE
    _CLIENT_STORE = None
