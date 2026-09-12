"""Filesystem tools scoped to a workspace directory (safety: path confinement)."""

from __future__ import annotations

import os
from pathlib import Path

from .base import Tool, tool

_WORKSPACE = Path(os.getenv("ADAPTAGENT_WORKSPACE", "workspace")).resolve()


def _safe_path(rel_path: str) -> Path:
    target = (_WORKSPACE / rel_path).resolve()
    if not str(target).startswith(str(_WORKSPACE)):
        raise ValueError(f"Path escapes workspace: {rel_path}")
    return target


@tool
def file_read(path: str) -> str:
    """Read a text file from the workspace.

    path: Relative path inside the workspace.
    """
    target = _safe_path(path)
    if not target.is_file():
        return f"Error: file not found: {path}"
    return target.read_text(encoding="utf-8", errors="replace")


@tool
def file_write(path: str, content: str) -> str:
    """Write a text file inside the workspace (creates parent dirs).

    path: Relative path inside the workspace.
    content: File content to write.
    """
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


@tool
def file_list(path: str = ".") -> str:
    """List files and directories inside the workspace.

    path: Relative directory path; default is the workspace root.
    """
    target = _safe_path(path)
    if not target.exists():
        return f"Error: not found: {path}"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def FileReadTool() -> Tool:
    return file_read


def FileWriteTool() -> Tool:
    return file_write


def FileListTool() -> Tool:
    return file_list
