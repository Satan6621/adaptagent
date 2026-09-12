"""Base Tool abstraction: any callable becomes a tool."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """Wraps a function as an agent-usable tool.

    The wrapper extracts the signature/description from the callable, so
    tools can be plain functions decorated with @tool.
    """

    name: str
    description: str
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator: promote a function to a Tool (name from __name__, doc from __doc__)."""
    name = fn.__name__.strip("_")
    doc = inspect.getdoc(fn) or ""
    # First line = description
    first_line = doc.splitlines()[0] if doc else f"Tool {name}"
    return Tool(name=name, description=first_line, fn=fn)
