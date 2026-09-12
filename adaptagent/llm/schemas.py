"""JSON-schema generation for tools from type hints (no pydantic needed)."""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, get_args, get_origin, get_type_hints

from ..tools.base import Tool


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation in (int,):
        return {"type": "integer"}
    if annotation in (float,):
        return {"type": "number"}
    if annotation in (bool,):
        return {"type": "boolean"}
    if annotation in (str,) or annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, typing.List):
        item = _annotation_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if origin in (tuple, typing.Tuple):
        return {"type": "array"}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(args)}
    if origin in (typing.Union,):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _annotation_to_schema(non_none[0])
            if type(None) in args:
                schema = {**schema, "nullable": True}
            return schema
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    return {"type": "string"}


def get_tool_schema(tool: Tool) -> dict[str, Any]:
    """OpenAI function-calling schema from the tool's callable."""
    fn = tool.fn
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {k: v for k, v in getattr(fn, "__annotations__", {}).items()}
    hints.pop("return", None)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self",):
            continue
        schema = _annotation_to_schema(hints.get(name, param.annotation))
        doc = None
        if fn.__doc__:
            for line in fn.__doc__.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{name}:"):
                    doc = stripped.partition(":")[2].strip()
                    break
        if doc:
            schema["description"] = doc
        props[name] = schema
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip() or f"Tool {tool.name}",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def get_tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    return [get_tool_schema(t) for t in tools]


def get_schema(fn: Callable) -> dict[str, Any]:
    """Schema for a bare function (used by evaluators)."""
    class _T:
        def __init__(self) -> None:
            self.name = fn.__name__.replace("_", " ")
            self.description = inspect.getdoc(fn) or ""
            self.fn = fn

    return get_tool_schema(_T())
