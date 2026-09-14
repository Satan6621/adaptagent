"""Structured outputs: JSON extraction + lightweight JSON Schema validation.

Keeps the engine dependency-free: supports the common subset of JSON Schema
(type, required, properties, items) enough to gate LLM outputs publishable to
the next step. Invalid JSON triggers a bounded retry inside the executor's
repeat loop.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> Any | None:
    """Return the first JSON value found in a text (fenced or raw), else None."""
    text = (text or "").strip()
    candidates = [text]
    candidates.extend(f.strip() for f in _FENCE.findall(text))
    for candidate in candidates:
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = candidate.find(open_ch)
            if start == -1:
                continue
            end = candidate.rfind(close_ch)
            if end <= start:
                continue
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


_TYPE_MAP = {
    "string": str,
    "integer": (int,),
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": (type(None),),
}


def _type_error(value: Any, expected: str) -> str | None:
    if expected == "null":
        return None if value is None else f"se esperaba null, se obtuvo {type(value).__name__}"
    types = _TYPE_MAP.get(expected)
    if types is None:
        return None  # unknown type keyword -> ignore
    if expected in ("integer", "number") and isinstance(value, bool):
        return f"se esperaba {expected}, se obtuvo bool"
    if not isinstance(value, types):  # type: ignore[arg-type]
        return f"se esperaba {expected}, se obtuvo {type(value).__name__}"
    if expected == "integer" and not float(value).is_integer():
        return "se esperaba integer, se obtuvo un número con decimales"
    return None


def validate(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """Validate a parsed value against a JSON-schema-like dict. Returns errors."""
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type:
        error = _type_error(value, expected_type)
        if error:
            errors.append(f"{path}: {error}")
            return errors
    if expected_type == "array":
        item_schema = schema.get("items")
        if item_schema and isinstance(value, list):
            for i, item in enumerate(value):
                errors.extend(validate(item_schema, item, f"{path}[{i}]"))
        return errors
    if expected_type != "object" or not isinstance(value, dict):
        return errors
    required = schema.get("required") or []
    for key in required:
        if key not in value:
            errors.append(f"{path}: falta el campo obligatorio '{key}'")
    for key, subschema in (schema.get("properties") or {}).items():
        if key in value:
            errors.extend(validate(subschema, value[key], f"{path}.{key}"))
    if schema.get("additionalProperties") is False:
        allowed = set((schema.get("properties") or {}).keys())
        for key in value.keys() - allowed:
            errors.append(f"{path}: campo no permitido '{key}'")
    return errors


def validate_step_output(schema: dict[str, Any], text: str) -> tuple[Any | None, list[str]]:
    """Extract JSON from text and validate it. Returns (parsed, errors)."""
    parsed = extract_json(text)
    if parsed is None:
        return None, ["no se encontró un JSON válido en la respuesta"]
    errors = validate(schema, parsed)
    return parsed, errors


def schema_hint(schema: dict[str, Any]) -> str:
    """A short prompt fragment instructing the agent to return valid JSON."""
    compact = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "RESPONDE únicamente con un objeto/arreglo JSON válido que cumpla exactamente "
        f"este esquema: {compact}"
    )
