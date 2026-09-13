"""User-defined HTTP tools built from a plain config dict.

Personal "skills" that add capability without executing arbitrary code:
the tool only performs an HTTP request against a user-supplied URL template.

Spec keys:
  name, description, method (GET/POST/...), url (supports {param} placeholders),
  headers (dict or JSON string), params (optional list of {name, type, required,
  description}).

Placeholders in the URL become required string parameters; extra kwargs are
sent as query string for GET or as JSON body otherwise.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ..tools.base import Tool

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _url_params(url: str) -> list[str]:
    return _PLACEHOLDER.findall(url)


def make_custom_http_tool(spec: dict[str, Any]) -> Tool:
    name = str(spec.get("name") or "http_tool").strip()
    if not name:
        raise ValueError("custom tool requires a 'name'")
    description = (spec.get("description") or "").strip() or f"HTTP tool {name}"
    url = (spec.get("url") or "").strip()
    if not url:
        raise ValueError(f"custom tool '{name}': 'url' is required")
    method = (spec.get("method") or "GET").upper()
    headers_raw = spec.get("headers") or {}
    if isinstance(headers_raw, str):
        try:
            headers_raw = json.loads(headers_raw or "{}")
        except json.JSONDecodeError:
            headers_raw = {}
    headers = {str(k): str(v) for k, v in headers_raw.items()}

    props: dict[str, dict[str, Any]] = {p: {"type": "string"} for p in _url_params(url)}
    required: list[str] = _url_params(url)
    params_spec = spec.get("params") or []
    if isinstance(params_spec, str):
        try:
            params_spec = json.loads(params_spec or "[]")
        except json.JSONDecodeError:
            params_spec = []
    for p in params_spec:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        name_p = str(p["name"])
        props[name_p] = {"type": p.get("type", "string")}
        if p.get("description"):
            props[name_p]["description"] = p["description"]
        if p.get("required", True) and name_p not in required:
            required.append(name_p)
    if "body" not in props and method != "GET":
        props["body"] = {"type": "object", "description": "JSON body to send"}

    async def _fn(**kwargs: Any) -> str:
        path = url
        used: set[str] = set()
        for placeholder in _url_params(url):
            val = kwargs.get(placeholder, "")
            path = path.replace("{" + placeholder + "}", str(val))
            used.add(placeholder)
        rest = {k: v for k, v in kwargs.items() if k not in used}
        async with httpx.AsyncClient(timeout=30.0) as http:
            if method == "GET":
                resp = await http.get(path, params=rest or None, headers=headers)
            else:
                body = rest.pop("body", None) if "body" in rest else (rest or None)
                resp = await http.request(method, path, json=body, headers=headers)
        resp.raise_for_status()
        return resp.text[:8000]

    _fn.__name__ = name.replace("-", "_").replace(" ", "_")
    _fn.__doc__ = description
    tool = Tool(name=name, description=description, fn=_fn)
    tool.openai_schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }
    return tool
