"""Generic HTTP request tool (GET/POST/PUT/DELETE)."""

from __future__ import annotations

from typing import Literal

from .base import Tool, tool


@tool
def http_request(
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET",
    body: str = "",
    headers: str = "",
) -> str:
    """Make an HTTP request and return the response text.

    url: The target URL.
    method: HTTP method (GET, POST, PUT, DELETE).
    body: Optional request body (JSON string) for POST/PUT.
    headers: Optional headers as a JSON object string.
    """
    import json as _json

    import httpx

    parsed_headers = _json.loads(headers) if headers else {}
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.request(method, url, content=body or None, headers=parsed_headers)
            text = resp.text
            return f"Status: {resp.status_code}\n{text[:5000]}"
    except Exception as exc:  # noqa: BLE001
        return f"HTTP error: {exc}"


def HTTPRequestTool() -> Tool:
    return http_request
