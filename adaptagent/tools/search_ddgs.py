"""DuckDuckGo search (optional dependency: pip install adaptagent[search])."""

from __future__ import annotations

from .base import Tool, tool


@tool
def ddgs_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo and return results.

    query: The search query.
    max_results: Maximum number of results to return.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed. Install with: pip install ddgs"
    try:
        results = DDGS().text(query, max_results=max_results)
        lines = []
        for r in results:
            lines.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as exc:  # noqa: BLE001
        return f"Search error: {exc}"


def DDGSSearchTool() -> Tool:
    return ddgs_search
