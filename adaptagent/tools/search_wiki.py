"""Wikipedia search via the public MediaWiki API (no key required)."""

from __future__ import annotations

from .base import Tool, tool


@tool
def wikipedia_search(query: str, max_results: int = 3) -> str:
    """Search Wikipedia articles and return title, summary and URL.

    query: The search query.
    max_results: Maximum number of articles to return.
    """
    import httpx

    try:
        with httpx.Client(timeout=20) as client:
            search = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": max_results,
                    "format": "json",
                },
            ).json()
            hits = search.get("query", {}).get("search", [])
            parts = []
            for hit in hits:
                title = hit["title"]
                summary = client.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
                ).json()
                parts.append(
                    f"Title: {title}\nURL: {summary.get('content_urls', {}).get('desktop', {}).get('page', '')}\nSummary: {summary.get('extract', '')}"
                )
            return "\n\n".join(parts) if parts else "No results found."
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia error: {exc}"


def WikipediaSearchTool() -> Tool:
    return wikipedia_search
