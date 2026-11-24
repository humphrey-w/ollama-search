"""SearXNG search tool for Ollama Search."""
import json
import urllib.error
import urllib.parse
import urllib.request

from ollama_web_search.config import (MAX_SEARCH_RESULTS, REQUEST_TIMEOUT,
                                      SEARXNG_BASE_URL)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web via the SearXNG metasearch engine.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                }
            },
            "required": ["query"],
        },
    },
}


def search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Perform a web search using SearXNG and return formatted results."""
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{SEARXNG_BASE_URL.rstrip('/')}/search?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "OllamaSearchCLI/0.1"}
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return f"Search error (SearXNG unreachable): {e.reason}"
    except (json.JSONDecodeError, TimeoutError) as e:
        return f"Search error: {e}"

    results = payload.get("results", [])[:max_results]

    if not results:
        return "No search results found."

    formatted = []

    for item in results:
        title = item.get("title", "Untitled")
        link = item.get("url", "")
        snippet = item.get("content", "").strip()
        formatted.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")

    return "\n\n".join(formatted)
