"""Integration tests for the SearXNG web search tool."""
from unittest.mock import patch
from ollama_web_search.tools import create_tool_dispatcher


def test_dispatcher_registry_and_searxng_search(mock_searxng_url: str) -> None:
    """Verifies end-to-end tool registration, dispatch, HTTP request, and
    result formatting."""
    dispatcher = create_tool_dispatcher()

    schema_names = [s["function"]["name"] for s in dispatcher.registry.schemas]
    assert "web_search" in schema_names

    with patch("ollama_web_search.tools.searxng.SEARXNG_BASE_URL",
               mock_searxng_url):
        output = dispatcher.dispatch(
            "web_search",
            {"query": "python hatchling tutorial", "max_results": 1},
        )

    assert "Title: Docs for python hatchling tutorial" in output
    assert "URL: https://example.org" in output
    assert "Snippet: Detailed guide about python hatchling tutorial." in output
    assert "Secondary Source" not in output


def test_dispatcher_searxng_empty_results(mock_searxng_url: str) -> None:
    """Verifies graceful handling when SearXNG returns zero matches over
    HTTP."""
    dispatcher = create_tool_dispatcher()

    with patch("ollama_web_search.tools.searxng.SEARXNG_BASE_URL",
               mock_searxng_url):
        output = dispatcher.dispatch("web_search", {"query": "empty_query"})

    assert output == "No search results found."


def test_dispatcher_handles_unreachable_searxng() -> None:
    """Verifies clean error string when SearXNG port is closed."""
    dispatcher = create_tool_dispatcher()

    with patch("ollama_web_search.tools.searxng.SEARXNG_BASE_URL",
               "http://127.0.0.1:9"):
        output = dispatcher.dispatch("web_search", {"query": "test"})

    assert output.startswith("Search error (SearXNG unreachable):")
