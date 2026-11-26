"""Unit tests for the scraper tool."""
from unittest.mock import patch

from ollama_web_search.orchestrator import Orchestrator
from ollama_web_search.tools import create_tool_dispatcher
from ollama_web_search.tools.scraper import fetch_page


def test_fetch_page_extracts_clean_text_and_strips_scripts(
        mock_web_url: str) -> None:
    """Test that fetch_page extracts clean text from an HTML page, removing
    scripts, styles, and navigation elements."""
    result = fetch_page(f"{mock_web_url}/article")

    assert "Release Notes" in result
    assert "Linux 6.12 introduces sched_ext support." in result
    assert "console.log" not in result
    assert "color:red" not in result
    assert "Menu" not in result


def test_fetch_page_truncates_long_content(mock_web_url: str) -> None:
    """Test that fetch_page truncates the extracted text to the specified
    maximum number of characters and appends a truncation notice."""
    result = fetch_page(f"{mock_web_url}/article", max_chars=25)
    assert "[Content truncated at 25 characters]" in result


def test_fetch_page_rejects_invalid_scheme_and_binary_content(
        mock_web_url: str) -> None:
    """Test that fetch_page rejects URLs with invalid schemes (like file://)"""
    assert "Invalid URL" in fetch_page("file:///etc/passwd")
    assert "Unsupported content type" in fetch_page(f"{mock_web_url}/binary")


def test_orchestrator_chains_search_then_fetch_page(mock_web_url: str) -> None:
    """Test that the Orchestrator can handle a tool call to web_search
    followed by a tool call to fetch_page, and that the final response is
    returned correctly."""
    dispatcher = create_tool_dispatcher()
    orch = Orchestrator(model="qwen3:8b", dispatcher=dispatcher)

    pass1_search = iter([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {
                    "name": "web_search",
                    "arguments": {"query": "linux 6.12"}}}],
            },
            "done": True,
        }
    ])
    pass2_fetch = iter([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {
                    "name": "fetch_page",
                    "arguments": {"url": f"{mock_web_url}/article"}}}],
            },
            "done": True,
        }
    ])
    pass3_final = iter([
        {
            "message": {"role": "assistant", "content": "It adds "},
            "done": False
        },
        {
            "message": {"role": "assistant", "content": "sched_ext."},
            "done": True
        },
    ])

    with (
        patch(
            "ollama_web_search.ollama_client.chat_stream",
            side_effect=[pass1_search, pass2_fetch, pass3_final],
        ),
        patch(
            "ollama_web_search.tools.searxng.search",
            return_value=f"URL: {mock_web_url}/article"
        ),
    ):
        reply = "".join(orch.run_turn("Summarize Linux 6.12 release notes"))

    assert reply == "It adds sched_ext."
    tool_messages = [
        m["content"] for m in orch.messages if m["role"] == "tool"
    ]
    assert len(tool_messages) == 2
    assert "sched_ext support" in tool_messages[1]
