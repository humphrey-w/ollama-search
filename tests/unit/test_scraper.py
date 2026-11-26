"""Unit tests for the scraper tool."""
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
