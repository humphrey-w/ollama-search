"""Unit tests for the searxng tool."""
import io
import json
from unittest.mock import patch

from ollama_web_search.tools import searxng


def test_search_formats_results_and_sources():
    """Test that search returns text and source metadata."""
    fake_payload = {
        "results": [
            {
                "title": "Python 3.13 Release",
                "url": "https://python.org",
                "content": "Python 3.13 is now available.",
            }
        ]
    }
    fake_resp = io.BytesIO(json.dumps(fake_payload).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=fake_resp):
        output = searxng.search("python release")

    assert "Title: Python 3.13 Release" in output.text
    assert "URL: https://python.org" in output.text
    assert "Snippet: Python 3.13 is now available." in output.text
    assert len(output.sources) == 1
    assert output.sources[0].title == "Python 3.13 Release"
    assert output.sources[0].url == "https://python.org"


def test_search_handles_empty_results():
    """Test that the searxng.search function handles empty search results."""
    fake_resp = io.BytesIO(json.dumps({"results": []}).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=fake_resp):
        output = searxng.search("nonexistent query")

    assert output.text == "No search results found."
    assert output.sources == ()
