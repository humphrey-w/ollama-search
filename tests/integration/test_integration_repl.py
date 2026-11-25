"""Integration tests for the REPL flow."""
from unittest.mock import patch

from ollama_web_search import cli


def test_repl_multi_turn_end_to_end_http(
    mock_ollama_url: tuple[str, list[list[dict]]],
    mock_searxng_url: str,
    capsys,
) -> None:
    """Verifies multi-turn REPL flow across real local Ollama and SearXNG HTTP
    servers."""
    ollama_url, captured_histories = mock_ollama_url
    inputs = iter([
        "What is the latest kernel?", "What did I just ask?", "/exit"
    ])

    with (
        patch("builtins.input", side_effect=inputs),
        patch("ollama_web_search.ollama_client.OLLAMA_BASE_URL", ollama_url),
        patch("ollama_web_search.tools.searxng.SEARXNG_BASE_URL",
              mock_searxng_url),
    ):
        exit_code = cli.main([])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Kernel is 6.12." in out
    assert "You previously asked: What is the latest kernel?" in out

    # Confirm Turn 2's HTTP payload carried Turn 1's full message chain
    turn2_messages = captured_histories[-1]
    assert [m["role"] for m in turn2_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
