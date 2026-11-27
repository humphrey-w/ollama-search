"""Unit tests for the REPL functionality of the Ollama Web Search CLI."""
from unittest.mock import patch

from ollama_web_search import cli
from ollama_web_search.orchestrator import Orchestrator


def test_orchestrator_retains_multi_turn_history_and_resets() -> None:
    """Test that the Orchestrator retains multi-turn history and can reset."""
    orch = Orchestrator(model="llama3.1:8b")

    turn1 = iter([{
        "message": {"role": "assistant", "content": "Hi Alice!"},
        "done": True
    }])
    turn2 = iter([{
        "message": {"role": "assistant", "content": "Your name is Alice."},
        "done": True
    }])

    with patch(
        "ollama_web_search.ollama_client.chat_stream",
        side_effect=[turn1, turn2],
    ):
        r1 = "".join(orch.run_turn("My name is Alice."))
        r2 = "".join(orch.run_turn("What is my name?"))

    assert r1 == "Hi Alice!"
    assert r2 == "Your name is Alice."
    assert orch.turn_count == 2
    assert len(orch.messages) == 5  # 1 system + 2 user + 2 assistant

    orch.reset()
    assert orch.turn_count == 0
    assert len(orch.messages) == 1


def test_orchestrator_rolls_back_on_interruption() -> None:
    """Test that the Orchestrator rolls back the current turn on
    interruption."""
    orch = Orchestrator(model="llama3.1:8b")

    pass1 = iter([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "test"}
                    }
                }],
            },
            "done": True,
        }
    ])

    def _aborting_stream():
        yield {
            "message": {"role": "assistant", "content": "Partial"},
            "done": False
        }
        raise KeyboardInterrupt()

    with (
        patch("ollama_web_search.ollama_client.chat_stream",
              side_effect=[pass1, _aborting_stream()]),
        patch.object(
            orch.dispatcher,
            "dispatch",
            return_value=cli.ToolResult("Some snippet"),
        ),
    ):
        try:
            list(orch.run_turn("Trigger search"))
        except KeyboardInterrupt:
            pass

    # Incomplete turn should be rolled back completely
    assert len(orch.messages) == 1
    assert orch.turn_count == 0


def test_cli_repl_handles_commands_and_exit(capsys) -> None:
    """Test that the CLI REPL handles commands and exits cleanly."""
    user_inputs = iter(["/search off", "/history", "Hello there", "/exit"])

    stream = iter([
        {
            "message": {"role": "assistant", "content": "General "},
            "done": False
        },
        {
            "message": {"role": "assistant", "content": "Kenobi!"},
            "done": True
        },
    ])

    with (
        patch("builtins.input", side_effect=user_inputs),
        patch(
            "ollama_web_search.ollama_client.chat_stream",
            return_value=stream,
        ),
    ):
        exit_code = cli.main([])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "[Web search disabled]" in captured
    assert "[Turns: 0 | Messages in context: 1]" in captured
    assert "General Kenobi!" in captured


def test_answer_urls_are_replaced_with_numbered_citations() -> None:
    """Test that source URLs in answers become numbered Markdown links."""
    sources = [
        cli.Source(title="Python", url="https://python.org"),
        cli.Source(title="Testing", url="https://pytest.org"),
    ]

    answer = cli._add_source_citations(
        "See https://python.org and https://pytest.org.",
        sources,
    )

    assert answer == "See [1](https://python.org) and [2](https://pytest.org)."
