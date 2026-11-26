"""Unit tests for the ThinkingFilter and Orchestrator classes."""
from unittest.mock import patch

from ollama_web_search.orchestrator import Orchestrator, ThinkingFilter


def test_thinking_filter_strips_split_tags_and_leading_newlines() -> None:
    """Test that ThinkingFilter correctly removes <think>...</think> blocks"""
    flt = ThinkingFilter()
    chunks = [
        "<thi", "nk>Let me ",
        "reason about this.</th",
        "ink>\n\nFinal ", "answer."
    ]

    out = "".join(flt.feed(c) for c in chunks) + flt.flush()
    assert out == "Final answer."


def test_thinking_filter_preserves_non_think_angle_brackets() -> None:
    """Test that ThinkingFilter does not remove angle brackets that are not
    <think> tags."""
    flt = ThinkingFilter()
    chunks = ["2 < 3 and ", "<thick>bold</thick> text"]

    out = "".join(flt.feed(c) for c in chunks) + flt.flush()
    assert out == "2 < 3 and <thick>bold</thick> text"


def test_orchestrator_hides_and_shows_thinking_tokens() -> None:
    """Test that Orchestrator correctly hides or shows <think> blocks based on
    the show_thinking flag."""
    def _make_stream():
        return iter([
            {
                "message": {
                    "role": "assistant",
                    "thinking": "Internal thought. "
                },
                "done": False
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>More thought</think>\n\n"
                },
                "done": False
            },
            {
                "message": {"role": "assistant", "content": "Clean answer."},
                "done": True
            },
        ])

    # Default: thinking hidden
    orch_hidden = Orchestrator(model="qwen3:8b", show_thinking=False)
    with patch("ollama_web_search.ollama_client.chat_stream",
               return_value=_make_stream()):
        reply_hidden = "".join(orch_hidden.run_turn("Hi"))

    assert reply_hidden == "Clean answer."
    assert orch_hidden.messages[-1]["content"] == "Clean answer."

    # With show_thinking=True: both channels rendered
    orch_shown = Orchestrator(model="qwen3:8b", show_thinking=True)
    with patch("ollama_web_search.ollama_client.chat_stream",
               return_value=_make_stream()):
        reply_shown = "".join(orch_shown.run_turn("Hi"))

    assert "Internal thought." in reply_shown
    assert "<think>More thought</think>" in reply_shown
    assert "Clean answer." in reply_shown
