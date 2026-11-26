"""Unit tests for the Orchestrator class."""
from unittest.mock import patch

from ollama_web_search.orchestrator import Orchestrator
from ollama_web_search.tools import create_tool_dispatcher


def test_direct_reply_without_tool_call():
    """Test that the Orchestrator correctly handles a direct reply from the
    model without invoking any tools."""
    orch = Orchestrator(model="llama3.1:8b")

    stream = iter([
        {"message": {"role": "assistant", "content": "4"}, "done": True},
    ])

    with patch(
        "ollama_web_search.ollama_client.chat_stream",
        return_value=stream,
    ) as mock_chat:
        result = "".join(orch.run_turn("What is 2 + 2?"))

    assert result == "4"
    assert mock_chat.call_count == 1


def test_tool_call_uses_injected_dispatcher():
    """Test that the Orchestrator uses the provided ToolDispatcher for tool
    calls."""
    dispatcher = create_tool_dispatcher()
    orch = Orchestrator(model="llama3.1:8b", dispatcher=dispatcher)

    pass1 = iter([
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "latest kernel"},
                        }
                    }
                ],
            },
            "done": True,
        }
    ])
    pass2 = iter([
        {"message": {"role": "assistant", "content": "Linux "}, "done": False},
        {"message": {"role": "assistant", "content": "6.12"}, "done": True},
    ])

    with (
        patch(
            "ollama_web_search.ollama_client.chat_stream",
            side_effect=[pass1, pass2]
        ) as mock_chat,
        patch.object(
            dispatcher,
            "dispatch",
            return_value=(
                "Title: Kernel\nURL: https://kernel.org\nSnippet: 6.12"
            ),
        ) as mock_dispatch,
    ):
        result = "".join(orch.run_turn("What is the latest Linux kernel?"))

    assert result == "Linux 6.12"
    mock_dispatch.assert_called_once_with(
        "web_search",
        {"query": "latest kernel"}
    )
    assert mock_chat.call_count == 2
