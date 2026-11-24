"""Unit tests for the Orchestrator class."""
from unittest.mock import patch

from ollama_web_search.orchestrator import Orchestrator
from ollama_web_search.tools import create_tool_dispatcher


def test_direct_reply_without_tool_call():
    """Test that the Orchestrator correctly handles a direct reply from the
    model without invoking any tools."""
    orch = Orchestrator(model="llama3.1:8b")

    with patch(
        "ollama_web_search.ollama_client.chat",
        return_value={"role": "assistant", "content": "4"},
    ) as mock_chat:
        result = "".join(orch.run_turn("What is 2 + 2?"))

    assert result == "4"
    assert mock_chat.call_count == 1


def test_tool_call_uses_injected_dispatcher():
    """Test that the Orchestrator uses the provided ToolDispatcher for tool
    calls."""
    dispatcher = create_tool_dispatcher()
    orch = Orchestrator(model="llama3.1:8b", dispatcher=dispatcher)

    first_pass = {
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
    }

    with patch(
        "ollama_web_search.ollama_client.chat",
        side_effect=[first_pass, iter(["Linux ", "6.12"])],
    ) as mock_chat:
        result = "".join(orch.run_turn("What is the latest Linux kernel?"))

    assert result == "Linux 6.12"
    assert mock_chat.call_count == 2
    tool_msgs = [m for m in orch.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "Active kernel releases" in tool_msgs[0]["content"]
