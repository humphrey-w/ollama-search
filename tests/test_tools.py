"""Unit tests for the tool registry and dispatcher."""
import pytest
from ollama_web_search.tools.dispatcher import ToolDispatcher
from ollama_web_search.tools.registry import ToolRegistry

SAMPLE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "echo_tool",
        "description": "Echoes input text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


def test_registry_registers_and_exposes_schemas():
    """Test that the ToolRegistry correctly registers tools and exposes their
    schemas."""
    registry = ToolRegistry()
    registry.register(SAMPLE_SCHEMA, lambda text: f"echo: {text}")

    assert len(registry) == 1
    assert registry.schemas == [SAMPLE_SCHEMA]
    assert registry.get("echo_tool") is not None


def test_registry_rejects_schema_without_name():
    """Test that the ToolRegistry raises an error when registering a schema"""
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register({"type": "function"}, lambda: "")


def test_dispatcher_executes_registered_tool():
    """Test that the ToolDispatcher correctly executes a registered tool."""
    registry = ToolRegistry()
    registry.register(SAMPLE_SCHEMA, lambda text: f"echo: {text}")
    dispatcher = ToolDispatcher(registry)

    assert dispatcher.dispatch("echo_tool", {"text": "hello"}) == "echo: hello"


def test_dispatcher_handles_unknown_tool_and_bad_args():
    """Test that the ToolDispatcher handles unknown tools and invalid
    arguments."""
    registry = ToolRegistry()
    registry.register(SAMPLE_SCHEMA, lambda text: f"echo: {text}")
    dispatcher = ToolDispatcher(registry)

    assert "Unknown tool" in dispatcher.dispatch("missing_tool", {})
    assert "Error executing" in dispatcher.dispatch("echo_tool",
                                                    {"wrong_param": 123})
