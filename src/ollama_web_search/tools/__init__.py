"""The module manages and dispatches tool calls."""

from ollama_web_search.tools import searxng
from ollama_web_search.tools.dispatcher import ToolDispatcher
from ollama_web_search.tools.registry import ToolRegistry


def create_tool_dispatcher() -> ToolDispatcher:
    """Create a ToolDispatcher with registered tools."""
    registry = ToolRegistry()
    registry.register(searxng.TOOL_SCHEMA, searxng.search)
    return ToolDispatcher(registry)


__all__ = ["ToolRegistry", "ToolDispatcher", "create_tool_dispatcher"]
