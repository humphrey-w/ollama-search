"""The module manages and dispatches tool calls."""

from ollama_web_search.tools.dispatcher import ToolDispatcher
from ollama_web_search.tools.registry import ToolRegistry


__all__ = ["ToolRegistry", "ToolDispatcher"]
