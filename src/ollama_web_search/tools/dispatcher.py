"""Dispatcher for executing tool calls from the tool registry."""
from typing import Any
from ollama_web_search.models import ToolResult
from ollama_web_search.tools.registry import ToolRegistry


class ToolDispatcher:
    """Resolves and executes tool calls requested by the LLM against a
    ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Dispatch a tool call and return structured result metadata."""
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(f"Error: Unknown tool '{tool_name}'.")

        if not isinstance(arguments, dict):
            return ToolResult(
                f"Error: Invalid arguments format for '{tool_name}'; "
                "expected an object."
            )

        try:
            result = tool.handler(**arguments)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(result)
        except TypeError as e:
            return ToolResult(f"Error executing '{tool_name}': {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            return ToolResult(f"Unexpected error in '{tool_name}': {e}")
