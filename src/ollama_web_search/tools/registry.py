"""Registry for tools that can be used by Ollama's tool-calling API."""
from dataclasses import dataclass
from typing import Any, Callable

from ollama_web_search.models import ToolResult


@dataclass(frozen=True)
class RegisteredTool:
    """Represents a registered tool with its schema and handler."""
    name: str
    schema: dict[str, Any]
    handler: Callable[..., str | ToolResult]


class ToolRegistry:
    """Stores tool schemas and maps tool names to their Python handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, schema: dict[str, Any],
                 handler: Callable[..., str | ToolResult]) -> None:
        """Register a tool with its schema and handler function."""
        name = schema.get("function", {}).get("name")
        if not name:
            raise ValueError("Tool schema must define 'function.name'.")
        self._tools[name] = RegisteredTool(name=name,
                                           schema=schema, handler=handler)

    def get(self, name: str) -> RegisteredTool | None:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Return a list of all registered tool schemas."""
        return [tool.schema for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
