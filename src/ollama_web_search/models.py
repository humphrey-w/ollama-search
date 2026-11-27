"""Shared data models used across CLI, orchestration, and tools."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A web source returned by a search or fetch tool."""
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class ToolResult:
    """Structured tool output with model text and optional source metadata."""
    text: str
    sources: tuple[Source, ...] = ()
