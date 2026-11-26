"""Orchestrator for managing the interaction between the user, the model, and
the tools."""
import sys
from typing import Any, Iterator

from ollama_web_search import ollama_client
from ollama_web_search.config import (DEFAULT_MODEL, SYSTEM_PROMPT,
                                      MAX_TOOL_ITERATIONS)
from ollama_web_search.tools import ToolDispatcher, create_tool_dispatcher


class ThinkingFilter:
    """Stateful stream filter that strips <think>...</think> blocks across
    split chunks."""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._in_thinking = False
        self._buffer = ""
        self._strip_leading_newlines = False

    def feed(self, text: str) -> str:
        """Process a streamed text chunk and return only non-thinking visible
        text."""
        self._buffer += text
        out: list[str] = []

        while self._buffer:
            if self._in_thinking:
                close_idx = self._buffer.find(self._CLOSE_TAG)
                if close_idx != -1:
                    self._buffer = self._buffer[close_idx +
                                                len(self._CLOSE_TAG):]
                    self._in_thinking = False
                    self._strip_leading_newlines = True
                    continue

                # Keep only enough trailing chars to detect a split
                # </think> tag
                keep = len(self._CLOSE_TAG) - 1
                if len(self._buffer) > keep:
                    self._buffer = self._buffer[-keep:]
                break

            open_idx = self._buffer.find(self._OPEN_TAG)
            if open_idx != -1:
                visible = self._buffer[:open_idx]
                if visible:
                    out.append(self._clean_leading(visible))
                self._buffer = self._buffer[open_idx + len(self._OPEN_TAG):]
                self._in_thinking = True
                continue

            # Hold back any suffix that could be the start of <think>
            hold = self._partial_tag_suffix(self._buffer, self._OPEN_TAG)
            if hold:
                visible = self._buffer[:-hold]
                if visible:
                    out.append(self._clean_leading(visible))
                self._buffer = self._buffer[-hold:]
                break

            out.append(self._clean_leading(self._buffer))
            self._buffer = ""

        return "".join(out)

    def flush(self) -> str:
        """Flush any remaining buffered characters at the end of a stream."""
        if self._in_thinking:
            self._buffer = ""
            return ""
        rem = self._clean_leading(self._buffer)
        self._buffer = ""
        return rem

    def _clean_leading(self, text: str) -> str:
        if self._strip_leading_newlines:
            text = text.lstrip("\r\n")
            if text:
                self._strip_leading_newlines = False
        return text

    @staticmethod
    def _partial_tag_suffix(text: str, tag: str) -> int:
        max_check = min(len(text), len(tag) - 1)
        for length in range(max_check, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0


class Orchestrator:
    """Manages the interaction between the user, the model, and the tools."""
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        enable_search: bool = True,
        show_thinking: bool = False,
        dispatcher: ToolDispatcher | None = None,
    ) -> None:
        self.model = model
        self.enable_search = enable_search
        self.show_thinking = show_thinking
        self.dispatcher = dispatcher or create_tool_dispatcher()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def reset(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def turn_count(self) -> int:
        """Return the number of user turns in the current session."""
        return sum(1 for m in self.messages if m.get("role") == "user")

    def _format_status(self, tool_name: str, args: Any) -> str:
        if isinstance(args, dict):
            if tool_name == "web_search":
                return f"[Searching: {args.get('query', '')}...]"
            if tool_name == "fetch_page":
                return f"[Fetching page: {args.get('url', '')}...]"
        return f"[Running {tool_name}...]"

    def run_turn(self, user_prompt: str) -> Iterator[str]:
        """Run a single turn of interaction: send the user prompt to the model,
        handle tool invocations, and yield the model's response."""
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_prompt})

        active_schemas = (
            self.dispatcher.registry.schemas
            if self.enable_search and self.dispatcher
            else None
        )

        try:
            iterations = 0
            while True:
                tools_for_pass = (
                    active_schemas
                    if (
                        active_schemas
                        and self.dispatcher
                        and iterations < MAX_TOOL_ITERATIONS
                    )
                    else None
                )

                full_reply: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                think_filter = (
                    ThinkingFilter() if not self.show_thinking else None
                )

                for chunk in ollama_client.chat_stream(
                    messages=self.messages,
                    model=self.model,
                    tools=tools_for_pass,
                ):
                    msg = chunk.get("message", {})

                    # Handle Ollama's dedicated thinking field
                    # (e.g., qwen3 / deepseek-r1)
                    if thinking := msg.get("thinking"):
                        if self.show_thinking:
                            yield f"\033[90m{thinking}\033[0m"

                    if content := msg.get("content"):
                        visible = (
                            think_filter.feed(content) if think_filter
                            else content
                        )
                        if visible:
                            full_reply.append(visible)
                            yield visible

                    if calls := msg.get("tool_calls"):
                        tool_calls.extend(calls)

                if think_filter:
                    if tail := think_filter.flush():
                        full_reply.append(tail)
                        yield tail

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(full_reply),
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                self.messages.append(assistant_msg)

                if not tool_calls or not self.dispatcher:
                    return

                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})

                    status = self._format_status(name, args)
                    print(f"\033[90m{status}\033[0m", file=sys.stderr)

                    tool_output = self.dispatcher.dispatch(name, args)
                    self.messages.append({
                        "role": "tool",
                        "content": tool_output
                    })

                iterations += 1

        except BaseException:
            del self.messages[checkpoint:]
            raise
