"""Orchestrator for managing the interaction between the user, the model, and
the tools."""
import sys
from typing import Any, Iterator

from ollama_web_search import ollama_client
from ollama_web_search.config import (DEFAULT_MODEL, SYSTEM_PROMPT,
                                      MAX_TOOL_ITERATIONS)
from ollama_web_search.tools import ToolDispatcher, create_tool_dispatcher


class Orchestrator:
    """Manages the interaction between the user, the model, and the tools."""
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        enable_search: bool = True,
        dispatcher: ToolDispatcher | None = None,
    ) -> None:
        self.model = model
        self.enable_search = enable_search
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
            while (
                active_schemas
                and self.dispatcher
                and iterations < MAX_TOOL_ITERATIONS
            ):
                response_msg = ollama_client.chat(
                    messages=self.messages,
                    model=self.model,
                    tools=active_schemas,
                    stream=False,
                )
                assert isinstance(response_msg, dict)
                tool_calls = response_msg.get("tool_calls") or []

                if not tool_calls:
                    self.messages.append(response_msg)
                    direct_reply = response_msg.get("content", "")
                    if direct_reply:
                        yield direct_reply
                    return

                self.messages.append(response_msg)
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

            # Final synthesis pass (streamed, tools omitted to guarantee
            # completion)
            token_stream = ollama_client.chat(
                messages=self.messages,
                model=self.model,
                tools=None,
                stream=True,
            )
            full_reply: list[str] = []
            for token in token_stream:
                full_reply.append(token)
                yield token

            self.messages.append({
                "role": "assistant",
                "content": "".join(full_reply)
            })

        except BaseException:
            del self.messages[checkpoint:]
            raise
