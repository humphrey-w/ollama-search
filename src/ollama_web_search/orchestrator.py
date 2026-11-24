"""Orchestrator for managing the interaction between the user, the model, and
the tools."""
import sys
from typing import Any, Iterator

from ollama_web_search import ollama_client
from ollama_web_search.config import DEFAULT_MODEL, SYSTEM_PROMPT
from ollama_web_search.tools import create_tool_dispatcher, ToolDispatcher


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
        self.dispatcher = dispatcher or (
            create_tool_dispatcher() if enable_search else None
        )
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def run_turn(self, user_prompt: str) -> Iterator[str]:
        """Run a single turn of interaction: send the user prompt to the
        model, handle tool invocations."""
        self.messages.append({"role": "user", "content": user_prompt})

        active_schemas = (
            self.dispatcher.registry.schemas
            if self.enable_search and self.dispatcher
            else None
        )

        # Pass 1: Evaluate whether the model wants to invoke a tool
        response_msg = ollama_client.chat(
            messages=self.messages,
            model=self.model,
            tools=active_schemas,
            stream=False,
        )
        assert isinstance(response_msg, dict)
        self.messages.append(response_msg)

        tool_calls = response_msg.get("tool_calls") or []
        if not tool_calls or not self.dispatcher:
            direct_reply = response_msg.get("content", "")
            if direct_reply:
                yield direct_reply
            return

        # Execute each requested tool call via ToolDispatcher
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})

            query_preview = (
                args.get("query", str(args))
                if isinstance(args, dict)
                else str(args)
            )
            print(f"\033[90m[Searching: {query_preview}...]\033[0m",
                  file=sys.stderr)

            tool_output = self.dispatcher.dispatch(name, args)
            self.messages.append({"role": "tool", "content": tool_output})

        # Pass 2: Stream final answer grounded in tool results
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

        self.messages.append(
            {"role": "assistant", "content": "".join(full_reply)}
        )
