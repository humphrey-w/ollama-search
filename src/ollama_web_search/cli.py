"""The entry point for the Ollama Search CLI."""
import argparse
import sys

try:
    # Enables command history and line editing in the REPL
    import readline  # pylint: disable=unused-import  # noqa: F401
except ImportError:
    pass

from ollama_web_search.config import DEFAULT_MODEL
from ollama_web_search.ollama_client import OllamaConnectionError
from ollama_web_search.orchestrator import Orchestrator

REPL_HELP = """Available commands:
  /help          Show this help message
  /clear, /reset Clear conversation history
  /history       Show message count in current session
  /search on|off Toggle SearXNG web search
  /exit, /quit   Exit interactive chat (or press Ctrl+D)"""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ask",
        description="Query a local Ollama model with SearXNG web search.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help=(
            "Optional prompt. Omit to start an interactive multi-turn chat "
            "session."
        ),
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help=(
            "Stay in interactive multi-turn chat mode after an initial prompt."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable SearXNG web search and query Ollama directly.",
    )
    return parser


def _execute_turn(orchestrator: Orchestrator, prompt: str) -> bool:
    """Stream a single turn to stdout. Returns False on fatal connection
    error."""
    try:
        for chunk in orchestrator.run_turn(prompt):
            print(chunk, end="", flush=True)
        print()
        return True
    except OllamaConnectionError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print("\n\033[90m[Generation interrupted]\033[0m", file=sys.stderr)
        return True


def _handle_slash_command(command: str, orchestrator: Orchestrator) -> bool:
    """Handle REPL slash commands. Returns False if the REPL should exit."""
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd in {"/exit", "/quit"}:
        return False
    if cmd in {"/clear", "/reset"}:
        orchestrator.reset()
        print("\033[90m[Conversation history cleared]\033[0m")
    elif cmd == "/history":
        print(
            f"\033[90m[Turns: {orchestrator.turn_count} | "
            f"Messages in context: {len(orchestrator.messages)}]\033[0m"
        )
    elif cmd == "/search":
        if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
            orchestrator.enable_search = parts[1].lower() == "on"
            state = "enabled" if orchestrator.enable_search else "disabled"
            print(f"\033[90m[Web search {state}]\033[0m")
        else:
            print("Usage: /search on|off")
    elif cmd == "/help":
        print(REPL_HELP)
    else:
        print(f"Unknown command '{cmd}'. Type /help for options.")
    return True


def run_repl(orchestrator: Orchestrator) -> int:
    """Run an interactive multi-turn terminal chat loop."""
    search_status = "on" if orchestrator.enable_search else "off"
    print(
        f"\033[1mOllama Chat\033[0m (model: {orchestrator.model},"
        f"search: {search_status}) "
        "— type /help for commands, /exit to quit.\n"
    )

    while True:
        try:
            user_input = input("\033[1;34m>>> \033[0m").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n\033[90m(Use /exit or Ctrl+D to quit)\033[0m")
            continue

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not _handle_slash_command(user_input, orchestrator):
                return 0
            continue

        _execute_turn(orchestrator, user_input)
        print()


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    orchestrator = Orchestrator(
        model=args.model,
        enable_search=not args.no_search,
    )

    initial_prompt = " ".join(args.prompt).strip()

    # Single-turn mode when a prompt is given without --chat
    if initial_prompt and not args.chat:
        return 0 if _execute_turn(orchestrator, initial_prompt) else 2

    # Multi-turn interactive mode
    if initial_prompt:
        if not _execute_turn(orchestrator, initial_prompt):
            return 2
        print()

    return run_repl(orchestrator)


if __name__ == "__main__":
    sys.exit(main())
