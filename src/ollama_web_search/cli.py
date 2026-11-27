"""The entry point for the Ollama Search CLI."""
import argparse
import os
import sys

try:
    # Enables command history and line editing in the REPL
    import readline  # pylint: disable=unused-import  # noqa: F401
except ImportError:
    pass

from ollama_web_search.config import DEFAULT_MODEL
from ollama_web_search.ollama_client import OllamaConnectionError
from ollama_web_search.orchestrator import Orchestrator
from ollama_web_search.models import Source, ToolResult

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text
except ImportError:
    Console = None
    Live = None
    Markdown = None
    Text = None

REPL_HELP = """Available commands:
  /help          Show this help message
  /clear, /reset Clear conversation history
  /history       Show message count in current session
  /search on|off Toggle web search
  /thinking on|off Toggle visibility of model <think> reasoning
  /exit, /quit   Exit interactive chat (or press Ctrl+D)"""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ask",
        description="Query a local Ollama model with web search.",
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
        help="Disable web search and query Ollama directly.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show model reasoning (<think> blocks) instead of hiding them.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable Rich terminal formatting.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors while keeping terminal formatting.",
    )
    return parser


def _add_source_citations(
    answer: str,
    sources: list[Source],
) -> str:
    """Convert source URLs cited by the model into numbered Markdown links."""
    cited = answer
    for index, source in enumerate(sources, start=1):
        cited = cited.replace(
            source.url,
            f"[{index}]({source.url})",
        )
    return cited


def _execute_turn(
    orchestrator: Orchestrator,
    prompt: str,
    *,
    plain: bool = False,
    no_color: bool = False,
) -> bool:
    """Stream a single turn to stdout. Returns False on fatal connection
    error."""
    use_rich = (
        not plain
        and Console is not None
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
    )
    rich_no_color = no_color or "NO_COLOR" in os.environ
    sources: list[Source] = []
    status_live = None

    if use_rich:
        console = Console(no_color=rich_no_color)
        status_console = Console(stderr=True, no_color=rich_no_color)

        def on_status(status: str) -> None:
            nonlocal status_live
            if status_live is not None:
                status_live.stop()
            status_live = Live(
                Text(status, style="dim"),
                console=status_console,
                transient=True,
                refresh_per_second=12,
            )
            status_live.start()

        def on_result(
            name: str,
            _args: dict,
            output: str | ToolResult,
        ) -> None:
            nonlocal status_live
            if status_live is not None:
                status_live.stop()
                status_live = None
            if isinstance(output, ToolResult):
                for source in output.sources:
                    if not any(
                        existing.url == source.url
                        for existing in sources
                    ):
                        sources.append(source)
                result_text = output.text
            else:
                result_text = output
            if "error" not in result_text.lower():
                status_console.print(
                    Text(f"✓ {name.replace('_', ' ').title()} complete",
                         style="dim")
                )
    else:
        def on_status(status: str) -> None:
            if no_color or plain:
                print(status, file=sys.stderr)
            else:
                print(f"\033[90m{status}\033[0m", file=sys.stderr)

        def on_result(
            name: str,
            _args: dict,
            output: str | ToolResult,
        ) -> None:
            del name
            if isinstance(output, ToolResult):
                for source in output.sources:
                    if not any(
                        existing.url == source.url
                        for existing in sources
                    ):
                        sources.append(source)

    orchestrator.on_tool_status = on_status
    orchestrator.on_tool_result = on_result

    try:
        if use_rich:
            console.print("[bold cyan]Assistant[/bold cyan]")
            answer = ""
            live = Live(
                Markdown(""),
                console=console,
                transient=True,
                refresh_per_second=12,
            )
            live.start()
            try:
                for chunk in orchestrator.run_turn(prompt):
                    answer += chunk
                    live.update(Markdown(answer), refresh=True)
            finally:
                live.stop()
                if status_live is not None:
                    status_live.stop()
                    status_live = None
            console.print(Markdown(_add_source_citations(answer, sources)))
        else:
            answer = ""
            for chunk in orchestrator.run_turn(prompt):
                answer += chunk
                print(chunk, end="", flush=True)
            print()

        if sources:
            if use_rich:
                console.print()
                console.print("[bold]Sources[/bold]")
                for index, source in enumerate(sources, start=1):
                    console.print(Markdown(
                        f"{index}. [{source.title}]({source.url})"
                    ))
            else:
                print("\nSources:")
                for index, source in enumerate(sources, start=1):
                    print(f"{index}. {source.title} — {source.url}")
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
    elif cmd == "/thinking":
        if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
            orchestrator.show_thinking = parts[1].lower() == "on"
            state = "shown" if orchestrator.show_thinking else "hidden"
            print(f"\033[90m[Model thinking {state}]\033[0m")
        else:
            print("Usage: /thinking on|off")
    else:
        print(f"Unknown command '{cmd}'. Type /help for options.")
    return True


def run_repl(
    orchestrator: Orchestrator,
    *,
    plain: bool = False,
    no_color: bool = False,
) -> int:
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

        _execute_turn(
            orchestrator,
            user_input,
            plain=plain,
            no_color=no_color,
        )
        print()


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    orchestrator = Orchestrator(
        model=args.model,
        enable_search=not args.no_search,
        show_thinking=args.show_thinking,
    )

    initial_prompt = " ".join(args.prompt).strip()

    # Single-turn mode when a prompt is given without --chat
    if initial_prompt and not args.chat:
        return (
            0
            if _execute_turn(
                orchestrator,
                initial_prompt,
                plain=args.plain,
                no_color=args.no_color,
            )
            else 2
        )

    # Multi-turn interactive mode
    if initial_prompt:
        if not _execute_turn(
            orchestrator,
            initial_prompt,
            plain=args.plain,
            no_color=args.no_color,
        ):
            return 2
        print()

    return run_repl(
        orchestrator,
        plain=args.plain,
        no_color=args.no_color,
    )


if __name__ == "__main__":
    sys.exit(main())
