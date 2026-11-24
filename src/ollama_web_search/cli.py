"""The entry point for the Ollama Search CLI."""
import sys
import argparse

from ollama_web_search.config import DEFAULT_MODEL
from ollama_web_search.ollama_client import OllamaConnectionError
from ollama_web_search.orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ask",
        description="Query a local Ollama model with web search.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="The question or prompt to send to the model.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        parser.print_help()
        return 1

    orchestrator = Orchestrator(
        model=args.model,
        enable_search=not args.no_search,
    )

    try:
        for chunk in orchestrator.run_turn(prompt):
            print(chunk, end="", flush=True)
        print()
        return 0
    except OllamaConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
