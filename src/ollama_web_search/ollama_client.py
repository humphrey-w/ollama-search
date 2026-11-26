"""Client for interacting with the local Ollama server."""
import json
import urllib.error
import urllib.request
from typing import Any, Iterator

from ollama_web_search.config import OLLAMA_BASE_URL, REQUEST_TIMEOUT


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def chat_stream(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Send a streaming chat request to Ollama and yield parsed NDJSON
    chunks."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT * 4)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise OllamaConnectionError(
            f"Ollama API error ({e.code}) at {url}: {detail or e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise OllamaConnectionError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}: {e.reason}"
        ) from e

    with resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            yield json.loads(raw_line.decode("utf-8"))


def _stream_tokens(resp: Any) -> Iterator[str]:
    with resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            chunk = json.loads(raw_line.decode("utf-8"))
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
