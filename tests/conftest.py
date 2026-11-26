"""Pytest fixtures for the integration tests."""
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest


class _MockSearXNGHandler(BaseHTTPRequestHandler):
    """Lightweight HTTP handler simulating SearXNG's /search JSON endpoint."""

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Responds to GET requests with a mock SearXNG JSON payload."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path != "/search" or params.get("format") != ["json"]:
            self.send_response(400)
            self.end_headers()
            return

        query = params.get("q", [""])[0]

        if query == "empty_query":
            payload = {"query": query, "results": []}
        else:
            payload = {
                "query": query,
                "results": [
                    {
                        "title": f"Docs for {query}",
                        "url": "https://example.org",
                        "content": f"Detailed guide about {query}.",
                    },
                    {
                        "title": "Secondary Source",
                        "url": "https://example.org",
                        "content": "Additional context.",
                    },
                ],
            }

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # pylint: disable=redefined-builtin
        # Suppress HTTP access logs during pytest execution
        return


class _MockOllamaHandler(BaseHTTPRequestHandler):
    """Lightweight HTTP handler simulating Ollama's /api/chat endpoint."""

    captured_histories: list[list[dict]] = []

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Responds to POST /api/chat requests with tool calls or streamed
        tokens."""
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = payload.get("messages", [])
        _MockOllamaHandler.captured_histories.append(messages)

        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        has_tool_result = any(m["role"] == "tool" for m in messages)

        # Turn 1, Pass 1: Trigger web_search
        if len(user_turns) == 1 and not has_tool_result:
            body = json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "web_search",
                                    "arguments": {"query": user_turns[0]},
                                }
                            }
                        ],
                    }
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        # Turn 1, Pass 2: Stream grounded answer
        elif len(user_turns) == 1 and has_tool_result:
            chunk = json.dumps(
                {
                    "message": {
                        "role": "assistant", "content": "Kernel is 6.12."
                    },
                    "done": True,
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        # Turn 2+: Direct reply demonstrating multi-turn context retention
        else:
            body = json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "content": f"You previously asked: {user_turns[0]}",
                    }
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # pylint: disable=redefined-builtin
        return


class _MockWebPageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Serve a simple HTML page or binary content for testing."""
        if self.path == "/article":
            html = (
                "<html><head><title>Test</title><style>body{color:red;}"
                "</style></head><body><nav>Menu</nav><h1>Release Notes</h1>"
                "<p>Linux 6.12 introduces sched_ext support.</p>"
                "<script>console.log('ignore me');</script></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/binary":
            data = b"\x89PNG\r\n\x1a\n"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # pylint: disable=redefined-builtin
        return


@pytest.fixture
def mock_web_url() -> Iterator[str]:
    """Start a temporary HTTP server serving mock pages for testing."""
    server = HTTPServer(("127.0.0.1", 0), _MockWebPageHandler)
    _, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def mock_ollama_url() -> Iterator[tuple[str, list[list[dict]]]]:
    """Starts a temporary local HTTP server mimicking Ollama and yields
    (base_url, history_log)."""
    _MockOllamaHandler.captured_histories.clear()
    server = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    _, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{port}", _MockOllamaHandler.captured_histories
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def mock_searxng_url() -> Iterator[str]:
    """Starts a temporary local HTTP server mimicking SearXNG and yields its
    base URL."""
    server = HTTPServer(("127.0.0.1", 0), _MockSearXNGHandler)
    _, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
