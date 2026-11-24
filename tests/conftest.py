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
