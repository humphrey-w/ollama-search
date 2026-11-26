"""HTML scraper tool for Ollama Search."""
from html.parser import HTMLParser
import re
import urllib.error
import urllib.parse
import urllib.request

from ollama_web_search.config import MAX_PAGE_CHARS, REQUEST_TIMEOUT

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetch and extract the main text content of a webpage URL when "
            "search snippets do not contain enough detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full http:// or https:// URL to fetch",
                }
            },
            "required": ["url"],
        },
    },
}

_IGNORED_TAGS = {"script", "style", "noscript", "svg", "nav", "footer",
                 "header"}
_MAX_DOWNLOAD_BYTES = 512 * 1024  # 512 KB safety cap


class _TextExtractor(HTMLParser):
    """Strips HTML tags and skips non-content blocks like scripts and
    styles."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _IGNORED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _IGNORED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self) -> str:
        """Return the extracted text with normalized whitespace."""
        joined = "\n".join(self._chunks)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def fetch_page(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Download a webpage and return cleaned plain text truncated to
    max_chars."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return (
            f"Fetch error: Invalid URL '{url}'. Must start with "
            "http:// or https://."
        )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; OllamaSearchCLI/0.1)",
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            supported_types = ("text/html", "text/plain", "application/xhtml")
            if not any(t in content_type for t in supported_types):
                return (
                    f"Fetch error: Unsupported content type '{content_type}' "
                    f"at {url}."
                )

            raw_bytes = resp.read(_MAX_DOWNLOAD_BYTES)
            charset = resp.headers.get_content_charset() or "utf-8"
            html_text = raw_bytes.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        return f"Fetch error (HTTP {e.code}) for {url}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Fetch error (unreachable) for {url}: {e.reason}"
    except TimeoutError:
        return f"Fetch error: Timed out fetching {url}."

    if "text/plain" in content_type:
        cleaned = html_text.strip()
    else:
        parser = _TextExtractor()
        parser.feed(html_text)
        cleaned = parser.get_text()

    if not cleaned:
        return f"No readable text extracted from {url}."

    if len(cleaned) > max_chars:
        cleaned = (
            f"{cleaned[:max_chars]}\n\n"
            f"[Content truncated at {max_chars} characters]"
        )

    return f"URL: {url}\n\n{cleaned}"
