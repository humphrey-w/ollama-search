"""HTML scraper tool for Ollama Search."""
from html.parser import HTMLParser
import re
import urllib.error
import urllib.parse
import urllib.request

from ollama_web_search.config import MAX_PAGE_CHARS, REQUEST_TIMEOUT
from ollama_web_search.models import Source, ToolResult

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
        self._title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() in _IGNORED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() in _IGNORED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                if self._in_title:
                    self._title_parts.append(text)
                else:
                    self._chunks.append(text)

    def get_text(self) -> str:
        """Return the extracted text with normalized whitespace."""
        joined = "\n".join(self._chunks)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()

    def get_title(self) -> str:
        """Return the normalized HTML document title."""
        return re.sub(r"\s+", " ", " ".join(self._title_parts)).strip()


def fetch_page(
    url: str,
    max_chars: int = MAX_PAGE_CHARS,
) -> ToolResult:
    """Download a webpage and return text plus source metadata."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ToolResult(
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
                return ToolResult(
                    f"Fetch error: Unsupported content type '{content_type}' "
                    f"at {url}."
                )

            raw_bytes = resp.read(_MAX_DOWNLOAD_BYTES)
            charset = resp.headers.get_content_charset() or "utf-8"
            html_text = raw_bytes.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        return ToolResult(f"Fetch error (HTTP {e.code}) for {url}: {e.reason}")
    except urllib.error.URLError as e:
        return ToolResult(f"Fetch error (unreachable) for {url}: {e.reason}")
    except TimeoutError:
        return ToolResult(f"Fetch error: Timed out fetching {url}.")

    page_title = ""
    if "text/plain" in content_type:
        cleaned = html_text.strip()
    else:
        parser = _TextExtractor()
        parser.feed(html_text)
        cleaned = parser.get_text()
        page_title = parser.get_title()

    if not cleaned:
        return ToolResult(f"No readable text extracted from {url}.")

    if len(cleaned) > max_chars:
        cleaned = (
            f"{cleaned[:max_chars]}\n\n"
            f"[Content truncated at {max_chars} characters]"
        )

    title = page_title or url.strip()
    return ToolResult(
        text=f"URL: {url}\n\n{cleaned}",
        sources=(Source(title=title, url=url.strip(), snippet=cleaned[:300]),),
    )
