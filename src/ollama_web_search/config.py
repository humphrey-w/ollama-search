"""Configuration settings for the Ollama Search package."""
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "6000"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "4"))

SYSTEM_PROMPT = (
    "You are a concise, accurate CLI assistant. "
    "Use web_search to find relevant pages for current events or new facts. "
    "If search snippets lack enough detail, use fetch_page on a promising URL "
    "to read the full content. "
    "When citing search results, include the source URL and briefly identify "
    "the source."
)
