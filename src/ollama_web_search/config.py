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

SYSTEM_PROMPT = (
    "You are a concise, accurate CLI assistant. "
    "Use the web_search tool whenever a question requires current events, "
    "live data, or facts you are unsure about. "
    "When you use search results, cite the source URLs briefly at the end."
)
