import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API keys — read from environment / .env file
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

# Model settings
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Paths
CACHE_DIR: Path = Path(os.getenv("REVIEWER_CACHE_DIR", "/tmp/paper_reviewer_cache"))

# Pipeline tuning
NUM_SEARCH_QUERIES: int = int(os.getenv("NUM_SEARCH_QUERIES", "12"))
TOP_K_PAPERS: int = int(os.getenv("TOP_K_PAPERS", "12"))
MAX_FULL_TEXT_PAPERS: int = int(os.getenv("MAX_FULL_TEXT_PAPERS", "5"))
MAX_TAVILY_RESULTS: int = int(os.getenv("MAX_TAVILY_RESULTS", "5"))
DEFAULT_VENUE: str = os.getenv("DEFAULT_VENUE", "ICLR")
