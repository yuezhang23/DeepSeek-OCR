import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Required API keys — raise immediately if missing
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY: str = os.environ["TAVILY_API_KEY"]

# Model settings
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
OCR_MODEL_NAME: str = os.getenv("OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
OCR_DTYPE: str = os.getenv("OCR_DTYPE", "bfloat16")

# Paths
CACHE_DIR: Path = Path(os.getenv("REVIEWER_CACHE_DIR", "/tmp/paper_reviewer_cache"))
HF_OCR_DIR: Path = (
    Path(__file__).parent.parent
    / "DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-hf"
)

# Pipeline tuning
NUM_SEARCH_QUERIES: int = int(os.getenv("NUM_SEARCH_QUERIES", "12"))
TOP_K_PAPERS: int = int(os.getenv("TOP_K_PAPERS", "12"))
MAX_FULL_TEXT_PAPERS: int = int(os.getenv("MAX_FULL_TEXT_PAPERS", "5"))
MAX_TAVILY_RESULTS: int = int(os.getenv("MAX_TAVILY_RESULTS", "5"))
DEFAULT_VENUE: str = os.getenv("DEFAULT_VENUE", "ICLR")
