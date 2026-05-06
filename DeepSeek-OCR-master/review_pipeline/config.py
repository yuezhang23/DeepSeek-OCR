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

# Concurrency — tune to stay within API rate limits
# Stage 4: parallel Tavily search workers (free tier: 1 req/s → keep ≤4)
TAVILY_SEARCH_WORKERS: int = int(os.getenv("TAVILY_SEARCH_WORKERS", "4"))
# Stage 8: parallel summary workers per paper (DeepSeek RPM limit → keep ≤8)
SUMMARY_WORKERS: int = int(os.getenv("SUMMARY_WORKERS", "6"))
# Batch mode: papers processed in parallel (each paper ~17 DeepSeek calls → keep ≤3)
PIPELINE_WORKERS: int = int(os.getenv("PIPELINE_WORKERS", "3"))


BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
