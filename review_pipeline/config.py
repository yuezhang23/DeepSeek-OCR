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

# ── Generic LLM provider (OpenAI-compatible Chat Completions API) ────────────
# The generation/scoring stages talk to ANY vendor that exposes an
# OpenAI-compatible endpoint — no code changes needed, just point these vars at
# the vendor: DeepSeek, OpenAI, OpenRouter, Together, Groq, Fireworks, Mistral,
# xAI, or a local server (vLLM, Ollama, LM Studio). Each falls back to the
# legacy DEEPSEEK_* values so existing setups keep working unchanged.
# Which SDK/protocol the (analysis-script) LLM calls use to reach the model:
#   "openai"    — OpenAI-compatible Chat Completions (DeepSeek, OpenAI, vLLM, …)
#   "anthropic" — native Anthropic Messages API (uses ANTHROPIC_API_KEY + CLAUDE_MODEL)
# This is distinct from LLM_PROVIDER below, which only shapes reasoning params
# within the OpenAI-compatible path.
LLM_VENDOR: str = os.getenv("LLM_VENDOR", "openai").strip().lower()
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "") or DEEPSEEK_API_KEY
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "") or DEEPSEEK_BASE_URL
LLM_MODEL: str = os.getenv("LLM_MODEL", "") or DEEPSEEK_MODEL
# Chain-of-thought / reasoning control:
#   "auto" — enable reasoning only for providers known to support it
#   "on"   — always request reasoning (reasoning-capable models only)
#   "off"  — never request reasoning
LLM_THINKING: str = os.getenv("LLM_THINKING", "auto").strip().lower()
# Providers that accept DeepSeek/OpenAI-style reasoning params under "auto".
LLM_THINKING_PROVIDERS: set[str] = {
    p.strip().lower()
    for p in os.getenv("LLM_THINKING_PROVIDERS", "deepseek,openai").split(",")
    if p.strip()
}

# Pipeline stages that issue OpenAI-compatible LLM calls. Each may target its
# own vendor independently via LLM_<STAGE>_* env vars (PROVIDER / API_KEY /
# BASE_URL / MODEL / THINKING); anything unset inherits the global LLM_* default
# above. Stage keys mirror the function-call stages (tools.py): e.g. set
# LLM_REVIEW_MODEL / LLM_REVIEW_BASE_URL to run only the review stage elsewhere.
LLM_STAGES: tuple[str, ...] = ("query", "relevance", "plan", "summary", "review", "score")


def apply_llm_overrides(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    vendor: str | None = None,
    anthropic_api_key: str | None = None,
) -> None:
    """Override the global default LLM settings at runtime (e.g. from CLI flags).

    Per-stage env overrides (LLM_<STAGE>_*) still take precedence over these
    globals — see llm_stage_settings(). Empty/None values are ignored.

    ``vendor`` selects the SDK/protocol (``"openai"`` vs ``"anthropic"``) and
    ``anthropic_api_key`` overrides the native-Claude key; the rest tune the
    OpenAI-compatible path.
    """
    global LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_THINKING
    global LLM_VENDOR, ANTHROPIC_API_KEY
    if provider:
        LLM_PROVIDER = provider.strip().lower()
    if api_key:
        LLM_API_KEY = api_key
    if base_url:
        LLM_BASE_URL = base_url
    if model:
        LLM_MODEL = model
    if thinking:
        LLM_THINKING = thinking.strip().lower()
    if vendor:
        LLM_VENDOR = vendor.strip().lower()
    if anthropic_api_key:
        ANTHROPIC_API_KEY = anthropic_api_key


def llm_stage_settings(stage: str) -> dict[str, str]:
    """Resolve (provider, api_key, base_url, model, thinking) for one stage.

    Reads LLM_<STAGE>_* env vars, each falling back to the global LLM_* default,
    so a stage only needs env overrides for the fields it wants to change.
    Reads the globals at call time so runtime apply_llm_overrides() is honored.
    """
    s = stage.upper()
    return {
        "provider": os.getenv(f"LLM_{s}_PROVIDER", "").strip().lower() or LLM_PROVIDER,
        "api_key": os.getenv(f"LLM_{s}_API_KEY", "") or LLM_API_KEY,
        "base_url": os.getenv(f"LLM_{s}_BASE_URL", "") or LLM_BASE_URL,
        "model": os.getenv(f"LLM_{s}_MODEL", "") or LLM_MODEL,
        "thinking": os.getenv(f"LLM_{s}_THINKING", "").strip().lower() or LLM_THINKING,
    }

# Paths
CACHE_DIR: Path = Path(os.getenv("REVIEWER_CACHE_DIR", "/tmp/paper_reviewer_cache"))

# Pipeline tuning
NUM_SEARCH_QUERIES: int = int(os.getenv("NUM_SEARCH_QUERIES", "10"))
TOP_K_PAPERS: int = int(os.getenv("TOP_K_PAPERS", "10"))
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
