"""
Initialized API clients for the review pipeline.

PipelineClients is created once in main.py from CLI-supplied keys and passed
through run_pipeline to each stage function.  No module outside this file
should construct API clients directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
from openai import OpenAI
from tavily import TavilyClient

from review_pipeline import config


def deepseek_chat(
    client: OpenAI,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    thinking: bool = True,
):
    """Thin wrapper around client.chat.completions.create with DeepSeek defaults.

    Args:
        tools:       Pass a list to enable function calling; omit or None to skip.
        tool_choice: Only used when tools is provided. Defaults to "auto".
        thinking:    Set True to enable chain-of-thought (thinking_mode=thinking).
                     Set False for simple completion calls (e.g. short summaries).
    """
    kwargs: dict[str, Any] = dict(
        model=config.DEEPSEEK_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if thinking:
        kwargs["extra_body"] = {"thinking": "enabled"}
    return client.chat.completions.create(**kwargs)


@dataclass
class PipelineClients:
    """Holds all initialized API clients for one pipeline run.

    Stage assignment (default):
      deepseek — Stages 3, 6, 7, 8, 9  (all LLM stages)
      tavily   — Stage 4                (arXiv search)
      claude   — available for override if needed
    """
    claude: anthropic.Anthropic
    deepseek: OpenAI
    tavily: TavilyClient

    @classmethod
    def build(
        cls,
        anthropic_key: str | None = None,
        deepseek_key: str | None = None,
        tavily_key: str | None = None,
    ) -> "PipelineClients":
        """Construct clients from explicit keys, falling back to config / .env values."""
        return cls(
            claude=anthropic.Anthropic(
                api_key=anthropic_key or config.ANTHROPIC_API_KEY
            ),
            deepseek=OpenAI(
                api_key=deepseek_key or config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
            ),
            tavily=TavilyClient(api_key=tavily_key or config.TAVILY_API_KEY),
        )
