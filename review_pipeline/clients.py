"""
Initialized API clients for the review pipeline.

PipelineClients is created once in main.py from CLI-supplied keys and passed
through run_pipeline to each stage function.  No module outside this file
should construct API clients directly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from openai import OpenAI
from tavily import TavilyClient

from review_pipeline import config

logger = logging.getLogger(__name__)


def deepseek_chat(
    client: OpenAI,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    thinking: bool = True,
    max_retries: int = 3,
):
    """Wrapper around client.chat.completions.create with DeepSeek defaults and retry.

    Args:
        tools:       Pass a list to enable function calling; omit or None to skip.
        tool_choice: Only used when tools is provided. Defaults to "auto".
        thinking:    True enables chain-of-thought (thinking_mode=thinking).
                     False for plain completion calls (e.g. short summaries).
        max_retries: When tools are provided and the model returns no tool call,
                     retry up to this many times before raising.
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
        kwargs["reasoning_effort"] = "medium"  # Controls the thinking depth
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}} # Explicitly turn thinking on

    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(**kwargs)
        if tools is None:
            return response
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            return response
        logger.warning(
            "Model returned no tool call (attempt %d/%d). Content: %s",
            attempt, max_retries,
            (response.choices[0].message.content or "")[:200],
        )
        if attempt < max_retries:
            time.sleep(2 ** attempt)

    raise ValueError(
        f"Model did not return a tool call after {max_retries} attempts. "
        f"Last content: {(response.choices[0].message.content or '')[:300]}"
    )


def get_tool_call(response):
    """Extract the first tool call from a response, raising clearly if absent."""
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        content = (response.choices[0].message.content or "")[:300]
        raise ValueError(f"Model returned no tool call. Content: {content}")
    return tool_calls[0]


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
