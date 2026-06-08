"""
Initialized API clients for the review pipeline.

PipelineClients is created once in main.py from CLI-supplied keys and passed
through run_pipeline to each stage function.  No module outside this file
should construct API clients directly.

The generation/scoring LLM is reached through the OpenAI-compatible Chat
Completions API, so any vendor exposing such an endpoint works without code
changes — see config.LLM_* (provider, base URL, model, api key). DeepSeek is
the default; OpenAI, OpenRouter, Together, Groq, Fireworks, Mistral, xAI and
local servers (vLLM, Ollama, LM Studio) all speak the same protocol.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from openai import OpenAI
from tavily import TavilyClient

from review_pipeline import config

logger = logging.getLogger(__name__)


def build_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Construct the OpenAI-compatible client for the configured LLM vendor.

    Centralizes the api_key/base_url resolution so every entry point (the
    pipeline and the standalone analysis scripts) talks to the same vendor.
    Falls back to config.LLM_* (which themselves fall back to the legacy
    DEEPSEEK_* values).
    """
    return OpenAI(
        api_key=api_key or config.LLM_API_KEY,
        base_url=base_url or config.LLM_BASE_URL,
    )


def _apply_thinking(kwargs: dict[str, Any], provider: str) -> None:
    """Add reasoning/thinking params appropriate to the vendor.

    Vendors differ in how they expose chain-of-thought:
      - deepseek: `reasoning_effort` + `extra_body={"thinking": {...}}`
      - openai (and most reasoning models): plain `reasoning_effort`
    Anything else gets only `reasoning_effort`, which OpenAI-compatible servers
    generally ignore when unsupported.
    """
    kwargs["reasoning_effort"] = "medium"  # Controls the thinking depth
    if provider == "deepseek":
        # Explicitly turn DeepSeek thinking on.
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}


def _thinking_requested(thinking: bool, provider: str, mode: str) -> bool:
    """Decide whether to send reasoning params, given the thinking mode."""
    if mode == "off" or not thinking:
        return False
    if mode == "on":
        return True
    # "auto": only providers known to accept reasoning params.
    return provider in config.LLM_THINKING_PROVIDERS


def llm_chat(
    client: OpenAI,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    thinking: bool = True,
    max_retries: int = 3,
    model: str | None = None,
    provider: str | None = None,
    thinking_mode: str | None = None,
):
    """Vendor-agnostic wrapper around client.chat.completions.create with retry.

    Works with any OpenAI-compatible vendor; the model and provider default to
    config.LLM_MODEL / config.LLM_PROVIDER but can be overridden per call.
    Prefer LLMVendor.chat() in pipeline stages — it binds the right client,
    model, provider, and thinking mode together so callers can't mismatch them.

    Args:
        tools:        Pass a list to enable function calling; omit or None to skip.
        tool_choice:  Only used when tools is provided. Defaults to "auto".
        thinking:     True requests chain-of-thought; whether it's actually sent
                      depends on the provider and the thinking mode.
        max_retries:  When tools are provided and the model returns no tool call,
                      retry up to this many times before raising.
        model:        Override the model name (default config.LLM_MODEL).
        provider:     Override the vendor key used to shape reasoning params
                      (default config.LLM_PROVIDER).
        thinking_mode: "auto"/"on"/"off" (default config.LLM_THINKING).
    """
    model = model or config.LLM_MODEL
    provider = (provider or config.LLM_PROVIDER).strip().lower()
    thinking_mode = (thinking_mode or config.LLM_THINKING).strip().lower()

    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if _thinking_requested(thinking, provider, thinking_mode):
        _apply_thinking(kwargs, provider)

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


# Backward-compatible alias: existing stages import `deepseek_chat`.
deepseek_chat = llm_chat


def get_tool_call(response):
    """Extract the first tool call from a response, raising clearly if absent."""
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        content = (response.choices[0].message.content or "")[:300]
        raise ValueError(f"Model returned no tool call. Content: {content}")
    return tool_calls[0]


@dataclass
class LLMVendor:
    """One OpenAI-compatible LLM target: client + model + provider + reasoning.

    Bundling these together means a pipeline stage never has to guess which
    model a bare client points at, or which reasoning params the vendor accepts
    — everything needed for a call travels with the object. Each stage gets its
    own LLMVendor (see PipelineClients), so stages can run on different vendors.
    """
    provider: str
    model: str
    base_url: str
    api_key: str = field(default="", repr=False)
    thinking: str = "auto"  # auto / on / off
    _client: OpenAI | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> OpenAI:
        """Lazily construct (and reuse) the underlying OpenAI-compatible client."""
        if self._client is None:
            self._client = build_llm_client(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @classmethod
    def default(cls) -> "LLMVendor":
        """The global default vendor (config.LLM_*)."""
        return cls(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            thinking=config.LLM_THINKING,
        )

    @classmethod
    def for_stage(cls, stage: str) -> "LLMVendor":
        """Resolve the vendor for a named stage (LLM_<STAGE>_* with global fallback)."""
        s = config.llm_stage_settings(stage)
        return cls(
            provider=s["provider"],
            model=s["model"],
            base_url=s["base_url"],
            api_key=s["api_key"],
            thinking=s["thinking"],
        )

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        tools: list[dict] | None = None,
        tool_choice: Any = "auto",
        thinking: bool = True,
        max_retries: int = 3,
    ):
        """Run a chat completion on this vendor. See llm_chat for arg semantics."""
        return llm_chat(
            self.client,
            system=system,
            user=user,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            max_retries=max_retries,
            model=self.model,
            provider=self.provider,
            thinking_mode=self.thinking,
        )

    def describe(self) -> str:
        """Human-readable one-liner for logging which vendor a stage uses."""
        return f"{self.provider}:{self.model} @ {self.base_url}"


@dataclass
class PipelineClients:
    """Holds all initialized API clients for one pipeline run.

    Stage assignment (default):
      vendors — one LLMVendor per LLM stage (3, 6, 7, 8, 9); each can target a
                different OpenAI-compatible vendor via config.LLM_<STAGE>_*.
      tavily  — Stage 4                (arXiv search)
      claude  — available for override if needed
    """
    claude: anthropic.Anthropic
    tavily: TavilyClient
    vendors: dict[str, LLMVendor]  # keyed by stage name; "default" is the fallback

    def vendor(self, stage: str) -> LLMVendor:
        """Return the LLMVendor for a stage, falling back to the default vendor."""
        return self.vendors.get(stage) or self.vendors["default"]

    @property
    def llm(self) -> OpenAI:
        """Backward-compatible default OpenAI client."""
        return self.vendors["default"].client

    @property
    def deepseek(self) -> OpenAI:
        """Backward-compatible alias for the default LLM client."""
        return self.llm

    @classmethod
    def build(
        cls,
        anthropic_key: str | None = None,
        tavily_key: str | None = None,
        llm_key: str | None = None,
        llm_base_url: str | None = None,
        deepseek_key: str | None = None,  # legacy alias for llm_key
    ) -> "PipelineClients":
        """Construct clients from config (apply any CLI overrides via
        config.apply_llm_overrides first). Legacy global key/base-url overrides
        passed here are folded into config for backward compatibility."""
        config.apply_llm_overrides(
            api_key=llm_key or deepseek_key,
            base_url=llm_base_url,
        )
        vendors = {"default": LLMVendor.default()}
        for stage in config.LLM_STAGES:
            vendors[stage] = LLMVendor.for_stage(stage)
        return cls(
            claude=anthropic.Anthropic(
                api_key=anthropic_key or config.ANTHROPIC_API_KEY
            ),
            tavily=TavilyClient(api_key=tavily_key or config.TAVILY_API_KEY),
            vendors=vendors,
        )
