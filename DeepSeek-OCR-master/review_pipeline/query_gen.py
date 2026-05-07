"""
Stage 3: Generate tiered arXiv search queries from the paper markdown.

Uses DeepSeek (OpenAI-compatible API) with function calling for structured
output.  The paper markdown is passed in the system message.
"""
from __future__ import annotations

import json

from openai import OpenAI

from review_pipeline import config
from review_pipeline.clients import deepseek_chat
from review_pipeline.tools import QUERY_TOOL as _QUERY_TOOL

_SYSTEM_PREAMBLE = """\
You are an expert academic paper analyst helping to ground a paper review in prior work.
You will be given the full text of a research paper (converted from PDF to Markdown).
Your task: analyze the paper and generate arXiv search queries to find the most relevant related work.
"""


def generate_search_queries(
    paper_markdown: str,
    venue: str = "ICLR",
    num_queries: int = 12,
    client: OpenAI | None = None,
) -> list[str]:
    """Analyze the paper and return a flat list of arXiv search queries.

    Produces three tiers: benchmark/baseline queries, same-problem queries,
    and related-technique queries.
    """
    per_bucket = max(3, num_queries // 3)
    if client is None:
        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    user_message = (
        f"The paper above is intended for submission to {venue}.\n"
        f"Generate {per_bucket} search queries per category (benchmark_queries, "
        f"problem_queries, technique_queries). Each query should be a concise phrase "
        f"suitable for searching arXiv. Avoid overly generic terms."
    )

    response = deepseek_chat(
        client,
        system=_SYSTEM_PREAMBLE + "\n\n" + paper_markdown,
        user=user_message,
        max_tokens=1024,
        tools=[_QUERY_TOOL],
    )

    tool_call = response.choices[0].message.tool_calls[0]
    tool_input: dict = json.loads(tool_call.function.arguments)
    return (
        tool_input.get("benchmark_queries", [])
        + tool_input.get("problem_queries", [])
        + tool_input.get("technique_queries", [])
    )
    # return response.choices[0].message.content.strip().split("\n")
