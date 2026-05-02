"""
Stage 3: Generate tiered arXiv search queries from the paper markdown using Claude.

Uses tool use for guaranteed structured JSON output and prompt caching on the
paper markdown to amortize token cost across the pipeline.
"""
from __future__ import annotations

from typing import Optional

import anthropic
from review_pipeline import config

_SYSTEM_PREAMBLE = """\
You are an expert academic paper analyst helping to ground a paper review in prior work.
You will be given the full text of a research paper (converted from PDF to Markdown).
Your task: analyze the paper and generate arXiv search queries to find the most relevant related work.
"""

_QUERY_TOOL = {
    "name": "submit_search_queries",
    "description": "Submit the generated arXiv search queries for the paper review pipeline.",
    "input_schema": {
        "type": "object",
        "properties": {
            "benchmark_queries": {
                "type": "array",
                "description": "Queries targeting benchmarks and baselines used or mentioned in the paper.",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
            "problem_queries": {
                "type": "array",
                "description": "Queries targeting papers that address the same problem or task.",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
            "technique_queries": {
                "type": "array",
                "description": "Queries targeting papers that use related techniques or methods.",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
        },
        "required": ["benchmark_queries", "problem_queries", "technique_queries"],
    },
}


def generate_search_queries(
    paper_markdown: str,
    venue: str = "ICLR",
    num_queries: int = 12,
    client: Optional[anthropic.Anthropic] = None,
) -> list[str]:
    """Analyze the paper and return a flat list of arXiv search queries.

    Produces three tiers: benchmark/baseline queries, same-problem queries,
    and related-technique queries. All queries are phrased to target arXiv.
    Uses prompt caching on the paper markdown.
    """
    client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    per_bucket = max(3, num_queries // 3)
    user_message = (
        f"The paper above is intended for submission to {venue}.\n"
        f"Generate {per_bucket} search queries per category (benchmark_queries, "
        f"problem_queries, technique_queries). Each query should be a concise phrase "
        f"suitable for searching arXiv (include 'arxiv' or technical keywords that "
        f"will surface arXiv papers). Avoid overly generic terms."
    )

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=[
            {"type": "text", "text": _SYSTEM_PREAMBLE},
            {
                "type": "text",
                "text": paper_markdown,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_message}],
        tools=[_QUERY_TOOL],
        tool_choice={"type": "tool", "name": "submit_search_queries"},
    )

    tool_input = _extract_tool_input(response)
    queries = (
        tool_input.get("benchmark_queries", [])
        + tool_input.get("problem_queries", [])
        + tool_input.get("technique_queries", [])
    )
    return queries


def _extract_tool_input(response: anthropic.types.Message) -> dict:
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("Claude did not return a tool_use block for query generation.")
