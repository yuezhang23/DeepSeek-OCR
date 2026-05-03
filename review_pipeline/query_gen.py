"""
Stage 3: Generate tiered arXiv search queries from the paper markdown.

Uses DeepSeek (OpenAI-compatible API) with function calling for structured
output.  The paper markdown is passed in the system message.
"""
from __future__ import annotations

import json

from openai import OpenAI

from review_pipeline import config

_SYSTEM_PREAMBLE = """\
You are an expert academic paper analyst helping to ground a paper review in prior work.
You will be given the full text of a research paper (converted from PDF to Markdown).
Your task: analyze the paper and generate arXiv search queries to find the most relevant related work.
"""

# OpenAI-compatible function-calling tool definition
_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_search_queries",
        "description": "Submit the generated arXiv search queries for the paper review pipeline.",
        "parameters": {
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
    },
}


def generate_search_queries(
    paper_markdown: str,
    venue: str = "ICLR",
    num_queries: int = 12,
    client: OpenAI = None,
) -> list[str]:
    """Analyze the paper and return a flat list of arXiv search queries.

    Produces three tiers: benchmark/baseline queries, same-problem queries,
    and related-technique queries.
    """
    per_bucket = max(3, num_queries // 3)
    user_message = (
        f"The paper above is intended for submission to {venue}.\n"
        f"Generate {per_bucket} search queries per category (benchmark_queries, "
        f"problem_queries, technique_queries). Each query should be a concise phrase "
        f"suitable for searching arXiv. Avoid overly generic terms."
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PREAMBLE + "\n\n" + paper_markdown},
            {"role": "user", "content": user_message},
        ],
        tools=[_QUERY_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_search_queries"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    tool_input: dict = json.loads(tool_call.function.arguments)
    return (
        tool_input.get("benchmark_queries", [])
        + tool_input.get("problem_queries", [])
        + tool_input.get("technique_queries", [])
    )
