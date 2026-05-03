"""
Stage 6: Evaluate the relevance of candidate papers to the target paper.

All candidates are evaluated in a single API call to minimize round trips.
"""
from __future__ import annotations

import json
from typing import TypedDict

from openai import OpenAI

from review_pipeline import config
from review_pipeline.arxiv_client import PaperMetadata

_SYSTEM_PREAMBLE = """\
You are an expert academic paper reviewer. You will be given the full text of a research paper \
(the "target paper") and a list of candidate related papers (title + abstract only). Your task \
is to score each candidate's relevance to the target paper for the purpose of grounding a peer review."""

_RELEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_relevance_scores",
        "description": "Submit relevance scores for each candidate paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "arxiv_id": {"type": "string"},
                            "relevance_score": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "1=unrelated, 10=directly comparable / must-cite",
                            },
                            "relevance_reason": {
                                "type": "string",
                                "description": "One sentence explaining the relationship.",
                            },
                            "relationship_type": {
                                "type": "string",
                                "enum": ["baseline", "competitor", "technique", "related"],
                            },
                        },
                        "required": [
                            "arxiv_id",
                            "relevance_score",
                            "relevance_reason",
                            "relationship_type",
                        ],
                    },
                },
            },
            "required": ["scores"],
        },
    },
}


class RelevanceScore(TypedDict):
    arxiv_id: str
    title: str
    relevance_score: int
    relevance_reason: str
    relationship_type: str


def evaluate_relevance(
    paper_markdown: str,
    candidates: dict[str, PaperMetadata],
    client: OpenAI | None = None,
    top_k: int = None,
) -> list[RelevanceScore]:
    """Score each candidate paper and return the top_k most relevant.

    Uses a single API call with all candidates in one user message.
    Returns list sorted by relevance_score descending.
    """
    top_k = top_k or config.TOP_K_PAPERS
    if client is None:
        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    if not candidates:
        return []

    candidate_lines = []
    for i, (arxiv_id, meta) in enumerate(candidates.items(), 1):
        authors_str = ", ".join(meta["authors"][:3])
        if len(meta["authors"]) > 3:
            authors_str += " et al."
        candidate_lines.append(
            f"[{i}] arxiv_id={arxiv_id}\n"
            f"Title: {meta['title']}\n"
            f"Authors: {authors_str} ({meta['published']})\n"
            f"Abstract: {meta['abstract']}\n"
        )

    user_message = (
        "Score each of the following candidate papers for relevance to the target paper "
        "above. Use the submit_relevance_scores tool.\n\n"
        + "\n".join(candidate_lines)
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM_PREAMBLE + "\n\n" + paper_markdown},
            {"role": "user", "content": user_message},
        ],
        tools=[_RELEVANCE_TOOL],
        tool_choice="auto",
        extra_body={"thinking_mode": "thinking"},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    raw_scores: list[dict] = json.loads(tool_call.function.arguments).get("scores", [])

    results: list[RelevanceScore] = []
    for s in raw_scores:
        arxiv_id = s["arxiv_id"]
        meta = candidates.get(arxiv_id, {})
        results.append(
            RelevanceScore(
                arxiv_id=arxiv_id,
                title=meta.get("title", ""),
                relevance_score=s["relevance_score"],
                relevance_reason=s["relevance_reason"],
                relationship_type=s["relationship_type"],
            )
        )

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]
