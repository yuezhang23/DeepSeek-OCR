"""
Stage 6: Evaluate the relevance of candidate papers to the target paper using Claude.

All candidates are evaluated in a single API call to minimize round trips.
The paper markdown is cached in the system block.
"""
from __future__ import annotations

from typing import Optional, TypedDict

import anthropic

from paper_reviewer import config
from paper_reviewer.arxiv_client import PaperMetadata

_SYSTEM_PREAMBLE = """\
You are an expert academic paper reviewer. You will be given the full text of a \
research paper (the "target paper") and a list of candidate related papers \
(title + abstract only). Your task is to score each candidate's relevance to \
the target paper for the purpose of grounding a peer review.
"""

_RELEVANCE_TOOL = {
    "name": "submit_relevance_scores",
    "description": "Submit relevance scores for each candidate paper.",
    "input_schema": {
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
            }
        },
        "required": ["scores"],
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
    top_k: int = None,
    client: Optional[anthropic.Anthropic] = None,
) -> list[RelevanceScore]:
    """Score each candidate paper and return the top_k most relevant.

    Uses a single Claude call with all candidates in one human message.
    Paper markdown is placed in the (cached) system block.
    Returns list sorted by relevance_score descending.
    """
    top_k = top_k or config.TOP_K_PAPERS
    client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

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

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system=[
            {"type": "text", "text": _SYSTEM_PREAMBLE},
            {
                "type": "text",
                "text": paper_markdown,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_message}],
        tools=[_RELEVANCE_TOOL],
        tool_choice={"type": "tool", "name": "submit_relevance_scores"},
    )

    raw_scores: list[dict] = []
    for block in response.content:
        if block.type == "tool_use":
            raw_scores = block.input.get("scores", [])
            break

    # Attach titles from metadata and sort
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
