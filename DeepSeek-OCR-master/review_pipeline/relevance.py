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
from review_pipeline.clients import deepseek_chat
from review_pipeline.tools import RELEVANCE_TOOL as _RELEVANCE_TOOL

_SYSTEM_PREAMBLE = """\
You are an expert academic paper reviewer. You will be given the full text of a research paper \
(the "target paper") and a list of candidate related papers (title + abstract only). Your task \
is to score each candidate's relevance to the target paper for the purpose of grounding a peer review."""


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

    response = deepseek_chat(
        client,
        system=_SYSTEM_PREAMBLE + "\n\n" + paper_markdown,
        user=user_message,
        max_tokens=4096,
        tools=[_RELEVANCE_TOOL],
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return []
    tool_call = tool_calls[0]
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
