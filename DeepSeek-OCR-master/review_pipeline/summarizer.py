"""
Stages 7 + 8: Plan which related papers get full-text vs abstract-only treatment,
then generate summaries for each.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, TypedDict

from openai import OpenAI

from review_pipeline import config
from review_pipeline.arxiv_client import PaperMetadata, download
from review_pipeline.relevance import RelevanceScore
from review_pipeline.ocr import convert_pdf_to_markdown

logger = logging.getLogger(__name__)

_SYSTEM_PREAMBLE = """\
You are an expert academic paper analyst. You will be given the full text of a \
research paper (the "target paper") and are helping to summarize related work \
for a peer review. Summaries should be accurate, concise, and focused on aspects \
that are most relevant to evaluating the target paper.
"""

_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_summarization_plan",
        "description": "Submit the summarization plan for each related paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "plans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "arxiv_id": {"type": "string"},
                            "method": {
                                "type": "string",
                                "enum": ["abstract_only", "full_text"],
                            },
                            "focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "For full_text papers: list specific aspects to focus on.",
                            },
                        },
                        "required": ["arxiv_id", "method", "focus_areas"],
                    },
                },
            },
            "required": ["plans"],
        },
    },
}


class SummarizationPlan(TypedDict):
    arxiv_id: str
    method: Literal["abstract_only", "full_text"]
    focus_areas: list[str]


class PaperSummary(TypedDict):
    arxiv_id: str
    title: str
    summary: str


def plan_summarization(
    paper_markdown: str,
    ranked_papers: list[RelevanceScore],
    client: OpenAI,
    max_full_text: int = None,
) -> list[SummarizationPlan]:
    """Decide summarization method for each ranked paper.

    Assigns 'full_text' to the most relevant papers (up to max_full_text)
    and 'abstract_only' to the rest, specifying focus areas for full_text papers.
    """
    max_full_text = max_full_text if max_full_text is not None else config.MAX_FULL_TEXT_PAPERS

    if not ranked_papers:
        return []

    paper_list = "\n".join(
        f"[{i+1}] arxiv_id={p['arxiv_id']} (score={p['relevance_score']}, "
        f"type={p['relationship_type']})\nTitle: {p['title']}\nReason: {p['relevance_reason']}"
        for i, p in enumerate(ranked_papers)
    )

    user_message = (
        f"For each related paper below, decide whether to use 'abstract_only' or 'full_text' "
        f"summarization. Assign 'full_text' to at most {max_full_text} of the most important "
        f"papers (higher relevance score = higher priority). For 'full_text' papers, specify "
        f"2-4 focus areas (e.g. 'training methodology', 'benchmark results', 'architecture').\n\n"
        + paper_list
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM_PREAMBLE + "\n\n" + paper_markdown},
            {"role": "user", "content": user_message},
        ],
        tools=[_PLAN_TOOL],
        tool_choice="auto",
        extra_body={"thinking_mode": "thinking"},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    plans_data = json.loads(tool_call.function.arguments).get("plans", [])
    return [SummarizationPlan(**p) for p in plans_data]


def _summarize_abstract_only(
    meta: PaperMetadata,
    paper_markdown: str,
    client: OpenAI,
) -> str:
    prompt = (
        f"Write a concise summary (150-250 words) of the following related paper as it "
        f"relates to the target paper. Focus on the main contribution, methodology, and "
        f"key results.\n\n"
        f"Title: {meta['title']}\n"
        f"Authors: {', '.join(meta['authors'][:5])}\n"
        f"Abstract: {meta['abstract']}"
    )
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM_PREAMBLE + "\n\n" + paper_markdown},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def _summarize_full_text(
    meta: PaperMetadata,
    related_markdown: str,
    focus_areas: list[str],
    paper_markdown: str,
    client: OpenAI,
) -> str:
    focus_str = "\n".join(f"- {area}" for area in focus_areas) if focus_areas else "- overall contribution"
    prompt = (
        f"Write a detailed summary (400-600 words) of the following related paper as it "
        f"relates to the target paper. Focus especially on:\n{focus_str}\n\n"
        f"Title: {meta['title']}\n"
        f"Authors: {', '.join(meta['authors'][:5])}\n\n"
        f"--- FULL PAPER TEXT ---\n{related_markdown[:40000]}"
    )
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PREAMBLE + "\n\n" + paper_markdown},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def build_all_summaries(
    paper_markdown: str,
    plans: list[SummarizationPlan],
    metadata_map: dict[str, PaperMetadata],
    cache_dir: Path,
    client: OpenAI,
    ocr_engine=None,
) -> dict[str, PaperSummary]:
    """Generate summaries for all planned papers.

    For 'full_text' papers: downloads the PDF, runs OCR, then summarizes.
    For 'abstract_only' papers: summarizes from title + abstract only.
    Related paper markdowns are cached under cache_dir/<arxiv_id>.mmd.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, PaperSummary] = {}

    for plan in plans:
        arxiv_id = plan["arxiv_id"]
        meta = metadata_map.get(arxiv_id)
        if not meta:
            logger.warning("No metadata for %s, skipping.", arxiv_id)
            continue

        try:
            if plan["method"] == "full_text":
                paper_id = arxiv_id.replace(".", "_")
                mmd_path = cache_dir / f"{paper_id}.mmd"
                if mmd_path.exists():
                    related_md = mmd_path.read_text(encoding="utf-8")
                else:
                    pdf_path = download(arxiv_id, cache_dir)
                    related_md, mmd_path = convert_pdf_to_markdown(paper_id, pdf_path, cache_dir, ocr_engine=ocr_engine)
                    # mmd_path.write_text(related_md, encoding="utf-8")

                summary_text = _summarize_full_text(
                    meta, related_md, plan["focus_areas"], paper_markdown, client
                )
            else:
                summary_text = _summarize_abstract_only(meta, paper_markdown, client)

            summaries[arxiv_id] = PaperSummary(
                arxiv_id=arxiv_id,
                title=meta["title"],
                summary=summary_text,
            )
            print(f"  Summarized [{plan['method']}]: {meta['title'][:70]}")

        except Exception as exc:
            logger.warning("Failed to summarize %s: %s", arxiv_id, exc)

    return summaries
