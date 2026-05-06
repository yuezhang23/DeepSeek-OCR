"""
Stage 9a: Generate a comprehensive ICLR-style peer review.
"""
from __future__ import annotations

import json
from typing import TypedDict

from openai import OpenAI

from review_pipeline import config
from review_pipeline.summarizer import PaperSummary

RATING_LABELS = {
    1: "Very Strong Reject: For instance, a paper with incorrect statements, improper (e.g., offensive) language, unaddressed ethical considerations, incorrect results and/or flawed methodology (e.g., training using a test set).",
    2: "Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.",
    3: "reject, not good enough",
    4: "Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.",
    5: "marginally below the acceptance threshold",
    6: "marginally above the acceptance threshold",
    7: "Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.",
    8: "accept, good paper",
    9: "Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI/ML and excellent impact on multiple areas of AI/ML, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.",
    10: "strong accept, should be highlighted at the conference"
}

CONFIDENCE_LABELS = {
    1: "Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.",
    2: "You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.",
    3: "You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.",
    4: "You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.",
    5: "You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully."
}


_SYSTEM_PREAMBLE = """\
You are a rigorous and fair academic peer reviewer for a top machine learning conference. \
You have deep expertise in the relevant research area. Your reviews are well-reasoned, \
specific, and constructive — pointing to concrete evidence from the paper rather than \
making vague claims. You assess novelty, technical soundness, experimental rigor, \
clarity, and broader impact.
"""

_ICLR_CRITERIA = """\
ICLR 2026 REVIEW CRITERIA:
- Novelty: Does the paper make a new contribution to the field?
- Technical soundness: Are the methods and proofs correct? Are experiments reproducible?
- Significance: Will this work influence future research or applications?
- Clarity: Is the paper well-written and easy to follow?
- Experimental evaluation: Are baselines fair and sufficient? Are ablations convincing?
- Related work: Is prior work appropriately cited and compared against?
"""

_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit the completed ICLR peer review.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "3-5 sentence summary of the paper's main contributions and approach.",
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific strengths (3-6 items).",
                    "minItems": 2,
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific weaknesses or concerns (3-6 items).",
                    "minItems": 2,
                },
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific questions for the authors to address (2-5 items).",
                    "minItems": 1,
                },
                "limitations_and_societal_impact": {
                    "type": "string",
                    "description": "Assessment of limitations acknowledged by the authors and potential societal impact.",
                },
                "rating": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Overall rating: 1=Strong Reject, 5=Borderline Accept, 8=Strong Accept, 10=Outstanding.",
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Reviewer confidence: 1=educated guess, 5=absolutely certain.",
                },
                "ethics_flag": {
                    "type": "boolean",
                    "description": "True if the paper raises significant ethical concerns requiring committee review.",
                },
            },
            "required": [
                "summary", "strengths", "weaknesses", "questions",
                "limitations_and_societal_impact", "rating", "confidence", "ethics_flag",
            ],
        },
    },
}


class ILCRReview(TypedDict):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    questions: list[str]
    limitations_and_societal_impact: str
    rating: int
    confidence: int
    ethics_flag: bool


def generate_review(
    paper_markdown: str,
    summaries: dict[str, PaperSummary],
    client: OpenAI,
    venue: str = "ICLR",
    year: int = 2026,
) -> tuple[ILCRReview, str]:
    """Generate a structured ICLR review and return (review_dict, markdown_string)."""
    related_work_context = _build_related_work_context(summaries)

    system_content = _SYSTEM_PREAMBLE + "\n\n" + _ICLR_CRITERIA + "\n\n" + paper_markdown
    if related_work_context:
        system_content += "\n\n" + related_work_context

    user_message = (
        f"Please write a complete {venue} {year} peer review for the paper above. "
        f"Ground your assessment in the related work summaries provided. "
        f"Be specific and cite concrete evidence from the paper. "
        f"Use the submit_review tool."
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ],
        tools=[_REVIEW_TOOL],
        tool_choice="auto",
        extra_body={"thinking_mode": "thinking"},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    review_dict: ILCRReview = json.loads(tool_call.function.arguments)

    if not review_dict:
        raise ValueError("Model did not return a tool_use block for review generation.")

    return review_dict, format_review_markdown(review_dict, venue=venue, year=year)


def _build_related_work_context(summaries: dict[str, PaperSummary]) -> str:
    if not summaries:
        return ""
    lines = ["=== RELATED WORK SUMMARIES ===\n"]
    for i, (arxiv_id, s) in enumerate(summaries.items(), 1):
        lines.append(f"[{i}] {s['title']} (arXiv:{arxiv_id})\n{s['summary']}\n")
    return "\n".join(lines)


def format_review_markdown(
    review: ILCRReview,
    venue: str = "ICLR",
    year: int = 2026,
) -> str:
    rating = review["rating"]
    confidence = review["confidence"]
    ethics = "Yes — flagged for ethics committee review" if review["ethics_flag"] else "No"

    strengths = "\n".join(f"- {s}" for s in review["strengths"])
    weaknesses = "\n".join(f"- {w}" for w in review["weaknesses"])
    questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(review["questions"]))

    return f"""\
# {venue} {year} Paper Review

## Summary
{review['summary']}

## Strengths
{strengths}

## Weaknesses
{weaknesses}

## Questions for Authors
{questions}

## Limitations and Societal Impact
{review['limitations_and_societal_impact']}

## Rating
**{rating}/10 — {RATING_LABELS.get(rating, '')}**

## Confidence
**{confidence}/5** — {CONFIDENCE_LABELS.get(confidence, '')}

## Ethics Review Flag
{ethics}
"""
