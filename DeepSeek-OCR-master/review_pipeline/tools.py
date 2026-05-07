"""
Centralised tool definitions for all OpenAI-compatible function-calling API calls
used throughout the review pipeline.
"""
from __future__ import annotations

# ── Dimension registry (shared with scorer.py) ────────────────────────────────

DIMENSIONS = [
    "originality",
    "importance_of_research_question",
    "claims_well_supported",
    "soundness_of_experiments",
    "clarity_of_writing",
    "value_to_research_community",
    "contextualization_relative_to_prior_work",
]

_DIMENSION_DESCRIPTIONS = {
    "originality": (
        "How novel and creative is the work? Does it introduce genuinely new ideas, "
        "methods, or perspectives rather than incremental improvements?"
    ),
    "importance_of_research_question": (
        "How important is the problem being addressed? Would solving it have a "
        "significant impact on the field or downstream applications?"
    ),
    "claims_well_supported": (
        "Are the paper's claims backed by sufficient evidence, proofs, or experiments? "
        "Are limitations honestly acknowledged?"
    ),
    "soundness_of_experiments": (
        "Are experiments well-designed, reproducible, and statistically sound? "
        "Are baselines, ablations, and metrics appropriate?"
    ),
    "clarity_of_writing": (
        "Is the paper well-written and well-organised? "
        "Are figures, tables, and notation clear and easy to follow?"
    ),
    "value_to_research_community": (
        "What practical or theoretical value does this work provide? "
        "Will it enable or accelerate future research or real-world applications?"
    ),
    "contextualization_relative_to_prior_work": (
        "Does the paper accurately characterise and compare to relevant prior work? "
        "Is the related-work section comprehensive and fair?"
    ),
}

# ── Stage 3: arXiv query generation ──────────────────────────────────────────

QUERY_TOOL = {
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

# ── Stage 6: relevance scoring ────────────────────────────────────────────────

RELEVANCE_TOOL = {
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

# ── Stage 7: summarization planning ──────────────────────────────────────────

PLAN_TOOL = {
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

# ── Stage 9a: ICLR-style peer review ─────────────────────────────────────────

REVIEW_TOOL = {
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

# ── Stage 9b: dimensional quality scoring ────────────────────────────────────

SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_dimension_scores",
        "description": (
            "Submit quality scores for the paper across 7 evaluation dimensions. "
            "Each score is an integer from 1 (very poor) to 10 (excellent)."
        ),
        "parameters": {
            "type": "object",
            "required": DIMENSIONS + ["rationale"],
            "properties": {
                **{
                    dim: {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": _DIMENSION_DESCRIPTIONS[dim],
                    }
                    for dim in DIMENSIONS
                },
                "rationale": {
                    "type": "object",
                    "description": "One-sentence justification for each dimension score.",
                    "required": DIMENSIONS,
                    "properties": {dim: {"type": "string"} for dim in DIMENSIONS},
                },
            },
        },
    },
}

# ── Convenience list of all tools ────────────────────────────────────────────

ALL_TOOLS = [QUERY_TOOL, RELEVANCE_TOOL, PLAN_TOOL, REVIEW_TOOL, SCORE_TOOL]
