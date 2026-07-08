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
        "Originality & Technical Novelty — How novel and creative is the work? Does it "
        "introduce genuinely new ideas, methods, or perspectives rather than incremental "
        "improvements? Assess via:\n"
        "  • The \"Gap\" Analysis: Does the work introduce a fundamentally new mechanism "
        "(e.g., a novel objective function, architecture, or optimization strategy), or is "
        "it a \"delta\" improvement?\n"
        "  • Creative Synthesis: Does it bridge disparate fields (e.g., Information Theory "
        "and Diffusion Models) in a non-obvious way?\n"
        "  • Path-Clearing: Does it challenge established \"folk wisdom\" in the ML "
        "community with a fresh perspective?"
    ),
    "importance_of_research_question": (
        "Significance & Research Impact — How important is the problem being addressed? "
        "Would solving it have a significant impact on the field or downstream "
        "applications? Assess via:\n"
        "  • Problem Criticality: Does the research address a major bottleneck (e.g., "
        "inference latency, data efficiency, or alignment fragility)?\n"
        "  • Breadth of Utility: Are the findings specific to a niche task, or are they "
        "generalizable across different modalities (Vision, Language, Robotics)?\n"
        "  • Field Evolution: If the claims are true, would this work change how other "
        "researchers approach their projects next year?"
    ),
    "claims_well_supported": (
        "Empirical & Theoretical Grounding — Are the paper's claims backed by sufficient "
        "evidence, proofs, or experiments? Are limitations honestly acknowledged? "
        "Assess via:\n"
        "  • Claim-Evidence Alignment: Are the central claims directly supported by the "
        "data presented? (e.g., if a paper claims \"efficiency,\" is there a "
        "Flops-vs-Accuracy curve?)\n"
        "  • Theoretical Rigor: For theoretical papers, are the assumptions realistic? For "
        "empirical papers, is the intuition backed by formal analysis?\n"
        "  • Transparency: Are \"failure cases\" discussed with the same level of detail "
        "as the successes?"
    ),
    "soundness_of_experiments": (
        "Soundness & Reproducibility — Are experiments well-designed, reproducible, and "
        "statistically sound? Are baselines, ablations, and metrics appropriate? "
        "Assess via:\n"
        "  • Baseline Integrity: Are the baselines strong and properly tuned? (Avoidance "
        "of \"straw-man\" comparisons.)\n"
        "  • Statistical Significance: Are results reported with error bars, multiple "
        "seeds, and sensitivity analyses?\n"
        "  • Scaling Consistency: Does the method hold up as the model size or data volume "
        "increases, or does the advantage vanish at scale?"
    ),
    "clarity_of_writing": (
        "Exposition & Clarity — Is the paper well-written and well-organised? Are figures, "
        "tables, and notation clear and easy to follow? Assess via:\n"
        "  • Structure & Flow: Is the \"story\" of the paper logical? Can a reader grasp "
        "the main contribution just by looking at the Abstract, Figure 1, and the "
        "Conclusion?\n"
        "  • Formalism: Is the mathematical notation standard, precise, and consistent?\n"
        "  • Visual Communication: Are charts and tables designed to be informative at a "
        "glance, with clear axes, labels, and captions?"
    ),
    "value_to_research_community": (
        "Collaborative Value & Open Science — What practical or theoretical value does "
        "this work provide? Will it enable or accelerate future research or real-world "
        "applications? Assess via:\n"
        "  • Resource Contribution: Does the work provide a new, high-quality dataset, a "
        "refined benchmark, or a robust codebase?\n"
        "  • Heuristic Value: Does the paper provide \"lessons learned\" or negative "
        "results that save the community from future dead-ends?\n"
        "  • Ethics & Safety: Does the paper proactively address potential dual-use "
        "concerns or biases in its findings?"
    ),
    "contextualization_relative_to_prior_work": (
        "Contextualization & Literature Mastery — Does the paper accurately characterise "
        "and compare to relevant prior work? Is the related-work section comprehensive and "
        "fair? Assess via:\n"
        "  • Historical Accuracy: Does the paper correctly attribute ideas to their "
        "original sources, moving beyond just citing the most recent \"famous\" paper?\n"
        "  • Critical Comparison: Does the related work section explain how this work "
        "differs conceptually, rather than just listing 20 papers?\n"
        "  • Fairness: Does the paper acknowledge contemporaneous work and provide a "
        "neutral, objective comparison?"
    ),
}

# ── Shared 1–10 score-band interpretation ────────────────────────────────────
# Single source of truth for what each band of the integer 1–10 dimension score MEANS,
# phrased as paper quality on the dimension. It deliberately uses the SAME numeric bands as
# review_scoring.py's reviewer-sentiment guide so model scores and human-aligned scores are
# directly comparable. Injected into both scorer backends' system prompts.
SCORE_SCALE_GUIDE = (
    "Score interpretation (integer 1–10; pick the band the evidence supports for THIS "
    "dimension only):\n"
    "  9–10  Exceptional — a clear, standout strength on this dimension with essentially no "
    "weaknesses.\n"
    "  7–8   Strong — solid on this dimension with only minor reservations.\n"
    "  5–6   Borderline — mixed; non-trivial concerns roughly balance the merits (use 5 when "
    "there is too little evidence to judge, and say so).\n"
    "  3–4   Weak — substantive flaws on this dimension that outweigh the merits.\n"
    "  1–2   Severely deficient — this dimension is a critical failure of the paper."
)

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
# Per-dimension schema is a NESTED object whose `rationale` field is declared
# (and emitted) BEFORE `score`. This enforces "reason before score" at the
# decoder level — the integer cannot be produced until the rationale tokens
# have been generated. Without this, the model commits to a number first and
# rationalises after, which defeats the design.
SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_dimension_scores",
        "description": (
            "Submit quality scores for the paper across 7 evaluation dimensions. "
            "For every dimension, emit the `rationale` field BEFORE the `score` "
            "field — the integer score must be a conclusion of the rationale you "
            "just wrote, not a number picked first and justified after. "
            "Scores are integers from 1 (very poor) to 10 (excellent)."
        ),
        "parameters": {
            "type": "object",
            "required": list(DIMENSIONS),
            "properties": {
                dim: {
                    "type": "object",
                    "description": _DIMENSION_DESCRIPTIONS[dim],
                    "required": ["rationale", "score"],
                    "properties": {
                        "rationale": {
                            "type": "string",
                            "description": (
                                "2–4 sentence evidence-grounded justification, "
                                "written BEFORE choosing this dimension's score. "
                                "Cite specific sections, figures, claims, or "
                                "baselines from the paper."
                            ),
                        },
                        "score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": (
                                "Integer 1–10 that follows from the rationale "
                                "above. Do not anchor on a number — let the "
                                "rationale you just wrote determine it."
                            ),
                        },
                    },
                }
                for dim in DIMENSIONS
            },
        },
    },
}

# ── Stage 9b (DeepSeek/langgraph variant): single-dimension scoring ──────────
# scorer_deepseek.py scores ONE dimension per LLM call (one langgraph node per
# dimension), so it needs a single-dimension function whose `rationale` is declared
# BEFORE `score` — same "reason before score" decoder trick as SCORE_TOOL, but for
# one dimension at a time. The dimension being scored is fixed by the system prompt.
DIMENSION_SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_score",
        "description": (
            "Record your FINAL rationale and integer score (1-10) for the ONE dimension "
            "named in the system prompt. Emit the `rationale` field BEFORE the `score` "
            "field — the integer must be a conclusion of the rationale you just wrote, not "
            "a number picked first and justified after. Call this exactly once."
        ),
        "parameters": {
            "type": "object",
            "required": ["rationale", "score"],
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": (
                        "2–4 sentence evidence-grounded justification, written BEFORE "
                        "choosing the score. Cite specific sections, figures, claims, or "
                        "baselines from the paper."
                    ),
                },
                "score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": (
                        "Integer 1–10 that follows from the rationale above. Do not anchor "
                        "on a number — let the rationale you just wrote determine it."
                    ),
                },
            },
        },
    },
}

# ── Stage 9b helper: per-section reading strategy ────────────────────────────
# Column headers of the section × dimension matrix in selection_strategy.md, in the SAME
# order as DIMENSIONS. This is the single source of truth that ties the human-readable
# table columns to the canonical dimension keys (used to parse AND to append rows).
SELECTION_DIMENSION_COLUMNS = [
    "Originality",
    "Research Importance",
    "Claims Supported",
    "Experiment Soundness",
    "Clarity",
    "Value to Community",
    "Prior Work Context",
]
assert len(SELECTION_DIMENSION_COLUMNS) == len(DIMENSIONS)

# Function schema for scoring how important an UNKNOWN paper section (one not yet in
# selection_strategy.md) is to each dimension. The decoder must return, per section, one
# integer per dimension following the rule below. scorer.section_control calls this once for
# all unknown sections, then appends the results back into the table.
_IMPORTANCE_RULE = (
    "0 = the section's RAW full text is required (its detail is critical for this dimension); "
    "1 = a SUMMARY preserves the key information needed for this dimension; "
    "2 = OMIT (the section is of little significance to this dimension)."
)

SECTION_IMPORTANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_section_importance",
        "description": (
            "Submit, for each given paper section, how its content should be presented to a "
            "reviewer evaluating each of the 7 quality dimensions. For every dimension emit "
            f"exactly one integer: {_IMPORTANCE_RULE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "selections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "The section title exactly as provided.",
                            },
                            **{
                                dim: {
                                    "type": "integer",
                                    "enum": [0, 1, 2],
                                    "description": f"{_DIMENSION_DESCRIPTIONS[dim].splitlines()[0]} — {_IMPORTANCE_RULE}",
                                }
                                for dim in DIMENSIONS
                            },
                        },
                        "required": ["section", *DIMENSIONS],
                    },
                },
            },
            "required": ["selections"],
        },
    },
}

# ── Convenience list of all tools ────────────────────────────────────────────
ALL_TOOLS = [QUERY_TOOL, RELEVANCE_TOOL, PLAN_TOOL, REVIEW_TOOL, SCORE_TOOL]