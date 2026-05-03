"""
Stage 9-b — Dimensional quality scoring with linear-regression final score.

Scores the paper on 7 dimensions (each 1–10) and maps them to a final score
via a linear regression model.  The default weight vector is pre-set to
reasonable values; call DimensionalScoringModel.fit() to retrain once you have
labelled examples.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
from openai import OpenAI

from review_pipeline import config

logger = logging.getLogger(__name__)

# ─── Dimension registry ───────────────────────────────────────────────────────

DIMENSIONS = [
    "originality",
    "importance_of_research_question",
    "claims_well_supported",
    "soundness_of_experiments",
    "clarity_of_writing",
    "value_to_research_community",
    "contextualization_relative_to_prior_work",
]

DIMENSION_LABELS = {
    "originality": "Originality",
    "importance_of_research_question": "Importance of Research Question",
    "claims_well_supported": "Claims Well Supported",
    "soundness_of_experiments": "Soundness of Experiments",
    "clarity_of_writing": "Clarity of Writing",
    "value_to_research_community": "Value to Research Community",
    "contextualization_relative_to_prior_work": "Contextualization Relative to Prior Work",
}

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

# ─── Default linear-regression weights ───────────────────────────────────────
# Weights sum to 1.0 and intercept = 0 so that a score vector in [1,10]^7
# maps to a final score also in [1,10].  These reflect typical ICLR priorities
# (originality and soundness weighted highest); retrain via fit() for custom use.
_DEFAULT_WEIGHTS = np.array([0.18, 0.16, 0.15, 0.16, 0.12, 0.14, 0.09])
_DEFAULT_INTERCEPT = 0.0


# ─── TypedDicts ───────────────────────────────────────────────────────────────

class DimensionScores(TypedDict):
    originality: int
    importance_of_research_question: int
    claims_well_supported: int
    soundness_of_experiments: int
    clarity_of_writing: int
    value_to_research_community: int
    contextualization_relative_to_prior_work: int
    rationale: dict[str, str]


# ─── Tool definition ─────────────────────────────────────────────────────────

_SCORE_TOOL = {
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

# ─── System prompts ───────────────────────────────────────────────────────────

_SYSTEM_PREAMBLE = """\
You are an expert academic reviewer with deep knowledge of machine learning, \
computer science, and related fields. You have reviewed hundreds of papers for \
top venues including NeurIPS, ICML, ICLR, ACL, and CVPR.

Your task is to score the provided research paper on 7 evaluation dimensions, \
each on a 1–10 integer scale. Base your scores on the full paper text and the \
provided summaries of related work.

Scoring guide:
  1–2   Very poor / major deficiency
  3–4   Below average / significant weaknesses
  5–6   Average / acceptable with weaknesses
  7–8   Good / above average
  9–10  Excellent / among the best in the field
"""

_SCORING_CRITERIA = """\
Dimension definitions
─────────────────────
1. Originality (1–10)
   Novelty and creativity. Does the work go meaningfully beyond prior art?

2. Importance of Research Question (1–10)
   How consequential is the problem? Would solving it move the field forward?

3. Claims Well Supported (1–10)
   Evidence quality. Are claims backed by proofs, experiments, or analysis?
   Are limitations honestly acknowledged?

4. Soundness of Experiments (1–10)
   Rigour and reproducibility. Appropriate baselines, ablations, and statistics?

5. Clarity of Writing (1–10)
   Readability and organisation. Clear figures, tables, notation, and exposition?

6. Value to Research Community (1–10)
   Practical or theoretical contribution. Will it enable follow-on work?

7. Contextualization Relative to Prior Work (1–10)
   Accurate and comprehensive treatment of related work. Fair comparisons?
"""


# ─── Linear regression model ──────────────────────────────────────────────────

class DimensionalScoringModel:
    """
    Linear model: final_score = weights · dimension_scores + intercept.

    Initialised with pre-set default weights.  Call fit() to retrain from
    labelled data; call save() / load() for persistence across runs.
    """

    _MODEL_FILE = "dimensional_lr_model.json"

    def __init__(self, model_dir: Path | None = None):
        self.weights: np.ndarray = _DEFAULT_WEIGHTS.copy()
        self.intercept: float = _DEFAULT_INTERCEPT
        if model_dir:
            self._try_load(model_dir)

    # ── persistence ──────────────────────────────────────────────────────────

    def _try_load(self, model_dir: Path) -> None:
        path = model_dir / self._MODEL_FILE
        if path.exists():
            data = json.loads(path.read_text())
            self.weights = np.array(data["weights"])
            self.intercept = float(data["intercept"])
            logger.info("Loaded dimensional scoring model from %s", path)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / self._MODEL_FILE
        path.write_text(
            json.dumps({"weights": self.weights.tolist(), "intercept": self.intercept},
                       indent=2)
        )
        logger.info("Saved dimensional scoring model to %s", path)

    # ── prediction ───────────────────────────────────────────────────────────

    def predict(self, scores: DimensionScores) -> float:
        x = np.array([scores[d] for d in DIMENSIONS], dtype=float)
        raw = float(np.dot(self.weights, x) + self.intercept)
        return round(float(np.clip(raw, 1.0, 10.0)), 2)

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, X: list[list[int]], y: list[float]) -> dict:
        """
        Retrain the model from labelled examples.

        Args:
            X: List of 7-element score vectors (one per paper).
            y: Corresponding ground-truth final scores (1–10).

        Returns:
            Dict with r2, weights, and intercept.
        """
        from sklearn.linear_model import LinearRegression  # soft dependency

        arr_X = np.array(X, dtype=float)
        arr_y = np.array(y, dtype=float)
        lr = LinearRegression(fit_intercept=True)
        lr.fit(arr_X, arr_y)
        self.weights = lr.coef_
        self.intercept = float(lr.intercept_)
        r2 = float(lr.score(arr_X, arr_y))
        logger.info(
            "Model refitted — R²=%.4f  weights=%s  intercept=%.4f",
            r2, self.weights.tolist(), self.intercept,
        )
        return {"r2": r2, "weights": self.weights.tolist(), "intercept": self.intercept}


# Module-level default model instance (no persistence, default weights)
_default_model = DimensionalScoringModel()


# ─── Public API ───────────────────────────────────────────────────────────────

def score_paper(
    paper_md: str,
    summaries: dict,
    client: OpenAI,
    model: DimensionalScoringModel | None = None,
) -> tuple[DimensionScores, float]:
    """Score the paper on 7 dimensions, then apply the linear model.

    Returns (scores_dict, final_score) where final_score is clipped to [1, 10].
    """
    related_ctx = _build_related_work_context(summaries)
    system_content = _SYSTEM_PREAMBLE + "\n\n" + _SCORING_CRITERIA + "\n\nPaper to evaluate:\n\n" + paper_md
    if related_ctx:
        system_content += "\n\n" + related_ctx

    user_message = (
        "Please evaluate this paper carefully on all 7 dimensions. "
        "Call submit_dimension_scores with integer scores (1–10) and a "
        "one-sentence rationale for each dimension."
    )

    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ],
        tools=[_SCORE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_dimension_scores"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call is None:
        raise ValueError("Model did not return a tool_use block for dimension scoring.")

    scores: DimensionScores = json.loads(tool_call.function.arguments)
    final = (model or _default_model).predict(scores)
    return scores, final


def _build_related_work_context(summaries: dict) -> str:
    if not summaries:
        return ""
    lines = ["=== RELATED WORK SUMMARIES ===\n"]
    for i, (arxiv_id, info) in enumerate(summaries.items(), 1):
        title = info.get("title", arxiv_id) if isinstance(info, dict) else getattr(info, "title", arxiv_id)
        summary = info.get("summary", "") if isinstance(info, dict) else getattr(info, "summary", "")
        lines.append(f"[{i}] {title} (arXiv:{arxiv_id})\n{summary}\n")
    return "\n".join(lines)


def format_scores_markdown(
    scores: DimensionScores,
    final_score: float,
    model: DimensionalScoringModel | None = None,
    venue: str = "ICLR",
    year: int = 2026,
) -> str:
    """Render dimension scores and final score as a markdown report."""
    m = model or _default_model
    rationale: dict[str, str] = scores.get("rationale", {})

    rows = []
    for dim in DIMENSIONS:
        label = DIMENSION_LABELS[dim]
        s = scores[dim]
        bar = "█" * s + "░" * (10 - s)
        note = rationale.get(dim, "")
        rows.append(f"| {label} | {s}/10 `{bar}` | {note} |")

    weight_rows = []
    for dim, w in zip(DIMENSIONS, m.weights):
        weight_rows.append(f"| {DIMENSION_LABELS[dim]} | {w:.3f} |")

    return f"""\
# {venue} {year} — Dimensional Quality Assessment

## Dimension Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
{chr(10).join(rows)}

## Final Score

**{final_score:.2f} / 10**

> Computed as a weighted linear combination of the 7 dimension scores.

### Linear Model Weights

| Dimension | Weight |
|-----------|--------|
{chr(10).join(weight_rows)}
| **Intercept** | {m.intercept:.3f} |
"""
