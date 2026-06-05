"""
Alignment analysis — map real ICLR reviewer comments (``official_values`` in
``metadata_300.json``) onto the 7 quality dimensions used by ``scorer.py``.

For each paper, the official reviews are passed to a DeepSeek LLM that judges
which dimension(s) each piece of reviewer feedback corresponds to, then emits
an integer 1-10 score per dimension plus a rationale — matching the output
shape produced by ``scorer.score_paper`` (``DimensionScores``).
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from review_pipeline import config
from review_pipeline.clients import deepseek_chat, get_tool_call
from review_pipeline.scorer import DIMENSION_LABELS, DimensionScores
from review_pipeline.tools import DIMENSIONS, SCORE_TOOL as _SCORE_TOOL

logger = logging.getLogger(__name__)

_METADATA_PATH = Path(__file__).parent / "metadata_300.json"
_REVIEW_CHAR_LIMIT = 60_000  # ~15K tokens; keeps total context within 128K window


# ─── Prompts ──────────────────────────────────────────────────────────────────
_ALIGNMENT_PREAMBLE = """\
You are an expert academic-review analyst. You will be shown the *real* peer
reviews submitted by ICLR reviewers for a single paper. Your job is to read
each review carefully and decide which of the 7 evaluation dimensions below
each piece of reviewer feedback is talking about. A single review usually
touches several dimensions — map every concrete comment, strength, or
weakness onto the dimension(s) it most directly addresses.

Then, using *only* the evidence in those reviews (do NOT invent details that
the reviewers did not raise), call the `submit_dimension_scores` tool with one
nested object per dimension.

For each dimension you will emit a nested object with two fields, in this exact order:
  1. `rationale` — a 2–4 sentence evidence-grounded justification citing the
     concrete reviewer comments (quotes, paraphrases, or specific
     strengths/weaknesses raised) that drove your judgement.
  2. `score` — an integer from 1 to 10 that follows from the rationale you just wrote.

The schema enforces this ordering: `rationale` is declared before `score` in every dimension object. Treat the score as a *conclusion* of the rationale, never a number picked first and justified after.

Scoring guide (same scale as the model scorer, so results are comparable):
  9–10  Reviewers are overwhelmingly positive about this dimension.
  7–8   Reviewers praise this dimension with only minor reservations.
  5–6   Reviewers are mixed or note non-trivial concerns.
  3–4   Reviewers flag substantive flaws on this dimension.
  1–2   Reviewers consider this dimension severely deficient.
If the reviewers say essentially nothing about a dimension, score it 5
(neutral) and state that explicitly in the rationale.
"""


def _dimension_block() -> str:
    return "Dimensions:\n" + "\n".join(
        f"  - {dim} ({DIMENSION_LABELS[dim]})" for dim in DIMENSIONS
    )


# ─── Review formatting ────────────────────────────────────────────────────────
def _format_reviews(official_values: Iterable[dict]) -> str:
    """Render the list of official reviews into a compact text block."""
    blocks: list[str] = []
    idx = 0
    for review in official_values:
        if not isinstance(review, dict) or not review:
            continue
        idx += 1
        parts = [f"=== Reviewer {idx} (review_id={review.get('review_id', 'n/a')}) ==="]
        for key in ("rating", "confidence", "soundness", "presentation", "contribution"):
            if review.get(key) is not None:
                parts.append(f"{key}: {review[key]}")
        for key in ("summary", "strengths", "weaknesses", "questions"):
            val = review.get(key)
            if val:
                parts.append(f"{key.upper()}:\n{val}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


# ─── Public API ───────────────────────────────────────────────────────────────
def align_reviews_to_dimensions(
    official_values: list[dict],
    client: OpenAI,
    paper_title: str | None = None,
) -> DimensionScores:
    """Map the real reviewer comments for a single paper onto the 7 dimensions.

    Returns a ``DimensionScores`` dict — same shape as ``scorer.score_paper``.
    """
    reviews_text = _format_reviews(official_values)
    if not reviews_text:
        raise ValueError("No non-empty reviews in official_values.")

    if len(reviews_text) > _REVIEW_CHAR_LIMIT:
        logger.warning(
            "Reviews truncated from %d to %d chars for alignment.",
            len(reviews_text), _REVIEW_CHAR_LIMIT,
        )
        reviews_text = reviews_text[:_REVIEW_CHAR_LIMIT]

    system_content = (
        _ALIGNMENT_PREAMBLE
        + "\n\n" + _dimension_block()
        + (f"\n\nPaper title: {paper_title}" if paper_title else "")
        + "\n\nOfficial ICLR reviews:\n\n" + reviews_text
    )

    user_message = (
        "Produce the dimension scores by calling `submit_dimension_scores`. "
        "Write each dimension's `rationale` before its `score`."
    )

    response = deepseek_chat(
        client,
        system=system_content,
        user=user_message,
        max_tokens=8192,
        tools=[_SCORE_TOOL],
        thinking=True,
    )
    tool_call = get_tool_call(response)
    return json.loads(tool_call.function.arguments)


# ─── Score-markdown → JSON converter ──────────────────────────────────────────
# Matches a row from `scorer.format_scores_markdown`:
#   | Originality | 6/10 `██████░░░░` | rationale text… |
_SCORE_ROW_RE = re.compile(
    r"\|\s*([^|]+?)\s*\|\s*(\d+)\s*/\s*10[^|]*\|\s*(.+?)\s*\|"
)
_LABEL_TO_KEY = {label: key for key, label in DIMENSION_LABELS.items()}


def parse_score_md(md_path: Path) -> DimensionScores | None:
    """Parse a stage 9-b dimension-score markdown file into a ``DimensionScores`` dict.

    Returns None if any of the 7 dimensions cannot be extracted.
    """
    text = md_path.read_text(encoding="utf-8")
    scores: dict = {}
    rationale: dict[str, str] = {}
    for label, score, note in _SCORE_ROW_RE.findall(text):
        key = _LABEL_TO_KEY.get(label.strip())
        if key is None:
            continue
        scores[key] = int(score)
        rationale[key] = note.strip()
    if any(dim not in scores for dim in DIMENSIONS):
        return None
    scores["rationale"] = rationale
    return scores  # type: ignore[return-value]


def _flush_results(out_path: Path, results: dict[str, DimensionScores]) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def align_all_papers(
    client: OpenAI,
    metadata_path: Path = _METADATA_PATH,
    limit: int | None = None,
    out_path: Path | None = None,
    resume: bool = True,
    batch_size: int = 8,
) -> dict[str, DimensionScores]:
    """Run alignment on every paper in ``metadata_300.json``.

    Papers are processed concurrently in batches of ``batch_size`` workers
    (LLM calls are I/O-bound). When ``out_path`` is given, the JSON file is
    atomically rewritten after each successful paper so partial progress is
    durable; if ``resume`` is True and the file already exists, previously-
    computed results are loaded and those papers are skipped.
    """
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = meta if limit is None else meta[:limit]

    results: dict[str, DimensionScores] = {}
    if out_path is not None and resume and out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            logger.info("Resuming from %s (%d papers already done).", out_path, len(results))
        except json.JSONDecodeError:
            logger.warning("Could not parse existing %s; starting fresh.", out_path)
            results = {}

    pending: list[tuple[str, list[dict], str | None]] = []
    for entry in entries:
        pdf_stem = Path(entry["pdf_name"]).stem
        if pdf_stem in results:
            continue
        reviews = entry.get("official_values") or []
        if not any(isinstance(r, dict) and r for r in reviews):
            logger.warning("Skipping %s: no official reviews.", pdf_stem)
            continue
        pending.append((pdf_stem, reviews, entry.get("paper_title")))

    if not pending:
        return results

    lock = threading.Lock()

    def _worker(stem: str, reviews: list[dict], title: str | None):
        return stem, align_reviews_to_dimensions(reviews, client, paper_title=title)

    with ThreadPoolExecutor(max_workers=max(1, batch_size)) as pool:
        futures = {pool.submit(_worker, stem, reviews, title): stem
                   for stem, reviews, title in pending}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc=f"Aligning reviews (batch={batch_size})", unit="paper")
        for fut in pbar:
            stem = futures[fut]
            pbar.set_postfix_str(stem)
            try:
                _, scores = fut.result()
            except Exception as exc:
                logger.error("Alignment failed for %s: %s", stem, exc)
                continue
            with lock:
                results[stem] = scores
                if out_path is not None:
                    _flush_results(out_path, results)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run alignment analysis on all papers.")
    parser.add_argument("--metadata", type=Path, default=_METADATA_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Max number of papers to process.")
    parser.add_argument("--batch-size", type=int, default=8, help="Concurrent LLM requests.")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "out", help="Directory to write results.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = args.output_dir
    # out_path = out_dir / "map2metrics.json"

    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set (check your .env / environment).")


    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    all_results = align_all_papers(
        client,
        metadata_path=args.metadata,
        limit=args.limit,
        out_path=out_dir,
        batch_size=args.batch_size,
    )
    print(f"Wrote {len(all_results)} results to {out_dir}")