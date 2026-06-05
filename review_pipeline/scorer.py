"""
Stage 9-b — Dimensional quality scoring.

Scores the paper on 7 dimensions (each 1–10) using DeepSeek.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import TypedDict

from openai import OpenAI

# Allow running directly (`python scorer.py …` or `python -m review_pipeline.scorer …`)
# by ensuring the package root is importable before the review_pipeline imports below.
sys.path.insert(0, str(Path(__file__).parent.parent))

from review_pipeline import config
from review_pipeline.clients import deepseek_chat, get_tool_call
from review_pipeline.tools import DIMENSIONS, SCORE_TOOL as _SCORE_TOOL

logger = logging.getLogger(__name__)

# Process-wide lock so concurrent batch workers serialize their JSON appends.
_SCORES_JSON_LOCK = threading.Lock()

# ─── Dimension registry ───────────────────────────────────────────────────────
DIMENSION_LABELS = {
    "originality": "Originality",
    "importance_of_research_question": "Importance of Research Question",
    "claims_well_supported": "Claims Well Supported",
    "soundness_of_experiments": "Soundness of Experiments",
    "clarity_of_writing": "Clarity of Writing",
    "value_to_research_community": "Value to Research Community",
    "contextualization_relative_to_prior_work": "Contextualization Relative to Prior Work",
}

# ─── TypedDicts ───────────────────────────────────────────────────────────────
class DimensionEntry(TypedDict):
    rationale: str
    score: int


class DimensionScores(TypedDict):
    originality: DimensionEntry
    importance_of_research_question: DimensionEntry
    claims_well_supported: DimensionEntry
    soundness_of_experiments: DimensionEntry
    clarity_of_writing: DimensionEntry
    value_to_research_community: DimensionEntry
    contextualization_relative_to_prior_work: DimensionEntry

# ─── System prompts ───────────────────────────────────────────────────────────
# Structure mirrors alignment_analysis.py: one consolidated preamble that combines role + task + scoring guide, then a dimensions block, then the paper content. Keeps the rationale-before-score instruction front and centre.
_SYSTEM_PREAMBLE = """\
You are an expert academic reviewer with deep knowledge of machine learning, computer science, and related fields. You have reviewed hundreds of papers for top venues including NeurIPS, ICML, ICLR, ACL, and CVPR.

Read the paper below and rate it on the 7 evaluation dimensions listed further down. For each dimension you will emit a nested object with two fields, in this exact order:
  1. `rationale` — a 2–4 sentence evidence-grounded justification citing concrete elements of the paper (specific sections, figures, claims, baselines).
  2. `score` — an integer from 1 to 10 that follows from the rationale you just wrote.

The schema enforces this ordering: `rationale` is declared before `score` in every dimension object. Treat the score as a *conclusion* of the rationale, never a number picked first and justified after. If your reasoning shifts mid-rationale, update the score; never anchor on the number.

Scoring guide:
  9–10  Paradigm-shifting; flawless execution; likely an Oral.
  7–8   Strong contribution; well-supported; clear "Accept".
  5–6   Interesting but has flaws or limited novelty; "Borderline".
  3–4   Significant technical errors, lack of clarity, or negligible novelty.
  1–2   Factually incorrect, plagiarized, or completely out of scope.
"""


def _dimension_block() -> str:
    """Compact listing of the 7 dimension names — used as a quick reference
    above the detailed criteria (matches the `_dimension_block()` pattern in
    alignment_analysis.py)."""
    return "Dimensions to score:\n" + "\n".join(
        f"  - {dim} ({DIMENSION_LABELS[dim]})" for dim in DIMENSIONS
    )


# ─── Public API ───────────────────────────────────────────────────────────────
_PAPER_CHAR_LIMIT = 100_000  # ~15K tokens; keeps total context well within 128K window


def score_paper(
    paper_md: str,
    summaries: dict,
    client: OpenAI,
) -> DimensionScores:
    """Score the paper on 7 dimensions. Returns DimensionScores dict."""
    truncated = paper_md[:_PAPER_CHAR_LIMIT]
    if len(paper_md) > _PAPER_CHAR_LIMIT:
        logger.warning("Paper truncated from %d to %d chars for scoring.", len(paper_md), _PAPER_CHAR_LIMIT)

    related_ctx = _build_related_work_context(summaries)
    parts = [
        _SYSTEM_PREAMBLE,
        _dimension_block(),
        "Paper to evaluate:\n\n" + truncated,
        related_ctx or None,
    ]
    system_content = "\n\n".join(p for p in parts if p)

    logger.info("Scorer system prompt: %d chars", len(system_content))

    user_message = (
        "Evaluate the paper on all 7 dimensions. For each dimension's nested "
        "object, write `rationale` first (evidence-grounded, citing the paper), "
        "then choose the `score` (1–10 integer) that follows from it."
    )

    response = deepseek_chat(
        client,
        system=system_content,
        user=user_message,
        max_tokens=8192,
        tools=[_SCORE_TOOL],
        # tool_choice={"type": "function", "function": {"name": "submit_dimension_scores"}},
        thinking=True,
    )
    tool_call = get_tool_call(response)
    return json.loads(tool_call.function.arguments)


def _build_related_work_context(summaries: dict) -> str:
    if not summaries:
        return ""
    lines = ["=== RELATED WORK SUMMARIES ===\n"]
    for i, (arxiv_id, info) in enumerate(summaries.items(), 1):
        title = info.get("title", arxiv_id) if isinstance(info, dict) else getattr(info, "title", arxiv_id)
        summary = info.get("summary", "") if isinstance(info, dict) else getattr(info, "summary", "")
        lines.append(f"[{i}] {title} (arXiv:{arxiv_id})\n{summary}\n")
    return "\n".join(lines)


# ─── Batch API ────────────────────────────────────────────────────────────────
def _append_scores_json(json_path: Path, paper_id: str, scores: DimensionScores) -> None:
    """Atomically merge ``{paper_id: scores}`` into the JSON file at json_path.

    Thread-safe: holds a process-wide lock around read-modify-write so concurrent
    batch workers don't clobber each other. Mirrors ``main._append_scores_json``.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with _SCORES_JSON_LOCK:
        existing: dict = {}
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Could not parse %s; starting fresh.", json_path)
        existing[paper_id] = scores
        tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(json_path)


def score_paper_batch(
    paper_mds: dict[str, str],
    client: OpenAI,
    summaries: dict[str, dict] | None = None,
    output_path: Path | str = Path("out/ai_results.json"),
    max_workers: int | None = None,
    resume: bool = True,
) -> dict[str, DimensionScores]:
    """Score many papers concurrently and persist incrementally to one JSON file.

    Mirrors the batch pattern in ``main.run_pipeline_batch``: a ThreadPoolExecutor
    fans out ``score_paper`` calls, and each successful result is appended to
    ``output_path`` under a shared lock so partial progress survives a crash.

    Args:
        paper_mds:   ``{paper_id: markdown_text}``. Keys become JSON keys.
        client:      DeepSeek OpenAI-compatible client.
        summaries:   Optional ``{paper_id: summaries_dict}``. Papers without an
                     entry are scored with no related-work context.
        output_path: JSON file to append into (default ``out/ai_results.json``).
        max_workers: Worker count (default ``config.PIPELINE_WORKERS``).
        resume:      If True and ``output_path`` exists, skip papers already
                     present and merge new results into the same file.

    Returns:
        ``{paper_id: DimensionScores}`` including any papers loaded via resume.
    """
    import concurrent.futures

    output_path = Path(output_path)
    summaries = summaries or {}
    max_workers = max_workers or config.PIPELINE_WORKERS

    results: dict[str, DimensionScores] = {}
    if resume and output_path.exists():
        try:
            results = json.loads(output_path.read_text(encoding="utf-8"))
            logger.info("Resuming from %s (%d papers already scored).", output_path, len(results))
        except json.JSONDecodeError:
            logger.warning("Could not parse %s; starting fresh.", output_path)

    pending = {pid: md for pid, md in paper_mds.items() if pid not in results}
    if not pending:
        logger.info("All %d papers already scored in %s.", len(paper_mds), output_path)
        return results

    def _worker(pid: str, md: str) -> tuple[str, DimensionScores]:
        return pid, score_paper(md, summaries.get(pid, {}), client)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, pid, md): pid for pid, md in pending.items()}
        for fut in concurrent.futures.as_completed(futures):
            pid = futures[fut]
            try:
                _, scores = fut.result()
            except Exception as exc:
                logger.error("Scoring failed for %s: %s", pid, exc)
                continue
            results[pid] = scores
            _append_scores_json(output_path, pid, scores)
            logger.info("  [batch] Scored %s → %s", pid, output_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch-score papers on 7 quality dimensions.")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        required=True,
        help="Directory to search (recursively) for *.md paper files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/ai_results.json"),
        help="Shared JSON file to append per-paper scores into.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent workers (default: config.PIPELINE_WORKERS).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing entries in --output and re-score every paper.",
    )
    parser.add_argument("--deepseek-api-key", default=None, help="DeepSeek API key.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    md_files = sorted(args.markdown_dir.rglob("*.md"))

    if not md_files:
        raise SystemExit(f"No .md files found under {args.markdown_dir}")

    paper_mds = {p.stem: p.read_text(encoding="utf-8") for p in md_files}

    client = OpenAI(
        api_key=args.deepseek_api_key or config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    results = score_paper_batch(
        paper_mds,
        client=client,
        output_path=args.output,
        max_workers=args.workers,
        resume=not args.no_resume,
    )
    print(f"Done. {len(results)} papers in {args.output}.")