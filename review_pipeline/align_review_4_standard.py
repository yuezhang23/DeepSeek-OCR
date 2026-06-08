"""
Parse the simulated 4-reviewer review files under ``out/review_4_standard/``
and run them through ``review_scoring.align_reviews_to_dimensions`` to
produce a JSON file with the same shape as ``alignment_results.json``.

File naming: ``review_<KEY>_4*.txt`` — KEY is the digits between the first
two underscores (e.g. ``review_23_4.txt`` -> ``"23"``). Some files have a
``(1)`` or ``(2)`` suffix; KEY is still extracted from the same position.

Each file embeds the 4 reviewers inside a ``\\boxed_simreviewers{ ... }``
block. We parse the markdown sections (Summary, Soundness, Presentation,
Contribution, Strengths, Weaknesses, Questions, Rating, Confidence) into
dicts that match the ``official_values`` shape expected by
``review_scoring._format_reviews``.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from review_pipeline import config
from review_pipeline.review_scoring import align_reviews_to_dimensions
from review_pipeline.clients import build_llm_client

logger = logging.getLogger(__name__)

REVIEW_DIR = Path(__file__).parent / "out" / "review_4_standard"
DEFAULT_OUT_PATH = Path(__file__).parent / "out" / "deepreview_review_4_ds.json"

_FILENAME_RE = re.compile(r"^review_(\d+)_4")
_BOXED_RE = re.compile(r"\\boxed_simreviewers\s*\{(.*?)\n\}", re.DOTALL)
# Matches "## Reviewer 1" OR "--- Reviewer 1 ---" / "<--- Review 1 --->" / "--- Review 1 ---"
_REVIEWER_SPLIT_RE = re.compile(
    r"^(?:##\s*Reviewer\s*\d+\s*$|<?-{2,}\s*Review(?:er)?\s*\d+\s*-{2,}>?\s*$)",
    re.MULTILINE,
)
# Section header at level 2 or 3, optional trailing colon (handles "## Summary:" too).
_SECTION_RE = re.compile(r"^#{2,3}\s+(.+?):?\s*$", re.MULTILINE)
# Ignore these section labels so they don't get stuffed into reviewer fields.
_IGNORE_SECTIONS = {
    "decision", "actual decision", "meta review", "meta-review",
    "final review output", "review", "validation",
}

_SECTION_MAP = {
    "summary": "summary",
    "strengths": "strengths",
    "weaknesses": "weaknesses",
    "questions": "questions",
    "suggestions": "suggestions",
    "rating": "rating",
    "confidence": "confidence",
    "soundness": "soundness",
    "presentation": "presentation",
    "contribution": "contribution",
}
_NUMERIC_KEYS = {"rating", "confidence", "soundness", "presentation", "contribution"}


def key_from_filename(name: str) -> str | None:
    m = _FILENAME_RE.match(name)
    return m.group(1) if m else None


def extract_boxed(text: str) -> str | None:
    m = _BOXED_RE.search(text)
    return m.group(1) if m else None


def _parse_sections(block: str) -> dict[str, str]:
    matches = list(_SECTION_RE.finditer(block))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        label = m.group(1).strip().lower()
        if label in _IGNORE_SECTIONS:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        body = block[start:end].strip()
        if body:
            sections[label] = body
    return sections


def _build_reviewer_dict(sections: dict[str, str], review_id: str) -> dict | None:
    if not sections:
        return None
    rev: dict = {"review_id": review_id}
    for label, body in sections.items():
        key = _SECTION_MAP.get(label)
        if key is None:
            continue
        if key in _NUMERIC_KEYS:
            num = _coerce_numeric(body)
            if num is not None:
                rev[key] = num
        else:
            rev[key] = body
    # Require at least a summary or strengths/weaknesses to consider it a real review.
    if not any(k in rev for k in ("summary", "strengths", "weaknesses", "questions")):
        return None
    return rev


def _coerce_numeric(val: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", val)
    return float(m.group(0)) if m else None


def _split_reviewers(text: str) -> list[str]:
    """Split text on `## Reviewer N` / `--- Reviewer N ---` markers."""
    matches = list(_REVIEWER_SPLIT_RE.finditer(text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end])
    return blocks


def parse_reviewers(text: str, paper_key: str) -> list[dict]:
    """Extract reviewer dicts from a chunk of text.

    Splits on `## Reviewer N` / `--- Reviewer N ---` markers, then for each
    block parses level-2/3 headings into the canonical reviewer fields.
    """
    reviewers: list[dict] = []
    for idx, raw in enumerate(_split_reviewers(text), start=1):
        clean = re.sub(r"\*{5,}\s*$", "", raw).strip()
        sections = _parse_sections(clean)
        rev = _build_reviewer_dict(sections, f"{paper_key}-sim{idx}")
        if rev is not None:
            reviewers.append(rev)
    return reviewers


def load_reviews(path: Path) -> tuple[str, list[dict], str] | None:
    """Return (paper_key, reviewers, source_tag) or None.

    Tries three formats, in order:
      1. boxed   — reviewers inside ``\\boxed_simreviewers{...}``.
      2. inline  — `## Reviewer N` / `--- Reviewer N ---` blocks anywhere in the file.
      3. meta    — no per-reviewer split; parse the whole file as one consolidated review.
    """
    key = key_from_filename(path.name)
    if key is None:
        logger.warning("Could not extract key from %s", path.name)
        return None
    text = path.read_text(encoding="utf-8", errors="replace")

    boxed = extract_boxed(text)
    if boxed:
        reviewers = parse_reviewers(boxed, key)
        if reviewers:
            return key, reviewers, "boxed"

    reviewers = parse_reviewers(text, key)
    if reviewers:
        return key, reviewers, "inline"

    # Format C: no reviewer split markers, but the file may still contain a
    # consolidated review with `## Summary:`, `## Strengths:`, etc. Parse those
    # as a single reviewer entry.
    sections = _parse_sections(text)
    rev = _build_reviewer_dict(sections, f"{key}-meta")
    if rev is not None:
        return key, [rev], "meta"

    logger.warning("No reviewers parsed from %s", path.name)
    return None


def _flush(out_path: Path, results: dict) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse files only; do not call the LLM.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    files = sorted(args.review_dir.glob("review_*_4*.txt"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No review files found under {args.review_dir}")

    # Parse all files up-front so LLM workers only do alignment.
    parsed: dict[str, list[dict]] = {}
    source_counts: dict[str, int] = {}
    for p in files:
        loaded = load_reviews(p)
        if loaded is None:
            continue
        key, reviewers, source = loaded
        if key in parsed:
            logger.warning("Duplicate key %s (file %s) — overwriting.", key, p.name)
        parsed[key] = reviewers
        source_counts[source] = source_counts.get(source, 0) + 1
    logger.info("Parsed %d files (out of %d) with at least one reviewer. Sources: %s",
                len(parsed), len(files), source_counts)

    if args.dry_run:
        sample = next(iter(parsed.items()))
        print(json.dumps({sample[0]: sample[1]}, indent=2)[:2000])
        return

    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set (check .env).")

    results: dict = {}
    if (not args.no_resume) and args.out.exists():
        try:
            results = json.loads(args.out.read_text(encoding="utf-8"))
            logger.info("Resuming from %s (%d already done).", args.out, len(results))
        except json.JSONDecodeError:
            logger.warning("Could not parse %s; starting fresh.", args.out)

    pending = [(k, revs) for k, revs in parsed.items() if k not in results]
    if not pending:
        logger.info("Nothing to do; %s is already complete.", args.out)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = build_llm_client()
    lock = threading.Lock()

    def worker(key: str, reviewers: list[dict]):
        return key, align_reviews_to_dimensions(reviewers, client, paper_title=None)

    with ThreadPoolExecutor(max_workers=max(1, args.batch_size)) as pool:
        futures = {pool.submit(worker, k, revs): k for k, revs in pending}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc=f"Aligning (batch={args.batch_size})", unit="paper")
        for fut in pbar:
            key = futures[fut]
            pbar.set_postfix_str(key)
            try:
                _, scores = fut.result()
            except Exception as exc:
                logger.error("Alignment failed for %s: %s", key, exc)
                continue
            with lock:
                results[key] = scores
                _flush(args.out, results)

    logger.info("Wrote %d results to %s", len(results), args.out)


if __name__ == "__main__":
    main()
