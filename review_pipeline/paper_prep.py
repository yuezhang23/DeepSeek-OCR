"""
Paper preparation stage — split a paper's markdown into MAJOR sections and attach a
faithful per-section summary.

This runs as its own cached pipeline stage (``paper_sections``) BEFORE Stage 9b scoring.
Its output, a JSON object of the form::

    {"sections": [{"title": str, "raw": str, "summary": str}, ...]}

is consumed by ``scorer.section_control`` to decide — per section, per dimension — whether
each dimension agent sees that section's raw text, its summary, or nothing at all.

The split is deterministic Python (no LLM); only the per-section summaries call the LLM
(the OpenAI-compatible ``vendor`` threaded through the pipeline).

This project's markdown (DeepSeek-OCR output) follows a fixed convention: the paper **title**
is the single level-1 heading (``#``) and **major section** names are level-2 headings
(``##``). So splitting happens at ``##`` boundaries — the ``#`` title and everything before
the first ``##`` become "Front Matter". Two wrinkles in the OCR output are handled: appendix
subsections are sometimes emitted at ``##`` too (``## B.1``, ``## C.2``, ``## D.1``) and inline
blocks occasionally are (``## Algorithm 1``, ``## QA Example``). Both are folded into their
parent via a dotted-number guard + an enumerator/keyword check, so the granularity matches the
canonical rows in ``selection_strategy.md`` — a typical paper yields ~10-20 sections, not one
per heading.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re

from review_pipeline import config

logger = logging.getLogger(__name__)

# Sections at or below this size are kept verbatim as their own "summary" (skip the LLM call).
_SHORT_SECTION_CHARS = int(getattr(config, "PREP_SHORT_SECTION_CHARS", 0) or 1200)
# Hard cap on the raw text sent to the summarizer for one (huge) section.
_MAX_SUMMARY_INPUT_CHARS = int(getattr(config, "PREP_MAX_SUMMARY_INPUT_CHARS", 0) or 60000)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Leading "1", "4", "A", "B" … but NOT dotted ("3.1") nor multi-letter run ("Algorithm").
_TOP_TOKEN_RE = re.compile(r"^(?:\d+|[A-Z])\b(?!\.)")
# Subsection numbering: "3.1", "B.4.3", "C.2", "D.1", "E.1.2" — always a subsection, never
# a major section, even when emitted at ## (OCR) or when its text contains a major keyword.
_SUB_NUMBER_RE = re.compile(r"^(?:\d+|[A-Za-z])\.\d")
# Markdown level at which major sections live by project convention (## = title's children).
_MAJOR_HEADING_LEVEL = 2
# Known major-section names (matched against the cleaned, lowercased heading text).
_MAJOR_KEYWORDS = (
    "abstract", "introduction", "related work", "prior work", "background",
    "preliminaries", "method", "approach", "methodology", "experiment", "result",
    "ablation", "discussion", "analysis", "limitation", "conclusion", "references",
    "bibliography", "acknowledg", "appendix", "broader impact", "ethic",
)

_PAGE_NUM_TAIL_RE = re.compile(r"\s+\d{1,3}\s*$")  # strip trailing TOC page numbers


def _clean_title(text: str) -> str:
    """Strip markdown emphasis and trailing TOC page numbers from a heading title."""
    t = text.replace("*", "").replace("`", "").strip()
    t = _PAGE_NUM_TAIL_RE.sub("", t).strip()
    return t


def _is_major_heading(level: int, title: str, major_level: int) -> bool:
    """A heading at the major level (## by convention; the # title is excluded) that is a real
    section: not a dotted subsection (3.1, B.2.1) and either enumerated (1, A, B…) or named by
    a known major keyword. Other levels and one-off ## blocks (Algorithm N, QA Example) fold in.
    """
    if level != major_level:
        return False
    if _SUB_NUMBER_RE.match(title):
        return False
    if _TOP_TOKEN_RE.match(title):
        return True
    low = title.lower()
    return any(kw in low for kw in _MAJOR_KEYWORDS)


def _split_major_sections(paper_md: str) -> list[dict]:
    """Split markdown into major sections [{title, raw}], folding subsections into parents.

    Major sections are level-2 (``##``) headings per the project convention — the ``#`` title
    and anything before the first major section become "Front Matter". Falls back to level-1 for
    documents that contain no ``##`` headings; a paper with no headings becomes "Full Paper".
    """
    lines = paper_md.splitlines(keepends=True)

    # Honor the ## convention when present; gracefully handle docs that only use #.
    levels = {len(m.group(1)) for ln in lines if (m := _HEADING_RE.match(ln))}
    major_level = _MAJOR_HEADING_LEVEL if _MAJOR_HEADING_LEVEL in levels else (min(levels) if levels else _MAJOR_HEADING_LEVEL)

    sections: list[dict] = []
    cur_title: str | None = None
    cur_buf: list[str] = []

    def flush():
        if cur_buf:
            title = cur_title if cur_title is not None else "Front Matter"
            raw = "".join(cur_buf).strip()
            if raw:
                sections.append({"title": title, "raw": raw})

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            title = _clean_title(m.group(2))
            if title and _is_major_heading(len(m.group(1)), title, major_level):
                flush()
                cur_title = title
                cur_buf = [line]
                continue
        cur_buf.append(line)
    flush()

    if not sections:
        return [{"title": "Full Paper", "raw": paper_md.strip()}]
    return sections


_SUMMARY_SYSTEM = (
    "You are condensing ONE section of an academic paper for downstream expert peer "
    "reviewers. Write a concise summary that PRESERVES the detail reviewers depend on: "
    "quantitative results, exact numbers and table values, baselines compared against, the "
    "specific claims made, datasets/benchmarks used, and key equations or method details. "
    "Do not editorialize, critique, or score — summarize faithfully. Output only the summary."
)


def _summarize_section(title: str, raw: str, vendor) -> str:
    """Return a faithful summary of one section. Short sections are returned verbatim; a
    failed/absent LLM call falls back to a truncated raw so the stage never crashes."""
    if vendor is None or len(raw) <= _SHORT_SECTION_CHARS:
        return raw
    user = f"Summarize this section faithfully.\n\n## {title}\n{raw[:_MAX_SUMMARY_INPUT_CHARS]}"
    try:
        resp = vendor.chat(system=_SUMMARY_SYSTEM, user=user, max_tokens=1024, tools=None)
        content = (resp.choices[0].message.content or "").strip()
        return content or raw[:_SHORT_SECTION_CHARS]
    except Exception as exc:  # noqa: BLE001 — one bad section must not kill the stage
        logger.warning("[paper_prep] summary failed for section %r: %s", title, exc)
        return raw[:_SHORT_SECTION_CHARS]


def build_paper_sections(paper_md: str, vendor=None) -> dict:
    """Split ``paper_md`` into major sections and attach a per-section summary.

    Returns ``{"sections": [{"title", "raw", "summary"}, ...]}``. Summaries are generated
    concurrently (bounded by ``config.SUMMARY_WORKERS``) via the OpenAI-compatible ``vendor``;
    pass ``vendor=None`` to skip summarization (each ``summary`` falls back to its ``raw``).
    """
    sections = _split_major_sections(paper_md)
    logger.info("[paper_prep] split into %d major sections (%d chars)", len(sections), len(paper_md))

    workers = max(1, min(config.SUMMARY_WORKERS, len(sections)))
    summaries: list[str] = [""] * len(sections)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_summarize_section, s["title"], s["raw"], vendor): i
            for i, s in enumerate(sections)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            summaries[i] = fut.result()

    for s, summary in zip(sections, summaries):
        s["summary"] = summary
    return {"sections": sections}
