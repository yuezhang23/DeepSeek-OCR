"""
Shared section-prep stage for both dimensional scorers (`scorer.py` / `scorer_deepseek.py`).

Stage 9b needs two DeepSeek-derived artifacts before any dimension can be scored, and both
scorer backends (the Claude multi-agent scorer and the DeepSeek/LangGraph scorer) consume
the EXACT same two artifacts:

  1. ``paper_sections`` — ``build_paper_sections`` splits the markdown into major sections
     and attaches a faithful per-section summary (DeepSeek summarization).
  2. ``selections`` — ``section_control`` decides, per section per dimension, whether the
     dimension sees that section's raw text (0), its summary (1), or nothing (2). Unknown
     section names are classified by a DeepSeek call and appended to ``selection_strategy.md``.

Both are expensive (DeepSeek calls) and identical regardless of which backend ultimately
scores the paper, so they are extracted here and CACHED — ``prepare_paper_sections`` returns
``(paper_sections, selections)`` reading from / writing to the pipeline cache (and the
``step_outputs`` mirror) so a rerun, or a switch between the two scorer backends, reuses the
work instead of re-spending the API calls.

``section_control`` is re-exported by ``scorer`` for backward compatibility (so existing
``from review_pipeline.scorer import section_control`` imports keep working), but it lives
here and carries no Claude dependency.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from review_pipeline.paper_prep import build_paper_sections  # noqa: F401  (re-exported)
from review_pipeline.tools import DIMENSIONS, SECTION_IMPORTANCE_TOOL
from review_pipeline.clients import get_tool_call

logger = logging.getLogger(__name__)

# Living section × dimension reading-strategy table (0=raw, 1=summary, 2=omit). New section
# names discovered at scoring time are appended back to it (see section_control).
_SELECTION_TABLE_PATH = Path(__file__).parent / "selection_strategy.md"
_TABLE_LOCK = threading.Lock()  # serialize read-modify-write across batch worker threads


# ─── Section control: per-section, per-dimension reading strategy ───────────────
# Exact (normalized) aliases mapping common paper headings to canonical table rows so the
# obvious sections never need an LLM call. Anything not resolved here OR present in the
# (growing) table is sent to the LLM, then appended to selection_strategy.md.
_ALIASES = {
    "abstract": "Abstract",
    "intro": "Introduction", "introduction": "Introduction",
    "related work": "Related Work", "prior work": "Related Work",
    "background": "Background / Preliminaries", "preliminaries": "Background / Preliminaries",
    "method": "Method / Approach", "methods": "Method / Approach",
    "approach": "Method / Approach", "methodology": "Method / Approach",
    "experiment": "Experiments / Setup", "experiments": "Experiments / Setup",
    "experimental setup": "Experiments / Setup", "setup": "Experiments / Setup",
    "result": "Results", "results": "Results",
    "ablation": "Ablation Study", "ablations": "Ablation Study", "ablation study": "Ablation Study",
    "discussion": "Discussion / Analysis",
    "limitation": "Limitations", "limitations": "Limitations",
    "conclusion": "Conclusion", "conclusions": "Conclusion",
    "references": "References", "bibliography": "References",
    "appendix": "Appendix / Supplementary",
    "broader impact": "Broader Impact / Ethics", "ethics statement": "Broader Impact / Ethics",
}

# Leading section label to strip for matching: "1", "3.2", "A", "A.", "IV", optionally
# followed by a separator and whitespace.
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[a-z](?:\.\d+)*|[ivxlcdm]+)[.):]?\s+", re.IGNORECASE
)


def _clamp012(v: Any, default: int = 1) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v in (0, 1, 2) else default


def _normalize_section(title: str) -> str:
    """Lowercase, drop leading numbering ('1', '3.2', 'A.', roman), and collapse whitespace
    so 'A RELATED WORK' / '1 Introduction' map onto canonical table rows."""
    t = title.replace("*", "").replace("`", "").strip().lower()
    t = _LEADING_LABEL_RE.sub("", t, count=1)
    t = re.sub(r"\s+", " ", t).strip(" .:-")
    return t


def _load_selection_table() -> dict[str, dict[str, int]]:
    """Parse the Section × dimension matrix from selection_strategy.md into
    {normalized_section_name: {dim: 0|1|2}}. Returns {} if the file/table is missing."""
    table: dict[str, dict[str, int]] = {}
    if not _SELECTION_TABLE_PATH.exists():
        return table
    lines = _SELECTION_TABLE_PATH.read_text(encoding="utf-8").splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0].lower() == "section" and len(cells) >= 1 + len(DIMENSIONS):
            header_i = i
            break
    if header_i is None:
        return table
    for ln in lines[header_i + 1:]:
        if not ln.lstrip().startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not cells or all(set(c) <= set("-: ") for c in cells):  # separator row
            continue
        display, vals = cells[0], cells[1:1 + len(DIMENSIONS)]
        if len(vals) < len(DIMENSIONS):
            continue
        try:
            choices = {dim: _clamp012(int(v)) for dim, v in zip(DIMENSIONS, vals)}
        except ValueError:
            continue
        table[_normalize_section(display)] = choices
    return table


def _resolve_row(norm: str, table: dict[str, dict[str, int]]) -> dict[str, int] | None:
    """Look a normalized section name up in the table directly, then via the alias map."""
    if norm in table:
        return table[norm]
    canon = _ALIASES.get(norm)
    if canon:
        return table.get(_normalize_section(canon))
    return None


def _query_section_importance(items: list[tuple[str, str]], vendor) -> dict[str, dict[str, int]]:
    """Ask the vendor LLM for a 0/1/2 vector per (title, content) section. Returns
    {title: {dim: choice}}; empty on any failure (callers default misses to summary)."""
    if not items or vendor is None:
        return {}
    listing = "\n\n".join(f"### {title}\n{(content or '')[:2000]}" for title, content in items)
    system = (
        "You are an expert ML/CS peer-review assistant. For each paper section you are given, "
        "decide — for each of the 7 evaluation dimensions — how that section should be "
        "presented to a reviewer of that dimension. Use the section's content to judge its "
        "importance. Return exactly one integer per dimension via submit_section_importance."
    )
    user = (
        "For each section below choose, per dimension: 0 = the RAW full text is required "
        "(detail is critical), 1 = a SUMMARY preserves the key information, 2 = OMIT (little "
        "significance to that dimension). Echo each section title exactly.\n\n"
        f"{listing}\n\nCall submit_section_importance now."
    )
    try:
        # NB: keep tool_choice="auto" (the default) — a forced tool_choice 400s under
        # DeepSeek's thinking mode. Disable thinking: this is a bulk classification and
        # reasoning tokens otherwise exhaust the budget before the tool call is emitted.
        resp = vendor.chat(
            system=system,
            user=user,
            max_tokens=4096,
            tools=[SECTION_IMPORTANCE_TOOL],
            thinking=False,
        )
        args = json.loads(get_tool_call(resp).function.arguments)
    except Exception as exc:  # noqa: BLE001 — fall back to summary defaults
        logger.warning("[section_prep] section-importance query failed: %s", exc)
        return {}
    out: dict[str, dict[str, int]] = {}
    for entry in args.get("selections", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("section")
        if not name:
            continue
        out[name] = {dim: _clamp012(entry.get(dim)) for dim in DIMENSIONS}
    return out


def _append_table_rows(new_rows: list[tuple[str, dict[str, int]]]) -> None:
    """Append newly learned section rows to selection_strategy.md (process-locked, dedup'd)."""
    if not new_rows:
        return
    with _TABLE_LOCK:
        existing = _load_selection_table()  # re-read inside the lock for dedup
        text = _SELECTION_TABLE_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        pipe_idxs = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")]
        if not pipe_idxs:
            return
        insert = []
        for display, vec in new_rows:
            if _normalize_section(display) in existing:
                continue
            cells = " | ".join(str(_clamp012(vec.get(d))) for d in DIMENSIONS)
            insert.append(f"| {display} | {cells} |")
            existing[_normalize_section(display)] = vec  # guard against intra-batch dups
        if not insert:
            return
        last = pipe_idxs[-1]
        merged = lines[:last + 1] + insert + lines[last + 1:]
        out = "\n".join(merged) + ("\n" if text.endswith("\n") else "")
        _SELECTION_TABLE_PATH.write_text(out, encoding="utf-8")
        logger.info("[section_prep] appended %d new section row(s) to %s", len(insert), _SELECTION_TABLE_PATH.name)


def section_control(paper_sections: dict, vendor: Any = None) -> list[dict[str, int]]:
    """For each section in `paper_sections`, return a {dim: 0|1|2} choice vector.

    Known section names (or aliases) are read from selection_strategy.md; unknown names are
    scored by the vendor LLM (using the section's summary as evidence) and appended back to
    the table so they're reused next time. With no vendor, unknown sections default to summary.
    """
    sections = paper_sections.get("sections", []) if isinstance(paper_sections, dict) else []
    table = _load_selection_table()
    norms = [_normalize_section(s.get("title", "")) for s in sections]
    selections: list[dict[str, int] | None] = [None] * len(sections)

    unique_misses: dict[str, tuple[str, str]] = {}  # norm -> (display_title, content)
    for i, sec in enumerate(sections):
        row = _resolve_row(norms[i], table)
        if row is not None:
            selections[i] = dict(row)
        else:
            unique_misses.setdefault(
                norms[i], (sec.get("title", ""), sec.get("summary") or sec.get("raw", ""))
            )

    learned = _query_section_importance(list(unique_misses.values()), vendor)
    # Map learned vectors (keyed by echoed title) back onto normalized misses.
    learned_by_norm = {
        norm: learned[title] for norm, (title, _) in unique_misses.items() if title in learned
    }
    for i, sec in enumerate(sections):
        if selections[i] is None:
            selections[i] = learned_by_norm.get(norms[i]) or {dim: 1 for dim in DIMENSIONS}

    new_rows = [
        (unique_misses[norm][0], vec)
        for norm, vec in learned_by_norm.items()
        if norm not in table
    ]
    if new_rows:
        _append_table_rows(new_rows)

    if unique_misses:
        logger.info(
            "[section_prep] section_control: %d sections (%d from table, %d learned/defaulted)",
            len(sections), len(sections) - sum(1 for n in norms if n in unique_misses), len(unique_misses),
        )
    return selections  # type: ignore[return-value]


# ─── Cached orchestration shared by both scorer backends ─────────────────────────
def prepare_selections(
    paper_sections: dict, vendor: Any = None, cache: Any = None, force_rerun: bool = False
) -> list[dict[str, int]]:
    """Return the per-section reading-strategy `selections`, reading from / writing to the
    cache (`section_selections` stage) when a cache is supplied so the DeepSeek
    section-importance work is reused across reruns and across the two scorer backends."""
    if cache is not None and not force_rerun and cache.exists("section_selections"):
        logger.info("[section_prep] section_selections cache hit.")
        return cache.load("section_selections")
    selections = section_control(paper_sections, vendor)
    if cache is not None:
        cache.save("section_selections", selections)
    return selections


def prepare_paper_sections(
    paper_md: str, vendor: Any = None, cache: Any = None, force_rerun: bool = False
) -> tuple[dict, list[dict[str, int]]]:
    """Build (or load) `paper_sections` and `selections` — the two DeepSeek-derived artifacts
    both scorer backends need — caching each as its own stage.

    Returns ``(paper_sections, selections)``. Pass the pipeline's cache object (anything with
    ``exists()/load()/save()``) to persist/reuse both stages; pass ``cache=None`` to compute
    fresh without caching (e.g. the standalone smoke tests).
    """
    if cache is not None and not force_rerun and cache.exists("paper_sections"):
        logger.info("[section_prep] paper_sections cache hit.")
        paper_sections = cache.load("paper_sections")
    else:
        paper_sections = build_paper_sections(paper_md, vendor=vendor)
        if cache is not None:
            cache.save("paper_sections", paper_sections)
        logger.info("[section_prep] prepared %d sections", len(paper_sections.get("sections", [])))

    selections = prepare_selections(paper_sections, vendor, cache=cache, force_rerun=force_rerun)
    return paper_sections, selections
