"""
Stage 9-b — Dimensional quality scoring via multi-agent orchestration.

Each of the 7 quality dimensions is scored by its OWN autonomous agent built on the
Claude Agent SDK. Every dimension agent:
  • has its own dimension-specific prompt + the shared `paper-quality-rubric` skill,
  • may gather evidence with WebSearch/WebFetch (general prior art only),
  • may ask the user for clarification via AskUserQuestion — and, when no human answers
    (autonomous batch mode), resolves the uncertainty itself and continues,
  • returns a strict `{rationale, score}` through an in-process MCP `submit_score` tool.

A PreToolUse hook blocks any WebSearch/WebFetch that targets OpenReview, so agents can
never probe the paper's human peer reviews. Filesystem tools are disabled for the same
reason (the human reviews also live locally in metadata_300.json).

The 7 agents run concurrently and their `{rationale, score}` results are assembled
directly into the final `DimensionScores`.

The public `score_paper(paper_md, summaries, vendor) -> DimensionScores` signature is
unchanged so `main.run_pipeline` needs no edits. `vendor` is accepted for backward
compatibility but unused — the agent vendor is selected by `SCORE_AGENT_VENDOR`.

The orchestration always runs on the Claude Agent SDK; only the model behind the agents
changes. By default it runs on **DeepSeek** — `clients.agent_sdk_target` points the SDK's
Claude Code CLI at DeepSeek's Anthropic-compatible endpoint via `ClaudeAgentOptions.env`.
Set `SCORE_AGENT_VENDOR=anthropic` to run Claude natively (via `ANTHROPIC_API_KEY`).

Vendor/model selection (env, with defaults):
  SCORE_AGENT_VENDOR      default deepseek            (deepseek | anthropic | custom)
  SCORE_DIMENSION_MODEL   (optional) override the resolved model name
  SCORE_AGENT_BASE_URL / SCORE_AGENT_API_KEY  (optional) custom Anthropic-compatible gateway
  SCORE_AGENT_CONCURRENCY default 4                   (max dimension agents in flight)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

# Allow running directly (`python -m review_pipeline.scorer …`) by making the package
# root importable before the review_pipeline imports below.
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_agent_sdk import (  # noqa: E402
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from review_pipeline import tools as _tools  # noqa: E402
from review_pipeline.tools import (  # noqa: E402
    DIMENSIONS,  # re-exported for main.py
    SCORE_SCALE_GUIDE,
)
# section_control + the cached section-prep orchestration now live in section_prep (shared,
# Claude-free) so scorer.py and scorer_deepseek.py reuse one implementation. Re-exported here
# for backward compatibility with `from review_pipeline.scorer import section_control`.
from review_pipeline.section_prep import (  # noqa: E402,F401
    prepare_paper_sections,
    section_control,
)

logger = logging.getLogger(__name__)

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

# ─── TypedDicts (shape contract shared with main.py / cache) ───────────────────
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

# ─── Tunables ──────────────────────────────────────────────────────────────────
# Vendor that powers the agents. The orchestration stays on the Claude Agent SDK; only
# the model behind it changes. DeepSeek is the default (routed through clients.py to its
# Anthropic-compatible endpoint); "anthropic" runs Claude natively. SCORE_DIMENSION_MODEL
# optionally overrides the resolved model name.
from review_pipeline.clients import agent_sdk_target  # noqa: E402

_AGENT_VENDOR = os.getenv("SCORE_AGENT_VENDOR", "deepseek")
_DIM_MODEL, _AGENT_ENV = agent_sdk_target(_AGENT_VENDOR)
if os.getenv("SCORE_DIMENSION_MODEL"):
    _DIM_MODEL = os.getenv("SCORE_DIMENSION_MODEL")
_AGENT_CONCURRENCY = max(1, int(os.getenv("SCORE_AGENT_CONCURRENCY", "4")))
_REPO_ROOT = Path(__file__).parent.parent  # where .claude/skills lives

# Cost-control knobs (per dimension agent). The Agent SDK is conversational — every turn
# re-sends the accumulated context — so turns, web scope, a hard $ ceiling, and reasoning
# effort are the levers that actually bound spend. See plan/CLAUDE.md Stage 9b.
_MAX_TURNS = max(1, int(os.getenv("SCORE_MAX_TURNS", "8")))
_EFFORT = os.getenv("SCORE_EFFORT", "low")  # low|medium|high|xhigh|max

# Hard per-agent $ ceiling. The CLI computes cost at *Anthropic* pricing regardless of the
# endpoint, so on a non-Anthropic vendor (DeepSeek et al.) a small cap trips almost
# immediately and kills agents before they submit. Default it OFF for those vendors (turns +
# web-scoping already bound spend; DeepSeek is cheap) and ON only for native Anthropic. An
# explicit SCORE_MAX_BUDGET_USD always wins; set it to 0 to disable.
_default_budget = "0" if _AGENT_ENV else "0.20"
_b = float(os.getenv("SCORE_MAX_BUDGET_USD", _default_budget))
_MAX_BUDGET_USD = _b if _b > 0 else None

# Only these dimensions benefit from prior-art lookups (novelty / literature comparison);
# they also get the related-work summaries as an in-context fallback. Every other dimension
# is an internal judgment over the paper text and gets NO web tools — web pages are large
# and, with conversational re-billing, dominate cost.
_WEB_DIMENSIONS = {
    d.strip()
    for d in os.getenv(
        "SCORE_WEB_DIMENSIONS",
        "originality,contextualization_relative_to_prior_work",
    ).split(",")
    if d.strip()
}

# Filesystem / shell tools the scoring agents must never use — reading the repo would
# leak the ground-truth human reviews in metadata_300.json.
_FS_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "Grep", "Glob"}

# ─── Permission gate ────────────────────────────────────────────────────────────
async def _gate_tool(tool_name: str, input_data: dict, context):
    """can_use_tool: auto-resolve AskUserQuestion (autonomous mode) and hard-deny
    filesystem/shell tools; allow everything else (web/skill/submit)."""
    if tool_name == "AskUserQuestion":
        questions = input_data.get("questions", [])
        summary = "; ".join(q.get("question", "") for q in questions) if isinstance(questions, list) else str(questions)
        logger.info("[scorer] AskUserQuestion auto-resolved (no human in autonomous mode): %s", summary)
        return PermissionResultDeny(
            message=(
                "No human reviewer is available. Resolve the uncertainty yourself using "
                "your best expert judgment, state the assumption in your rationale, and "
                "continue to a final score."
            )
        )
    if tool_name in _FS_TOOLS:
        return PermissionResultDeny(
            message="Filesystem/shell access is disabled for scoring agents."
        )
    return PermissionResultAllow(updated_input=input_data)


# ─── In-process MCP "submit" tools (strict structured output) ──────────────────
def _make_dimension_recorder():
    """Per-run MCP server exposing `submit_score`; returns (server, holder)."""
    holder: dict[str, Any] = {}

    @tool(
        "submit_score",
        "Record your FINAL rationale and integer score (1-10) for this single dimension. "
        "Write the rationale FIRST, then choose the score that follows from it. "
        "Call this exactly once when you are done; produce no other final output.",
        {"rationale": str, "score": int},
    )
    async def submit_score(args: dict) -> dict:
        holder["rationale"] = str(args.get("rationale", "")).strip()
        try:
            holder["score"] = max(1, min(10, int(args.get("score"))))
        except (TypeError, ValueError):
            holder["score"] = None
        return {"content": [{"type": "text", "text": "Score recorded. You may stop now."}]}

    server = create_sdk_mcp_server(name="score", version="1.0.0", tools=[submit_score])
    return server, holder


# ─── Prompts ────────────────────────────────────────────────────────────────────
_DIMENSION_SYSTEM = """\
You are a meticulous, world-class peer reviewer for top ML/CS venues (NeurIPS, ICML, \
ICLR, ACL, CVPR). You are evaluating ONE paper on EXACTLY ONE quality dimension — ignore \
all other dimensions:

  Dimension : {label}  (key: {key})
  Measures  : {desc}

{scale}

Method:
1. Read the paper text in the user message (for long papers this may be a faithful
   section-by-section summary rather than the raw prose).
2. Invoke the `paper-quality-rubric` skill and apply its criteria for THIS dimension and
   the shared 1-10 scale.
3. Ground every judgment in concrete evidence (specific sections, tables, claims,
   baselines). {web_clause}You MUST NOT look up or rely on this paper's peer reviews,
   ratings, rebuttals, or accept/reject decision (especially on OpenReview). Score on
   merits only.
4. If something is genuinely ambiguous you may use AskUserQuestion; if no answer comes
   back, resolve it yourself with expert judgment, note the assumption, and proceed.
5. Write a 2-4 sentence, evidence-grounded rationale FIRST, then choose the integer score
   (1-10) that the rationale implies — using the score interpretation above; never pick the
   number first.
6. Finish by calling `submit_score` exactly once with your rationale and score."""

# Filled into {web_clause} of _DIMENSION_SYSTEM depending on whether this dimension's agent
# is granted web tools.
_WEB_CLAUSE_ON = (
    "You MAY use WebSearch/WebFetch to verify prior art, novelty, or context — "
)
_WEB_CLAUSE_OFF = (
    "Base your judgment solely on the paper text and the related-work summaries provided "
    "below; do not attempt web searches. "
)


_DIMENSION_USER = """\
Evaluate the dimension "{label}" for the following paper, then call submit_score.

=== PAPER ===
{paper}
{related}"""


async def _single_user_msg(text: str):
    """One-shot streaming-input prompt. `can_use_tool` requires streaming mode, so the
    prompt must be an AsyncIterable of message dicts rather than a plain string. Yielding
    a single user message and returning closes the input stream so the agent runs one turn
    and finishes."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
        "session_id": "default",
    }


def _build_related_work_context(summaries: dict) -> str:
    if not summaries:
        return ""
    lines = ["\n=== RELATED WORK SUMMARIES ===\n"]
    for i, (arxiv_id, info) in enumerate(summaries.items(), 1):
        title = info.get("title", arxiv_id) if isinstance(info, dict) else getattr(info, "title", arxiv_id)
        summary = info.get("summary", "") if isinstance(info, dict) else getattr(info, "summary", "")
        lines.append(f"[{i}] {title} (arXiv:{arxiv_id})\n{summary}\n")
    return "\n".join(lines)


# ─── Agent runs ─────────────────────────────────────────────────────────────────
def _dimension_options(server, allow_web: bool) -> ClaudeAgentOptions:
    allowed = ["Skill", "TodoWrite", "mcp__score__submit_score"]
    if allow_web:
        allowed = ["WebSearch", "WebFetch", *allowed]
    return ClaudeAgentOptions(
        model=_DIM_MODEL,
        cwd=str(_REPO_ROOT),
        setting_sources=["project"],   # discover .claude/skills/paper-quality-rubric
        skills="all",
        mcp_servers={"score": server},
        allowed_tools=allowed,
        disallowed_tools=sorted(_FS_TOOLS),
        can_use_tool=_gate_tool,
        permission_mode="default",
        max_turns=_MAX_TURNS,
        max_budget_usd=_MAX_BUDGET_USD,
        effort=_EFFORT,
        env=_AGENT_ENV,  # routes the CLI to the chosen vendor (DeepSeek by default)
    )


async def _run_dimension(dim: str, paper: str, related: str, sem: asyncio.Semaphore) -> DimensionEntry:
    label = DIMENSION_LABELS[dim]
    allow_web = dim in _WEB_DIMENSIONS
    cost = 0.0
    async with sem:
        server, holder = _make_dimension_recorder()
        opts = _dimension_options(server, allow_web)
        opts.system_prompt = _DIMENSION_SYSTEM.format(
            label=label,
            key=dim,
            desc=_tools._DIMENSION_DESCRIPTIONS[dim],
            scale=SCORE_SCALE_GUIDE,
            web_clause=_WEB_CLAUSE_ON if allow_web else _WEB_CLAUSE_OFF,
        )
        prompt = _DIMENSION_USER.format(label=label, paper=paper, related=related)
        try:
            async for msg in query(prompt=_single_user_msg(prompt), options=opts):
                if isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0.0
                    if not holder.get("rationale"):
                        # Fallback: try to recover from the agent's final text if it forgot
                        # to call submit_score (incl. budget/turn-capped early exits).
                        _maybe_recover_dim(holder, msg.result or "")
        except Exception as exc:  # noqa: BLE001 — never let one dimension crash the batch
            logger.warning("[scorer] dimension '%s' agent failed: %s", dim, exc)

    score = holder.get("score")
    rationale = holder.get("rationale") or "(agent produced no rationale; defaulted)"
    if score is None:
        logger.warning("[scorer] dimension '%s' produced no score; defaulting to 5", dim)
        score = 5
    logger.info("[scorer] %s -> %d/10  ($%.3f, web=%s)", label, score, cost, allow_web)
    return {"rationale": rationale, "score": int(score), "_cost_usd": cost}  # type: ignore[typeddict-unknown-key]


def _maybe_recover_dim(holder: dict, text: str) -> None:
    """Best-effort extraction of {rationale, score} from a free-text final message."""
    obj = _extract_json_obj(text)
    if isinstance(obj, dict) and "score" in obj:
        holder.setdefault("rationale", str(obj.get("rationale", "")).strip())
        try:
            holder.setdefault("score", max(1, min(10, int(obj["score"]))))
        except (TypeError, ValueError):
            pass


# ─── Result assembly ────────────────────────────────────────────────────────────
def _extract_json_obj(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _assemble(dim_results: dict[str, DimensionEntry]) -> DimensionScores:
    """Assemble the final DimensionScores from the per-dimension agent results, ensuring
    all 7 keys are present with valid 1-10 scores."""
    final: dict[str, DimensionEntry] = {}
    for dim in DIMENSIONS:
        entry = dim_results.get(dim, {"rationale": "(missing)", "score": 5})
        try:
            score = max(1, min(10, int(entry["score"])))
        except (TypeError, ValueError, KeyError):
            score = 5
        final[dim] = {"rationale": str(entry.get("rationale") or "(missing)").strip(), "score": score}
    return final  # type: ignore[return-value]


def _build_dimension_text(paper_sections: dict, selections: list[dict[str, int]], dim: str) -> str:
    """Concatenate sections for one dimension: raw (0), summary (1), or omitted (2)."""
    sections = paper_sections.get("sections", []) if isinstance(paper_sections, dict) else []
    parts: list[str] = []
    for sec, sel in zip(sections, selections):
        choice = sel.get(dim, 1) if isinstance(sel, dict) else 1
        if choice == 0:
            body = sec.get("raw", "")
        elif choice == 1:
            body = sec.get("summary") or sec.get("raw", "")
        else:
            continue
        if body.strip():
            parts.append(f"## {sec.get('title', '')}\n{body.strip()}")
    return "\n\n".join(parts)


async def _score_paper_async(
    paper_sections: dict, selections: list[dict[str, int]], summaries: dict
) -> DimensionScores:
    related = _build_related_work_context(summaries)
    sem = asyncio.Semaphore(_AGENT_CONCURRENCY)

    async def _run_dim(dim: str) -> DimensionEntry:
        text = _build_dimension_text(paper_sections, selections, dim)
        return await _run_dimension(dim, text, related, sem)

    results = await asyncio.gather(*(_run_dim(dim) for dim in DIMENSIONS))
    dim_results: dict[str, DimensionEntry] = dict(zip(DIMENSIONS, results))
    total_cost = sum(float(r.get("_cost_usd", 0.0) or 0.0) for r in results)  # type: ignore[arg-type]
    logger.info("[scorer] paper scored: $%.3f total (%d dims)", total_cost, len(results))
    cost_report = {
        "vendor": _AGENT_VENDOR,
        "model": _DIM_MODEL,
        # On a non-Anthropic vendor the CLI still prices tokens at Anthropic rates, so the
        # figures are an over-estimate, not the real DeepSeek bill — label it honestly.
        "cost_basis": "native" if not _AGENT_ENV else "anthropic-cli-estimate",
        "total_cost_usd": round(total_cost, 4),
        "dimensions": {
            dim: {
                "score": dim_results[dim].get("score"),
                "cost_usd": round(float(dim_results[dim].get("_cost_usd", 0.0) or 0.0), 4),  # type: ignore[arg-type]
                "web": dim in _WEB_DIMENSIONS,
            }
            for dim in DIMENSIONS
        },
    }
    return _assemble(dim_results), cost_report


def _write_cost_log(path: str | Path, paper_id: str | None, report: dict) -> None:
    """Append one JSON-lines API-cost record for this paper next to the main output.

    JSONL (one record per paper) so batch runs accumulate without clobbering. Each record
    is {paper_id, vendor, model, cost_basis, total_cost_usd, dimensions:{dim:{score,cost_usd,
    web}}}. For non-Anthropic vendors cost_basis flags the totals as an Anthropic-priced
    CLI estimate, not the real vendor bill.
    """
    record = {"paper_id": paper_id, **report}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        "[scorer] cost log appended -> %s ($%.4f, basis=%s)",
        p, report.get("total_cost_usd", 0.0), report.get("cost_basis"),
    )


# ─── Public API ─────────────────────────────────────────────────────────────────
def score_paper(
    paper_sections: dict,
    summaries: dict,
    vendor: Any = None,
    selections: list[dict[str, int]] | None = None,
    cost_log_path: str | Path | None = None,
    paper_id: str | None = None,
) -> DimensionScores:
    """Score the paper on 7 dimensions via multi-agent orchestration.

    `paper_sections` is the prep-stage output: {"sections": [{title, raw, summary}, ...]}.
    `selections` is the per-section/per-dimension raw/summary/omit plan from
    `section_prep.section_control`. It is normally precomputed and cached by
    `section_prep.prepare_paper_sections` (so both scorer backends reuse it); when omitted it
    is computed here on the fly (using `vendor` for any unknown-section importance lookups).
    Each dimension agent then receives its own tailored paper text; the 7 agents run on the
    vendor selected by `SCORE_AGENT_VENDOR` (DeepSeek by default).

    If `cost_log_path` is given, a per-paper API-cost record is appended there as JSON-lines
    (keyed by `paper_id`) alongside the main score output — see `_write_cost_log`.
    """
    logger.info(
        "[scorer] multi-agent scoring: vendor=%s model=%s concurrency=%d",
        _AGENT_VENDOR, _DIM_MODEL, _AGENT_CONCURRENCY,
    )
    if selections is None:
        selections = section_control(paper_sections, vendor)
    scores, cost_report = asyncio.run(
        _score_paper_async(paper_sections, selections, summaries)
    )
    if cost_log_path:
        _write_cost_log(cost_log_path, paper_id, cost_report)
    return scores


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    logging.basicConfig(level=logging.INFO)
    from review_pipeline.paper_prep import build_paper_sections
    from review_pipeline.clients import LLMVendor

    path = sys.argv[1] if len(sys.argv) > 1 else "./data/mds_a/4.md"
    md = Path(path).read_text(encoding="utf-8")
    try:
        _vendor = LLMVendor.default()
    except Exception:  # noqa: BLE001 — allow running with no vendor configured
        _vendor = None
    _sections = build_paper_sections(md, vendor=_vendor)

    _paper_id = Path(path).stem
    _out_dir = Path(os.getenv("SCORE_OUT_DIR", "out"))
    _scores_path = _out_dir / f"{_paper_id}.scores.json"
    _cost_path = _out_dir / "scorer_costs.jsonl"

    out = score_paper(
        _sections, {}, vendor=_vendor, cost_log_path=_cost_path, paper_id=_paper_id
    )

    _out_dir.mkdir(parents=True, exist_ok=True)
    _scores_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[scorer] scores  -> {_scores_path}")
    print(f"[scorer] costs   -> {_cost_path}")
