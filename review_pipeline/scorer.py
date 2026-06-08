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

Finally a *judge* agent (Opus) reads all 7 results, resolves cross-dimension
inconsistencies in the rationales, and emits the merged `DimensionScores` that is saved.

The public `score_paper(paper_md, summaries, vendor) -> DimensionScores` signature is
unchanged so `main.run_pipeline` needs no edits. `vendor` is accepted for backward
compatibility but unused — this stage runs on Claude via ANTHROPIC_API_KEY.

Model selection (env, with defaults):
  SCORE_DIMENSION_MODEL  default claude-sonnet-4-6   (the 7 dimension agents)
  SCORE_JUDGE_MODEL      default claude-opus-4-8      (the reconciliation judge)
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
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from review_pipeline import tools as _tools  # noqa: E402
from review_pipeline.tools import DIMENSIONS  # noqa: E402  (re-exported for main.py)

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
_DIM_MODEL = os.getenv("SCORE_DIMENSION_MODEL", "claude-sonnet-4-6")
_JUDGE_MODEL = os.getenv("SCORE_JUDGE_MODEL", "claude-opus-4-8")
_AGENT_CONCURRENCY = max(1, int(os.getenv("SCORE_AGENT_CONCURRENCY", "4")))
_PAPER_CHAR_LIMIT = 100_000  # ~15K tokens; keeps each agent's context modest
_REPO_ROOT = Path(__file__).parent.parent  # where .claude/skills lives

# Filesystem / shell tools the scoring agents must never use — reading the repo would
# leak the ground-truth human reviews in metadata_300.json.
_FS_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "Grep", "Glob"}

# ─── Cross-cutting hook + permission gate ──────────────────────────────────────
async def _block_openreview_hook(input_data: dict, tool_use_id, context) -> dict:
    """PreToolUse: deny any web search/fetch that touches OpenReview, so a scoring
    agent can never read the paper's human peer reviews."""
    tool_name = input_data.get("tool_name", "")
    if tool_name in ("WebSearch", "WebFetch"):
        blob = json.dumps(input_data.get("tool_input", {})).lower()
        if "openreview" in blob:
            logger.info("[scorer] blocked %s targeting OpenReview", tool_name)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Probing human peer reviews from OpenReview is forbidden during "
                        "autonomous scoring. Score only from the paper's own content and "
                        "general prior art."
                    ),
                }
            }
    return {}


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


_PRETOOL_HOOKS = {
    "PreToolUse": [HookMatcher(matcher="WebSearch|WebFetch", hooks=[_block_openreview_hook])]
}

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


def _make_judge_recorder():
    """Per-run MCP server exposing `submit_reconciled_scores`; returns (server, holder)."""
    holder: dict[str, Any] = {}

    @tool(
        "submit_reconciled_scores",
        "Submit the final reconciled scores for ALL 7 dimensions as a single JSON object "
        'mapping each dimension key to {"rationale": <str>, "score": <int 1-10>}. '
        "Call this exactly once.",
        {"scores_json": str},
    )
    async def submit_reconciled_scores(args: dict) -> dict:
        holder["raw"] = args.get("scores_json", "")
        return {"content": [{"type": "text", "text": "Reconciled scores recorded. Stop now."}]}

    server = create_sdk_mcp_server(name="judge", version="1.0.0", tools=[submit_reconciled_scores])
    return server, holder

# ─── Prompts ────────────────────────────────────────────────────────────────────
_DIMENSION_SYSTEM = """\
You are a meticulous, world-class peer reviewer for top ML/CS venues (NeurIPS, ICML, \
ICLR, ACL, CVPR). You are evaluating ONE paper on EXACTLY ONE quality dimension — ignore \
all other dimensions:

  Dimension : {label}  (key: {key})
  Measures  : {desc}

Method:
1. Read the paper text in the user message.
2. Invoke the `paper-quality-rubric` skill and apply its criteria for THIS dimension and
   the shared 1-10 scale.
3. Ground every judgment in concrete evidence (specific sections, figures, tables, claims,
   baselines). You MAY use WebSearch/WebFetch to verify prior art, novelty, or context —
   but you MUST NOT look up or rely on this paper's peer reviews, ratings, rebuttals, or
   accept/reject decision (especially on OpenReview). Score on merits only.
4. If something is genuinely ambiguous you may use AskUserQuestion; if no answer comes
   back, resolve it yourself with expert judgment, note the assumption, and proceed.
5. Write a 2-4 sentence, evidence-grounded rationale FIRST, then choose the integer score
   (1-10) that the rationale implies — never pick the number first.
6. Finish by calling `submit_score` exactly once with your rationale and score."""

_DIMENSION_USER = """\
Evaluate the dimension "{label}" for the following paper, then call submit_score.

=== PAPER ===
{paper}
{related}"""

_JUDGE_SYSTEM = """\
You are the senior area chair reconciling seven INDEPENDENT single-dimension reviews of \
one paper. Each dimension was scored by a separate reviewer who saw only their own \
dimension, so their rationales may conflict.

Your job:
- Read all seven {{rationale, score}} pairs.
- Detect and resolve INCONSISTENCIES across rationales: factual contradictions about the
  paper, a rationale that argues one way but lands on a mismatched score, a claim in one
  dimension that undercuts another, or contradictory evidence.
- Lightly revise rationales so they are mutually consistent and so each score follows from
  its (possibly corrected) rationale. Keep each rationale 2-4 sentences and grounded in the
  paper. Do not invent new evidence; do not change a score without a reason traceable to
  the rationales.
- You MUST NOT look up external peer reviews, ratings, or the decision.
- Finish by calling `submit_reconciled_scores` exactly once with a JSON object whose keys
  are EXACTLY: {keys}. Each value is {{"rationale": <str>, "score": <int 1-10>}}."""

_JUDGE_USER = """\
Here are the seven independent dimension reviews as JSON:

{results_json}

Paper (for reference):
{paper}

Reconcile any inconsistencies across the rationales, then call \
submit_reconciled_scores with all 7 dimensions."""


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
def _dimension_options(server) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=_DIM_MODEL,
        cwd=str(_REPO_ROOT),
        setting_sources=["project"],   # discover .claude/skills/paper-quality-rubric
        skills="all",
        mcp_servers={"score": server},
        allowed_tools=["WebSearch", "WebFetch", "Skill", "TodoWrite", "mcp__score__submit_score"],
        disallowed_tools=sorted(_FS_TOOLS),
        can_use_tool=_gate_tool,
        hooks=_PRETOOL_HOOKS,
        permission_mode="default",
        max_turns=30,
    )


async def _run_dimension(dim: str, paper: str, related: str, sem: asyncio.Semaphore) -> DimensionEntry:
    label = DIMENSION_LABELS[dim]
    async with sem:
        server, holder = _make_dimension_recorder()
        opts = _dimension_options(server)
        opts.system_prompt = _DIMENSION_SYSTEM.format(
            label=label, key=dim, desc=_tools._DIMENSION_DESCRIPTIONS[dim]
        )
        prompt = _DIMENSION_USER.format(label=label, paper=paper, related=related)
        try:
            async for msg in query(prompt=_single_user_msg(prompt), options=opts):
                if isinstance(msg, ResultMessage) and not holder.get("rationale"):
                    # Fallback: try to recover from the agent's final text if it forgot
                    # to call submit_score.
                    _maybe_recover_dim(holder, msg.result or "")
        except Exception as exc:  # noqa: BLE001 — never let one dimension crash the batch
            logger.warning("[scorer] dimension '%s' agent failed: %s", dim, exc)

    score = holder.get("score")
    rationale = holder.get("rationale") or "(agent produced no rationale; defaulted)"
    if score is None:
        logger.warning("[scorer] dimension '%s' produced no score; defaulting to 5", dim)
        score = 5
    logger.info("[scorer] %s -> %d/10", label, score)
    return {"rationale": rationale, "score": int(score)}


def _maybe_recover_dim(holder: dict, text: str) -> None:
    """Best-effort extraction of {rationale, score} from a free-text final message."""
    obj = _extract_json_obj(text)
    if isinstance(obj, dict) and "score" in obj:
        holder.setdefault("rationale", str(obj.get("rationale", "")).strip())
        try:
            holder.setdefault("score", max(1, min(10, int(obj["score"]))))
        except (TypeError, ValueError):
            pass


async def _run_judge(dim_results: dict[str, DimensionEntry], paper: str) -> DimensionScores:
    server, holder = _make_judge_recorder()
    opts = ClaudeAgentOptions(
        model=_JUDGE_MODEL,
        cwd=str(_REPO_ROOT),
        setting_sources=["project"],
        skills="all",
        mcp_servers={"judge": server},
        allowed_tools=["Skill", "TodoWrite", "mcp__judge__submit_reconciled_scores"],
        disallowed_tools=sorted(_FS_TOOLS | {"WebSearch", "WebFetch"}),
        can_use_tool=_gate_tool,
        permission_mode="default",
        max_turns=20,
    )
    opts.system_prompt = _JUDGE_SYSTEM.format(keys=", ".join(DIMENSIONS))
    prompt = _JUDGE_USER.format(
        results_json=json.dumps(dim_results, indent=2, ensure_ascii=False),
        paper=paper,
    )
    last_text = ""
    try:
        async for msg in query(prompt=_single_user_msg(prompt), options=opts):
            if isinstance(msg, ResultMessage):
                last_text = msg.result or last_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("[scorer] judge agent failed (%s); using unreconciled scores", exc)

    raw = holder.get("raw") or last_text
    return _finalize(raw, dim_results)


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


def _finalize(raw: str, dim_results: dict[str, DimensionEntry]) -> DimensionScores:
    """Build the final DimensionScores from the judge output, falling back per-dimension
    to the unreconciled result whenever the judge omitted or malformed an entry."""
    parsed = _extract_json_obj(raw) or {}
    final: dict[str, DimensionEntry] = {}
    for dim in DIMENSIONS:
        fallback = dim_results.get(dim, {"rationale": "(missing)", "score": 5})
        entry = parsed.get(dim) if isinstance(parsed, dict) else None
        if isinstance(entry, dict) and "score" in entry:
            try:
                score = max(1, min(10, int(entry["score"])))
            except (TypeError, ValueError):
                score = int(fallback["score"])
            rationale = str(entry.get("rationale") or fallback["rationale"]).strip()
            final[dim] = {"rationale": rationale, "score": score}
        else:
            final[dim] = {"rationale": fallback["rationale"], "score": int(fallback["score"])}
    return final  # type: ignore[return-value]


async def _score_paper_async(paper_md: str, summaries: dict) -> DimensionScores:
    truncated = paper_md[:_PAPER_CHAR_LIMIT]
    if len(paper_md) > _PAPER_CHAR_LIMIT:
        logger.warning("Paper truncated from %d to %d chars for scoring.", len(paper_md), _PAPER_CHAR_LIMIT)
    related = _build_related_work_context(summaries)

    sem = asyncio.Semaphore(_AGENT_CONCURRENCY)
    results = await asyncio.gather(
        *(_run_dimension(dim, truncated, related, sem) for dim in DIMENSIONS)
    )
    dim_results: dict[str, DimensionEntry] = dict(zip(DIMENSIONS, results))

    logger.info("[scorer] reconciling 7 dimensions with judge (%s)…", _JUDGE_MODEL)
    return await _run_judge(dim_results, truncated)


# ─── Public API (unchanged signature) ──────────────────────────────────────────
def score_paper(paper_md: str, summaries: dict, vendor: Any = None) -> DimensionScores:
    """Score the paper on 7 dimensions via multi-agent orchestration.

    `vendor` is accepted for backward compatibility with the pipeline call site but is
    unused — this stage runs on Claude (Agent SDK) via ANTHROPIC_API_KEY.
    """
    logger.info(
        "[scorer] multi-agent scoring: dims=%s judge=%s concurrency=%d",
        _DIM_MODEL, _JUDGE_MODEL, _AGENT_CONCURRENCY,
    )
    return asyncio.run(_score_paper_async(paper_md, summaries))


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "review_pipeline/test_files/505.md"
    md = Path(path).read_text(encoding="utf-8")
    out = score_paper(md, {})
    print(json.dumps(out, indent=2, ensure_ascii=False))
