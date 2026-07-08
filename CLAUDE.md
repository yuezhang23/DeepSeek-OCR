# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of DeepSeek's open-source **DeepSeek-OCR** project, extended with a custom
**`review_pipeline`** that turns research-paper PDFs (or pre-converted markdown) into
ICLR-style peer reviews and 7-dimension quality scores. The OCR model is used as one
stage (PDF → markdown); the bulk of the original work lives in the top-level
`review_pipeline/` package.

Two layers coexist:
- **Upstream OCR code** — `DeepSeek-OCR-master/DeepSeek-OCR-vllm/` (vLLM backend, GPU)
  and `DeepSeek-OCR-master/DeepSeek-OCR-hf/` (transformers backend). Largely unmodified
  from upstream; the README at repo root documents install/inference for these.
- **The review pipeline** — `review_pipeline/` at the repo root. This is where almost
  all custom work happens.

## Setup

- Python 3.12. Secrets live in `.env` at repo root (git-ignored), loaded by
  `review_pipeline/config.py` via `python-dotenv`. Required keys: `ANTHROPIC_API_KEY`,
  `TAVILY_API_KEY`, `DEEPSEEK_API_KEY`. Optional overrides (models, cache dir, worker
  counts, `TOP_K_PAPERS`, etc.) are also read from `.env` — see `config.py` for the full list.
- OCR (PDF→markdown) requires a GPU + vLLM and the DeepSeek-OCR model weights. The
  pipeline can skip OCR entirely by feeding it pre-converted `.md`/`.mmd` files, which is
  the common path during development on non-GPU machines.

### Markdown input format (images)
The paper markdown files have the **inserted images stripped out**, but each image's
**original position is preserved as a placeholder** of the form `![](images/2_0.jpg)`. The
number in the image filename matches the figure indexing in the text, and both the figure's
in-text **citation and its explanatory caption/discussion are preserved** in the markdown.
So a placeholder marks where a figure sits, but the actual image bytes are absent — any
stage reasoning over a paper sees the surrounding text and caption, not the image itself.

## Running the pipeline

All commands run from the **repo root** (the package is imported as `review_pipeline`):

```bash
# Single markdown file → one review
python -m review_pipeline.main --markdown review_pipeline/test_files/505.md --output out/505.md

# Batch: every .md found recursively under a dir, parallelized
python -m review_pipeline.main --markdown-dir review_pipeline/data/mds_a --output-dir out/

# Score on 7 quality dimensions instead of writing a prose review
python -m review_pipeline.main --markdown-dir review_pipeline/data/mds_a --output-dir out/ --score-dimensions

# Single PDF (runs OCR — needs GPU)
python -m review_pipeline.main --pdf paper.pdf --output out/paper.md
```

Key flags: `--force-rerun` (ignore all caches), `--skip-ocr-related` (abstracts-only for
related papers, no GPU OCR of references), `--workers N` (batch parallelism), `--venue`,
and `--migrate-cache` (one-shot backfill, see Caching below). API keys can be passed as
`--anthropic-api-key` / `--deepseek-api-key` / `--tavily-api-key` to override `.env`.

Batch mode auto-skips papers whose output already exists, so reruns are resumable. In
`--score-dimensions` batch mode every paper appends to one shared `out/ai_results.json`.

## Pipeline architecture (the big picture)

`main.run_pipeline` orchestrates a fixed sequence of cached stages. Each stage reads the
previous stage's output, calls an LLM/API, and caches its result. The stage modules are
deliberately thin and single-purpose:

1. **Stage 2 — OCR** (`ocr.py` → `run_dpsk_ocr_pdf.py`): PDF → markdown.
   Lazily imported so the GPU model only loads when actually needed; skipped when a
   `--markdown` file is supplied. `build_ocr_engine()` warms the vLLM model once so it's
   reused across papers in a batch. Note: `ocr.py` resolves the backend at
   `<repo-root>/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py` (`__file__`'s `parent.parent`),
   but the upstream code actually lives at `DeepSeek-OCR-master/DeepSeek-OCR-vllm/` — so
   a GPU/OCR run needs that dir reachable at the repo root (move/symlink it).
2. **Stage 3 — Query generation** (`query_gen.py`): DeepSeek generates arXiv search queries.
3. **Stage 4 — Search** (`search.py`): Tavily search → candidate arXiv IDs.
4. **Stage 5 — Metadata** (`arxiv_client.py`): fetch title/abstract/authors from the
   `arxiv` library, falling back to the Semantic Scholar batch API on rate-limits.
5. **Stage 6 — Relevance ranking** (`relevance.py`): DeepSeek scores all candidates in one
   call, keeps top-K (`config.TOP_K_PAPERS`).
6. **Stage 7 — Summarization plan** (`summarizer.plan_summarization`): decide which related
   papers get full-text treatment vs. abstract-only (capped by `MAX_FULL_TEXT_PAPERS`).
   `run_pipeline` then pre-downloads needed PDFs into `related_pdfs/<paper_id>/`.
7. **Stage 8 — Summaries** (`summarizer.build_all_summaries`): per-paper related-work
   summaries (full-text papers are OCR'd here too).
8. **Stage 9a — Review** (`reviewer.py`): final ICLR-style review with rating + confidence
   (label scales defined as `RATING_LABELS` / `CONFIDENCE_LABELS`), OR
   **Stage 9b — Scoring** (`scorer.py`): integer 1–10 on each of 7 `DIMENSIONS`.
   9b is a **multi-agent orchestration on the Claude Agent SDK** (`claude-agent-sdk`): one
   autonomous agent per dimension (each with the `paper-quality-rubric` skill, web
   search, and an in-process MCP `submit_score` tool), run concurrently, then their 7
   `{rationale, score}` results are assembled directly into the merged `DimensionScores`
   (no separate judge/reconciliation agent). A PreToolUse hook blocks any web search/fetch
   touching OpenReview (and filesystem tools are disabled) so agents can't read the paper's
   human reviews; `AskUserQuestion` is auto-resolved when no human answers.
   **Vendor:** the orchestration always runs on the Agent SDK, but the model behind the
   agents is vendor-routed and defaults to **DeepSeek** (not Anthropic). `SCORE_AGENT_VENDOR`
   (default `deepseek`) selects it; `clients.agent_sdk_target()` maps the vendor to the model
   + an env dict merged into `ClaudeAgentOptions.env` that points the SDK's Claude Code CLI at
   that vendor's Anthropic-compatible endpoint (DeepSeek → `…/anthropic`, auth via
   `DEEPSEEK_API_KEY`). `SCORE_AGENT_VENDOR=anthropic` runs Claude natively via
   `ANTHROPIC_API_KEY`; `SCORE_AGENT_BASE_URL`/`SCORE_AGENT_API_KEY` drive any custom
   Anthropic-compatible gateway. `SCORE_DIMENSION_MODEL` optionally overrides the model name.
   Models/concurrency/cost are env tunable (`SCORE_AGENT_CONCURRENCY` default 4). Cost
   controls (the SDK is conversational, so each turn re-bills accumulated context — these
   bound spend): `SCORE_MAX_TURNS` (default 8), `SCORE_MAX_BUDGET_USD` (default 0.20, a hard
   per-agent `--max-budget-usd` ceiling — only enforced on the Anthropic vendor, since the CLI
   needs model pricing), `SCORE_EFFORT` (default `low`), and `SCORE_WEB_DIMENSIONS` (default
   `originality,contextualization_relative_to_prior_work` — only these agents get
   WebSearch/WebFetch; the other 5 score from paper text + related-work summaries only).
   Per-paper total cost is logged as `[scorer] paper scored: $X total`. The skill lives
   at `.claude/skills/paper-quality-rubric/SKILL.md` (discovered via
   `setting_sources=["project"]`).
   **Stage 9-prep — shared section prep** (`section_prep.py`): both scorer backends (`scorer.py`
   on Claude and the alternative `scorer_deepseek.py` on DeepSeek/LangGraph) consume the same two
   DeepSeek-derived artifacts — `paper_sections` (`paper_prep.build_paper_sections` splits the
   markdown + summarizes each section) and `selections` (`section_control` picks raw/summary/omit
   per section per dimension, learning unknown section types into `selection_strategy.md`).
   `section_prep.prepare_paper_sections(paper_md, vendor, cache=…)` builds both and caches each as
   its own stage (`paper_sections` / `section_selections`), so reruns and a switch between the two
   backends reuse the work instead of re-spending DeepSeek calls. `section_control` is re-exported
   from `scorer`/`scorer_deepseek` for backward compatibility but lives in `section_prep` (which
   has no Claude dependency). Both backends expose
   `score_paper(paper_sections, summaries, vendor, selections=None)` — `main.run_pipeline` passes
   the cached `selections`; when omitted, `score_paper` computes them itself.

Cross-cutting modules:
- **`clients.py`** — the *only* place API clients are constructed. `PipelineClients.build()`
  is created once in `main.py` and threaded through every stage; no other module should
  instantiate Anthropic/OpenAI/Tavily clients. `deepseek_chat()` is the shared DeepSeek
  wrapper (DeepSeek uses the OpenAI-compatible SDK with `thinking` mode + retry).
  `get_tool_call()` extracts structured function-call results.
- **`tools.py`** — centralized OpenAI-style function-calling tool schemas for every stage
  (`QUERY_TOOL`, `RELEVANCE_TOOL`, `PLAN_TOOL`, `REVIEW_TOOL`, `SCORE_TOOL`) plus the
  canonical `DIMENSIONS` list. Structured output on the DeepSeek/`LLMVendor` stages goes
  through these schemas. (Note: `scorer.py` no longer uses `SCORE_TOOL` — it moved to the
  Claude Agent SDK with its own MCP `submit_score` tool; `SCORE_TOOL` is still used by the
  `review_scoring.py` eval script. `DIMENSIONS` remains the single source of truth for both
  paths.)
- **`config.py`** — all tunables and keys. Note the worker knobs are tuned to API rate
  limits: `TAVILY_SEARCH_WORKERS`, `SUMMARY_WORKERS` (per-paper), `PIPELINE_WORKERS`
  (papers in parallel — each paper ≈17 DeepSeek calls, so keep small).

### Adding or changing a stage
Add the tool schema to `tools.py`, write a thin stage module that takes a client from
`PipelineClients` and returns plain JSON-serializable data, register a cache filename in
`cache.py`'s `_STAGE_FILES`, and wire it into `run_pipeline` following the existing
`exists()/load()/save()` pattern.

## Caching — important

Two layers, both keyed by paper ID:
- **`StageCache`** (`cache.py`) — primary cache under `config.CACHE_DIR`
  (default `/tmp/paper_reviewer_cache`). One file per stage; `_STAGE_FILES` maps stage →
  filename, `_TEXT_STAGES` (`ocr`, `review`) are stored as raw text, the rest as JSON.
- **`_PersistentStageCache`** (`main.py`) — wraps `StageCache` and mirrors every stage to
  `step_outputs/<paper_id>/<stage>.json`. This survives a `/tmp` wipe (server restart,
  ephemeral storage) and re-hydrates the primary cache on load, so no API calls are
  re-spent. Use `--migrate-cache` to backfill `step_outputs/` from an existing primary
  cache after the wrapper was introduced.

Because of caching, **changing a stage's logic does not invalidate old cached output** —
use `--force-rerun` (or delete the relevant cache/`step_outputs` files) when iterating on
a stage, or you'll keep seeing stale results.

## Evaluation / analysis (separate from the pipeline)

These compare pipeline output against real ICLR reviews and are run standalone, not via
`main.py`. Ground-truth lives in `review_pipeline/metadata_300.json`; results land in
`review_pipeline/out/`.

- **`review_scoring.py`** — maps real ICLR reviewer comments (the `official_values` in
  `metadata_300.json`) onto the 7 scoring dimensions via DeepSeek, producing the same
  `DimensionScores` shape as `scorer.py`. This is how human reviews become comparable to AI scores.
- **`align_review_4_standard.py`** — parses the simulated 4-reviewer files in
  `out/review_4_standard/` and runs them through the same alignment.
- **`compare_distributions.py`** — plots score distributions across deepReview / deepseek /
  human sources (`out/distribution_comparison.png`).
- **`train_classifier.py`** — trains sklearn classifiers mapping the 7 dimension scores to
  ICLR accept/reject decisions.

Data directories under `review_pipeline/`: `data/mds_a/` and `data/mds_r/`
(accepted/rejected paper markdown inputs), `data/related_pdfs/` (downloaded references),
`test_files/` (sample inputs), `step_outputs/` (the persistent cache mirror), `out/`
(reviews, scores, plots). All of `out/`, `data/`, `test_files/`, and `step_outputs/` are
git-ignored.

## Conventions

- Stage modules return plain JSON-serializable data and rely on `TypedDict`s
  (`PaperMetadata`, `RelevanceScore`, `PaperSummary`, `DimensionScores`) for shape — keep
  these in sync when changing payloads.
- The 7 quality dimensions are defined once in `tools.py` (`DIMENSIONS`); `scorer.py`
  provides the human-readable `DIMENSION_LABELS`. Don't redefine the list ad hoc.
- DeepSeek is the workhorse model (cheap, function-calling) and drives generation for every
  stage **except 9b**; Anthropic/Tavily clients exist in `PipelineClients`. Model names come
  from `config`. The one exception is dimensional scoring (Stage 9b, `scorer.py`), which runs
  on the Claude Agent SDK (multi-agent) via `ANTHROPIC_API_KEY` — see Stage 9b above.
