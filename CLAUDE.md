# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of DeepSeek's open-source **DeepSeek-OCR** project, extended with a custom
**`review_pipeline`** that turns research-paper PDFs (or pre-converted markdown) into
ICLR-style peer reviews and 7-dimension quality scores. The OCR model is used as one
stage (PDF → markdown); the bulk of the original work lives in
`DeepSeek-OCR-master/review_pipeline/`.

Two layers coexist:
- **Upstream OCR code** — `DeepSeek-OCR-master/DeepSeek-OCR-vllm/` (vLLM backend, GPU)
  and `DeepSeek-OCR-master/DeepSeek-OCR-hf/` (transformers backend). Largely unmodified
  from upstream; the README at repo root documents install/inference for these.
- **The review pipeline** — `DeepSeek-OCR-master/review_pipeline/`. This is where almost
  all custom work happens.

## Setup

- Python 3.12. Secrets live in `.env` at repo root (git-ignored), loaded by
  `review_pipeline/config.py` via `python-dotenv`. Required keys: `ANTHROPIC_API_KEY`,
  `TAVILY_API_KEY`, `DEEPSEEK_API_KEY`. Optional overrides (models, cache dir, worker
  counts, `TOP_K_PAPERS`, etc.) are also read from `.env` — see `config.py` for the full list.
- OCR (PDF→markdown) requires a GPU + vLLM and the DeepSeek-OCR model weights. The
  pipeline can skip OCR entirely by feeding it pre-converted `.md`/`.mmd` files, which is
  the common path during development on non-GPU machines.

## Running the pipeline

All commands run from `DeepSeek-OCR-master/` (the package is imported as `review_pipeline`):

```bash
# Single markdown file → one review
python -m review_pipeline.main --markdown review_pipeline/test_files/604.md --output out/604.md

# Batch: every .md found recursively under a dir, parallelized
python -m review_pipeline.main --markdown-dir review_pipeline/mds_a --output-dir out/

# Score on 7 quality dimensions instead of writing a prose review
python -m review_pipeline.main --markdown-dir review_pipeline/mds_a --output-dir out/ --score-dimensions

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

1. **Stage 2 — OCR** (`ocr.py` → `DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py`): PDF → markdown.
   Lazily imported so the GPU model only loads when actually needed; skipped when a
   `--markdown` file is supplied. `build_ocr_engine()` warms the vLLM model once so it's
   reused across papers in a batch.
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

Cross-cutting modules:
- **`clients.py`** — the *only* place API clients are constructed. `PipelineClients.build()`
  is created once in `main.py` and threaded through every stage; no other module should
  instantiate Anthropic/OpenAI/Tavily clients. `deepseek_chat()` is the shared DeepSeek
  wrapper (DeepSeek uses the OpenAI-compatible SDK with `thinking` mode + retry).
  `get_tool_call()` extracts structured function-call results.
- **`tools.py`** — centralized OpenAI-style function-calling tool schemas for every stage
  (`QUERY_TOOL`, `RELEVANCE_TOOL`, `PLAN_TOOL`, `REVIEW_TOOL`, `SCORE_TOOL`) plus the
  canonical `DIMENSIONS` list. Structured output everywhere goes through these schemas.
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

- **`alignment_analysis.py`** — maps real ICLR reviewer comments (the `official_values` in
  `metadata_300.json`) onto the 7 scoring dimensions via DeepSeek, producing the same
  `DimensionScores` shape as `scorer.py`. This is how human reviews become comparable to AI scores.
- **`align_review_4_standard.py`** — parses the simulated 4-reviewer files in
  `out/review_4_standard/` and runs them through the same alignment.
- **`compare_distributions.py`** — plots score distributions across deepReview / deepseek /
  human sources (`out/distribution_comparison.png`).
- **`train_classifier.py`** — trains sklearn classifiers mapping the 7 dimension scores to
  ICLR accept/reject decisions.

Data directories under `review_pipeline/`: `mds_a/` and `mds_r/` (accepted/rejected paper
markdown inputs), `test_files/` (sample inputs), `related_pdfs/` (downloaded references),
`step_outputs/` (the persistent cache mirror), `out/` (reviews, scores, plots).

## Conventions

- Stage modules return plain JSON-serializable data and rely on `TypedDict`s
  (`PaperMetadata`, `RelevanceScore`, `PaperSummary`, `DimensionScores`) for shape — keep
  these in sync when changing payloads.
- The 7 quality dimensions are defined once in `tools.py` (`DIMENSIONS`); `scorer.py`
  provides the human-readable `DIMENSION_LABELS`. Don't redefine the list ad hoc.
- DeepSeek is the workhorse model (cheap, function-calling); Anthropic/Tavily clients exist
  in `PipelineClients` but DeepSeek drives generation/scoring. Model names come from `config`.
