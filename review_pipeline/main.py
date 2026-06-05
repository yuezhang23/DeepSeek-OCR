#!/usr/bin/env python3
"""
Agentic Academic Paper Reviewer — CLI entry point.

Usage:
    python -m review_pipeline.main --pdf paper.pdf [--venue ICLR] [--output review.md]
                                    [--markdown existing.mmd] [--force-rerun]
                                    [--skip-ocr-related] [--score-dimensions]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
SEARCHED = [0] * 10

# Shared lock so concurrent batch workers serialize their JSON appends.
_SCORES_JSON_LOCK = threading.Lock()

# Where to mirror every per-stage output as a plain JSON file. Survives a
# cache-dir wipe so we can resume any paper without re-spending API calls.
_STEP_OUTPUTS_DIR = Path("step_outputs")


class _PersistentStageCache:
    """Wraps ``StageCache`` with a parallel JSON-on-disk backup.

    Each stage is mirrored to ``step_outputs/<paper_id>/<stage>.json`` (or a
    custom root). If the primary cache dir is wiped (server restart, ephemeral
    storage, manual clean-up) but the backup still exists, ``load()`` re-hydrates
    the primary cache from the JSON file — no API calls re-spent.

    Drop-in for the upstream ``StageCache``: the same ``exists()`` / ``load()``
    / ``save()`` / ``dir`` surface is exposed, so per-stage code in
    ``run_pipeline`` is unchanged.
    """

    def __init__(self, paper_id: str, cache_dir, backup_root: Path = _STEP_OUTPUTS_DIR):
        from review_pipeline.cache import StageCache
        self._upstream = StageCache(paper_id, cache_dir)
        self.paper_id = paper_id
        self._backup_dir = Path(backup_root) / paper_id
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self):
        return self._upstream.dir

    def _backup_path(self, stage: str) -> Path:
        return self._backup_dir / f"{stage}.json"

    def exists(self, stage: str) -> bool:
        return self._upstream.exists(stage) or self._backup_path(stage).exists()

    def load(self, stage: str):
        if self._upstream.exists(stage):
            return self._upstream.load(stage)
        bp = self._backup_path(stage)
        if bp.exists():
            data = json.loads(bp.read_text(encoding="utf-8"))
            # Rehydrate the primary cache so subsequent loads in this run
            # hit it directly without re-reading the backup.
            self._upstream.save(stage, data)
            return data
        return self._upstream.load(stage)  # let upstream raise its usual error

    def save(self, stage: str, data) -> None:
        self._upstream.save(stage, data)
        try:
            tmp = self._backup_path(stage).with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._backup_path(stage))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Could not back up stage %s for %s: %s", stage, self.paper_id, exc,
            )


def migrate_cache_to_step_outputs(
    cache_dir: Path | None = None,
    backup_root: Path = _STEP_OUTPUTS_DIR,
    overwrite: bool = False,
) -> dict:
    """One-shot migration: mirror every stage already present in the primary
    StageCache into the ``step_outputs/<paper>/<stage>.json`` backup layout.

    Use this when ``_PersistentStageCache`` is introduced after some papers
    have already been processed under the old setup. Existing backups are
    preserved unless ``overwrite=True``.

    Returns a counter dict: ``{"papers", "stages_migrated", "stages_skipped"}``.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from review_pipeline import config
    from review_pipeline.cache import _STAGE_FILES, _TEXT_STAGES

    if cache_dir is None:
        cache_dir = Path(config.CACHE_DIR)
    else:
        cache_dir = Path(cache_dir)
    backup_root = Path(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)

    counters = {"papers": 0, "stages_migrated": 0, "stages_skipped": 0}

    if not cache_dir.exists():
        logger.warning("Cache dir does not exist: %s", cache_dir)
        return counters

    for paper_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        counters["papers"] += 1
        backup_dir = backup_root / paper_dir.name
        backup_dir.mkdir(parents=True, exist_ok=True)

        for stage, filename in _STAGE_FILES.items():
            src = paper_dir / filename
            if not src.exists():
                continue
            dst = backup_dir / f"{stage}.json"
            if dst.exists() and not overwrite:
                counters["stages_skipped"] += 1
                continue

            try:
                if stage in _TEXT_STAGES:
                    data = src.read_text(encoding="utf-8")
                else:
                    data = json.loads(src.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read %s: %s", src, exc)
                continue

            tmp = dst.with_suffix(".json.tmp")
            try:
                tmp.write_text(
                    json.dumps(data, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(dst)
                counters["stages_migrated"] += 1
            except (TypeError, ValueError) as exc:
                logger.warning("Could not back up %s/%s: %s", paper_dir.name, stage, exc)
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    return counters


def _append_scores_json(json_path: Path, paper_id: str, scores: dict) -> None:
    """Atomically merge ``{paper_id: scores}`` into the JSON file at json_path.

    Thread-safe: holds a process-wide lock around read-modify-write so concurrent
    batch workers don't clobber each other.
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
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.replace(json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ICLR-style peer reviews for one or more research papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Batch examples:
  # All .mmd files found recursively under a directory:
  python -m review_pipeline.main --markdown-dir /content/pdf2md --output-dir /content/reviews

  # Explicit list of files:
  python -m review_pipeline.main --markdown 604.mmd 817.mmd --output-dir /content/reviews

  # Single file (original behaviour):
  python -m review_pipeline.main --markdown paper.mmd --output review.md
""",
    )
    # ── Input ────────────────────────────────────────────────────────────────
    parser.add_argument("--pdf", default=None, help="Path to a single input PDF (used when --markdown is absent).")
    parser.add_argument("--pdf_name", default=None, help="Paper ID for a single PDF run (defaults to PDF stem).")
    parser.add_argument("--markdown",
        nargs="+",
        default=None,
        metavar="FILE",
        help="One or more .mmd markdown files. Skips OCR. For a single file you may also pass --output.",
    )
    parser.add_argument("--markdown-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to search for .mmd files (searched recursively). "
            "Mutually exclusive with --markdown."
        ),
    )
    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument("--output", default=None, help="Output path for a single-file run.")
    parser.add_argument("--output-dir",
        default='out/',
        metavar="DIR",
        help="Output directory for batch runs. Review files are written as <stem>.md inside this dir.",
    )
    # ── Pipeline options ─────────────────────────────────────────────────────
    parser.add_argument("--venue", default="ICLR", help="Target venue (default: ICLR).")
    parser.add_argument("--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of papers to process in parallel (default: PIPELINE_WORKERS from config).",
    )
    parser.add_argument("--force-rerun",
        action="store_true",
        help="Ignore all cached results and rerun every stage.",
    )
    parser.add_argument("--skip-ocr-related",
        action="store_true",
        help="Use abstracts only for related papers — skips GPU-intensive OCR of related PDFs.",
    )
    parser.add_argument("--score-dimensions",
        action="store_true",
        help=(
            "Run Stage 9-b instead of the default review: score the paper on 7 "
            "quality dimensions and compute a final score via linear regression."
        ),
    )
    parser.add_argument("--migrate-cache",
        action="store_true",
        help=(
            "One-shot: mirror every stage already in the primary cache "
            f"into {_STEP_OUTPUTS_DIR}/<paper>/<stage>.json and exit. "
            "Use this after introducing the persistent-cache wrapper to "
            "back-fill papers processed under the old setup."
        ),
    )
    parser.add_argument("--migrate-cache-overwrite",
        action="store_true",
        help="With --migrate-cache: overwrite any existing backup files.",
    )
    # ── API keys ─────────────────────────────────────────────────────────────
    parser.add_argument("--anthropic-api-key", default=None, help="Anthropic API key.")
    parser.add_argument("--deepseek-api-key", default=None, help="DeepSeek API key.")
    parser.add_argument("--tavily-api-key", default=None, help="Tavily API key.")
    return parser.parse_args()

def run_pipeline(
    paper_id: str,
    pdf_path: Path,
    venue: str = "ICLR",
    output_path: Path | None = None,
    force_rerun: bool = True,
    skip_ocr_related: bool = False,
    markdown_path: Path | None = None,
    score_dimensions: bool = False,
    anthropic_api_key: str | None = None,
    deepseek_api_key: str | None = None,
    tavily_api_key: str | None = None,
    ocr_engine=None,
) -> str:
    """Execute all pipeline stages with per-stage caching. Returns output file path."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from review_pipeline import config
    from review_pipeline.clients import PipelineClients
    from review_pipeline import query_gen, search, arxiv_client, relevance, summarizer, reviewer, scorer

    clients = PipelineClients.build(
        anthropic_key=anthropic_api_key,
        deepseek_key=deepseek_api_key,
        tavily_key=tavily_api_key,
    )

    cache_key = paper_id or (markdown_path.stem if markdown_path else pdf_path.stem)
    cache = _PersistentStageCache(cache_key, config.CACHE_DIR)

    # ── Stage 2: Markdwon or OCR ─────────────────────────────────────────────────────────
    if markdown_path is not None:
        print(f"\n[Stage 2/9] Using provided markdown file: {markdown_path}")
        paper_md = markdown_path.read_text(encoding="utf-8")
        cache.save("ocr", paper_md)
    elif force_rerun or not cache.exists("ocr"):
        from review_pipeline import ocr
        print("\n[Stage 2/9] Converting PDF to Markdown (DeepSeek OCR-2)...")
        ocr_output_path = pdf_path.parent / f"{cache_key}_ocr_output"
        paper_md, _ = ocr.convert_pdf_to_markdown(paper_id, pdf_path, ocr_output_path, ocr_engine=ocr_engine)
        cache.save("ocr", paper_md)
        SEARCHED[2] += 1
        print(f"  Saved markdown ({len(paper_md):,} chars)")
    else:
        print("\n[Stage 2/9] OCR cache hit — loading existing markdown.")
        paper_md = cache.load("ocr")

    # ── Stage 3: Query Generation (DeepSeek) ─────────────────────────────────
    if force_rerun or not cache.exists("queries"):
        print("\n[Stage 3/9] Generating arXiv search queries...")
        queries = query_gen.generate_search_queries(
            paper_md, venue=venue, client=clients.deepseek
        )
        cache.save("queries", queries)
        SEARCHED[3] += 1
        print(f"  Generated {len(queries)} queries")
    else:
        print("\n[Stage 3/9] Query cache hit.")
        queries = cache.load("queries")
        print(f"  Loaded {len(queries)} cached queries. e.g. {queries[0] if queries else ''}")

    # ── Stage 4: Tavily Search ────────────────────────────────────────────────
    if force_rerun or not cache.exists("search_results") or not cache.load("search_results"):
        print(f"\n[Stage 4/9] Searching arXiv via Tavily ({len(queries)} queries)...")
        print(f"example qry: {queries[0] if queries else ''}")
        results = search.run_searches(queries, client=clients.tavily)
        cache.save("search_results", results)
        SEARCHED[4] += 1
        print(f"  Found {len(results)} unique results")
    else:
        print("\n[Stage 4/9] Search cache hit.")
        results = cache.load("search_results")

    arxiv_ids = search.extract_arxiv_ids(results)
    if not arxiv_ids:
        logger.warning("No arXiv IDs extracted from search results.")
        # redo stage 4
        results = search.run_searches(queries, client=clients.tavily)
        cache.save("search_results", results)
        print(f"  Found {len(results)} unique results")
        arxiv_ids = search.extract_arxiv_ids(results)
    print(f"  Extracted {len(arxiv_ids)} arXiv IDs. e.g. {arxiv_ids[0] if arxiv_ids else ''}")

    # ── Stage 5: ArXiv Metadata ───────────────────────────────────────────────
    if force_rerun or not cache.exists("arxiv_metadata") or not cache.load("arxiv_metadata"):
        print(f"\n[Stage 5/9] Fetching metadata for {len(arxiv_ids)} papers from arXiv...")
        metadata_map = arxiv_client.fetch_metadata(arxiv_ids)
        cache.save("arxiv_metadata", metadata_map)
        SEARCHED[5] += 1
        print(f"  Fetched metadata for {len(metadata_map)} papers")
    else:
        print("\n[Stage 5/9] arXiv metadata cache hit.")
        metadata_map = cache.load("arxiv_metadata")

    # ── Stage 6: Relevance Ranking (DeepSeek) ───────────────────────────────────
    if force_rerun or not cache.exists("ranked_papers") or not cache.load("ranked_papers"):
        print(f"\n[Stage 6/9] Evaluating relevance of {len(metadata_map)} papers...")
        ranked = relevance.evaluate_relevance(
            paper_md, metadata_map, client=clients.deepseek, top_k=config.TOP_K_PAPERS
        )
        cache.save("ranked_papers", ranked)
        SEARCHED[6] += 1
        print(f"  Top-{len(ranked)} papers selected")
    else:
        print("\n[Stage 6/9] Relevance ranking cache hit.")
        ranked = cache.load("ranked_papers")

    # ── Stage 7: Summarization Plan (DeepSeek) ──────────────────────────────────
    if force_rerun or not cache.exists("summarization_plan") or not cache.load("summarization_plan"):
        print("\n[Stage 7/9] Planning summarization strategy...")
        max_ft = 0 if skip_ocr_related else config.MAX_FULL_TEXT_PAPERS
        plans = summarizer.plan_summarization(
            paper_md, ranked, client=clients.deepseek, max_full_text=max_ft
        )
        cache.save("summarization_plan", plans)
        SEARCHED[7] += 1
        n_full = sum(1 for p in plans if p["method"] == "full_text")
        print(f"  {n_full} full-text, {len(plans) - n_full} abstract-only")
    else:
        print("\n[Stage 7/9] Summarization plan cache hit.")
        plans = cache.load("summarization_plan")
    
    # pre-check if we have downloaded locally the pdfs if plan requires full text, if not, download it and save to local dir
    rank_pad = max(2, len(str(len(plans))))
    for rank, plan in enumerate(plans, 1):
        # if plan["method"] == "full_text":
        arxiv_id = plan["arxiv_id"]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pid = arxiv_id.replace(".", "_")
        filename = f"r{rank:0{rank_pad}d}_{pid}.pdf"
        local_dir = f"related_pdfs/{paper_id}/{filename}"
        if not Path(local_dir).exists():
            import requests, os
            print(f"Downloading rank-{rank} PDF for {arxiv_id} from {pdf_url}...")
            response = requests.get(pdf_url)
            os.makedirs(os.path.dirname(local_dir), exist_ok=True)
            with open(local_dir, "wb") as f:
                f.write(response.content)

    # ── Stage 8: Generate Summaries (DeepSeek + OCR) ──────────────────────────────────
    if force_rerun or not cache.exists("summaries") or not cache.load("summaries"):
        print("\n[Stage 8/9] Generating related work summaries...")
        related_cache = cache.dir / "related"
        summaries = summarizer.build_all_summaries(
            paper_md, plans, metadata_map, cache_dir=related_cache, client=clients.deepseek, ocr_engine=ocr_engine
        )
        cache.save("summaries", summaries)
        SEARCHED[8] += 1
        print(f"  Summarized {len(summaries)} papers")
    else:
        print("\n[Stage 8/9] Summary cache hit.")
        summaries = cache.load("summaries")

    # ── Stage 9a / 9b: Review or Dimensional Scoring (DeepSeek) ────────────────
    if score_dimensions:
        print("\n[Stage 9/9] Scoring paper on 7 quality dimensions...")
        scores = scorer.score_paper(paper_md, summaries, client=clients.deepseek)
        cache.save("dimension_scores", scores)
        for dim in scorer.DIMENSIONS:
            print(f"  {scorer.DIMENSION_LABELS[dim]}: {scores[dim]['score']}/10")
        json_path = Path(output_path) if output_path else Path("out/ai_results.json")
        _append_scores_json(json_path, cache_key, scores)
        return str(json_path)

    print(f"\n[Stage 9/9] Generating {venue} 2026 review...")
    _, output_md = reviewer.generate_review(
        paper_md, summaries, client=clients.deepseek, venue=venue, year=2026
    )
    cache.save("review", output_md)

    # output_path is the file path to save the final markdown review.
    if output_path is not None:
        import shutil
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Handle case where output_path exists as a directory
        if output_path.is_dir():
            shutil.rmtree(output_path)
        output_path.write_text(output_md, encoding="utf-8")
    return str(output_path)


def run_pipeline_batch(
    papers: list[dict],
    max_workers: int = None,
    anthropic_api_key: str | None = None,
    deepseek_api_key: str | None = None,
    tavily_api_key: str | None = None,
    ocr_engine=None,
) -> list[str]:
    """Process multiple papers in parallel. Returns list of output file paths.

    Each entry in `papers` is a dict of kwargs forwarded to run_pipeline()
    (paper_id, pdf_path, markdown_path, output_path, venue, …).

    max_workers defaults to config.PIPELINE_WORKERS. Tune it to stay within
    DeepSeek's rate limit — each paper issues ~17 API calls, so 3 workers
    means ~51 concurrent requests during Stage 8.

    Example Colab usage::

        from review_pipeline.main import run_pipeline_batch
        from pathlib import Path

        papers = [
            dict(paper_id="604", pdf_path=Path("placeholder.pdf"),
                 markdown_path=Path("/content/.../604.md"),
                 output_path="/content/.../604.md")
            for name in names[:200]
            if Path(f".../{name}.md").exists()
        ]
        results = run_pipeline_batch(
            papers,
            anthropic_api_key=ANTHROPIC_API_KEY,
            deepseek_api_key=DEEPSEEK_API_KEY,
            tavily_api_key=TAVILY_API_KEY,
        )
    """
    import concurrent.futures
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from review_pipeline import config

    max_workers = max_workers or config.PIPELINE_WORKERS

    common = dict(
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
        tavily_api_key=tavily_api_key,
        ocr_engine=ocr_engine,
    )
    outputs: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_pipeline, **{**paper, **common}): paper.get("paper_id", "?")
            for paper in papers
        }
        for fut in concurrent.futures.as_completed(futures):
            paper_id = futures[fut]
            try:
                path = fut.result()
                outputs.append(path)
                print(f"  [batch] Done: {paper_id} → {path}")
            except Exception as exc:
                print(f"  [batch] Failed: {paper_id}: {exc}")
                logger.warning("Pipeline failed for %s: %s", paper_id, exc)
    return outputs


def main():
    args = parse_args()

    # One-shot migration path — runs before everything else and exits.
    if args.migrate_cache:
        counts = migrate_cache_to_step_outputs(
            overwrite=args.migrate_cache_overwrite,
        )
        print(
            f"Migration done: {counts['papers']} papers scanned, "
            f"{counts['stages_migrated']} stage files backed up, "
            f"{counts['stages_skipped']} already present (skipped)."
        )
        return

    api_kwargs = dict(
        anthropic_api_key=args.anthropic_api_key,
        deepseek_api_key=args.deepseek_api_key,
        tavily_api_key=args.tavily_api_key,
    )
    pipeline_kwargs = dict(
        venue=args.venue,
        force_rerun=args.force_rerun,
        skip_ocr_related=args.skip_ocr_related,
        score_dimensions=args.score_dimensions,
        **api_kwargs,
    )

    # ── Collect markdown files ────────────────────────────────────────────────
    md_files: list[Path] = []

    if args.markdown_dir and args.markdown:
        print("Error: --markdown-dir and --markdown are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.markdown_dir:
        md_dir = Path(args.markdown_dir)
        if not md_dir.is_dir():
            print(f"Error: --markdown-dir is not a directory: {md_dir}", file=sys.stderr)
            sys.exit(1)
        md_files = sorted(md_dir.rglob("*.md"))
        if not md_files:
            print(f"Error: no .md files found under {md_dir}", file=sys.stderr)
            sys.exit(1)

    elif args.markdown:
        for p in args.markdown:
            path = Path(p)
            if not path.exists():
                print(f"Warning: markdown file not found, skipping: {path}", file=sys.stderr)
            else:
                md_files.append(path)
        if not md_files:
            print("Error: none of the provided --markdown files exist.", file=sys.stderr)
            sys.exit(1)

    # ── Single-file mode (PDF without markdown) ───────────────────────────────
    if not md_files:
        if not args.pdf:
            print("Error: provide --markdown, --markdown-dir, or --pdf.", file=sys.stderr)
            sys.exit(1)
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        output = run_pipeline(
            paper_id=args.pdf_name,
            pdf_path=pdf_path,
            output_path=args.output,
            **pipeline_kwargs,
        )
        print(f"\nDone. Output saved to: {output}")
        return

    # ── Single-file mode (one markdown file + --output) ───────────────────────
    if len(md_files) == 1 and args.output:
        md_path = md_files[0]
        output = run_pipeline(
            paper_id=args.pdf_name or md_path.stem,
            pdf_path=Path("placeholder.pdf"),
            markdown_path=md_path,
            output_path=args.output,
            **pipeline_kwargs,
        )
        print(f"\nDone. Output saved to: {output}")
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # In --score-dimensions mode every paper appends to ONE shared JSON file.
    # Skip papers that are already keys in that file.
    shared_scores_json: Path | None = None
    already_scored: set[str] = set()
    if args.score_dimensions:
        shared_scores_json = (
            output_dir / "ai_results.json" if output_dir else Path("out/ai_results.json")
        )
        if shared_scores_json.exists():
            try:
                already_scored = set(
                    json.loads(shared_scores_json.read_text(encoding="utf-8")).keys()
                )
            except json.JSONDecodeError:
                pass

    papers = []
    for md_path in md_files:
        stem = md_path.stem
        if args.score_dimensions:
            if stem in already_scored:
                print(f"  Skipping {stem} — already in {shared_scores_json.name}.")
                continue
            out_path: str | None = str(shared_scores_json)
        else:
            md_out = (output_dir / f"{stem}.md") if output_dir else None
            if md_out and md_out.exists():
                print(f"  Skipping {stem} — output already exists.")
                continue
            out_path = str(md_out) if md_out else None

        papers.append(dict(
            paper_id=stem,
            pdf_path=Path("placeholder.pdf"),
            markdown_path=md_path,
            output_path=out_path,
            **pipeline_kwargs,
        ))

    if not papers:
        print("All papers already have output files. Nothing to do.")
        return

    print(f"\nBatch: {len(papers)} papers, {args.workers or 'default'} workers.\n")
    results = run_pipeline_batch(papers, max_workers=args.workers, **api_kwargs)
    # print(f"\nDone. {len(results)} reviews written.")
    print(f"missed: search: {SEARCHED}")


if __name__ == "__main__":
    main()
