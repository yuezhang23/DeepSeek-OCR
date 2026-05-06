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
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--markdown",
        nargs="+",
        default=None,
        metavar="FILE",
        help="One or more .mmd markdown files. Skips OCR. For a single file you may also pass --output.",
    )
    parser.add_argument(
        "--markdown-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to search for .mmd files (searched recursively). "
            "Mutually exclusive with --markdown."
        ),
    )
    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument("--output", default=None, help="Output path for a single-file run.")
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for batch runs. Review files are written as <stem>.md inside this dir.",
    )
    # ── Pipeline options ─────────────────────────────────────────────────────
    parser.add_argument("--venue", default="ICLR", help="Target venue (default: ICLR).")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of papers to process in parallel (default: PIPELINE_WORKERS from config).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore all cached results and rerun every stage.",
    )
    parser.add_argument(
        "--skip-ocr-related",
        action="store_true",
        help="Use abstracts only for related papers — skips GPU-intensive OCR of related PDFs.",
    )
    parser.add_argument(
        "--score-dimensions",
        action="store_true",
        help=(
            "Run Stage 9-b instead of the default review: score the paper on 7 "
            "quality dimensions and compute a final score via linear regression."
        ),
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
    from review_pipeline.cache import StageCache
    from review_pipeline.clients import PipelineClients
    from review_pipeline import query_gen, search, arxiv_client, relevance, summarizer, reviewer, scorer

    clients = PipelineClients.build(
        anthropic_key=anthropic_api_key,
        deepseek_key=deepseek_api_key,
        tavily_key=tavily_api_key,
    )

    cache_key = paper_id or (markdown_path.stem if markdown_path else pdf_path.stem)
    cache = StageCache(cache_key, config.CACHE_DIR)

    # ── Stage 2: OCR ─────────────────────────────────────────────────────────
    if markdown_path is not None:
        print(f"\n[Stage 2/9] Using provided markdown file: {markdown_path}")
        paper_md = markdown_path.read_text(encoding="utf-8")
        cache.save("ocr", paper_md)
    elif force_rerun or not cache.exists("ocr"):
        from review_pipeline import ocr
        print("\n[Stage 2/9] Converting PDF to Markdown (DeepSeek OCR-2)...")
        ocr_output_path = pdf_path.parent / f"{cache_key}_ocr_output"
        paper_md, dest_path = ocr.convert_pdf_to_markdown(paper_id, pdf_path, ocr_output_path, ocr_engine=ocr_engine)
        cache.save("ocr", paper_md)
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
        print(f"  Generated {len(queries)} queries")
    else:
        print("\n[Stage 3/9] Query cache hit.")
        queries = cache.load("queries")

    # ── Stage 4: Tavily Search ────────────────────────────────────────────────
    if force_rerun or not cache.exists("search_results"):
        print(f"\n[Stage 4/9] Searching arXiv via Tavily ({len(queries)} queries)...")
        results = search.run_searches(queries, client=clients.tavily)
        cache.save("search_results", results)
        print(f"  Found {len(results)} unique results")
    else:
        print("\n[Stage 4/9] Search cache hit.")
        results = cache.load("search_results")

    arxiv_ids = search.extract_arxiv_ids(results)
    print(f"  Extracted {len(arxiv_ids)} arXiv IDs. e.g. {arxiv_ids[0] if arxiv_ids else ''}")

    # ── Stage 5: ArXiv Metadata ───────────────────────────────────────────────
    if force_rerun or not cache.exists("arxiv_metadata"):
        print(f"\n[Stage 5/9] Fetching metadata for {len(arxiv_ids)} papers from arXiv...")
        metadata_map = arxiv_client.fetch_metadata(arxiv_ids)
        cache.save("arxiv_metadata", metadata_map)
        print(f"  Fetched metadata for {len(metadata_map)} papers")
    else:
        print("\n[Stage 5/9] arXiv metadata cache hit.")
        metadata_map = cache.load("arxiv_metadata")

    # ── Stage 6: Relevance Ranking (DeepSeek) ───────────────────────────────────
    if force_rerun or not cache.exists("ranked_papers"):
        print(f"\n[Stage 6/9] Evaluating relevance of {len(metadata_map)} papers...")
        ranked = relevance.evaluate_relevance(
            paper_md, metadata_map, client=clients.deepseek, top_k=config.TOP_K_PAPERS
        )
        cache.save("ranked_papers", ranked)
        print(f"  Top-{len(ranked)} papers selected")
    else:
        print("\n[Stage 6/9] Relevance ranking cache hit.")
        ranked = cache.load("ranked_papers")

    # ── Stage 7: Summarization Plan (DeepSeek) ──────────────────────────────────
    if force_rerun or not cache.exists("summarization_plan"):
        print("\n[Stage 7/9] Planning summarization strategy...")
        max_ft = 0 if skip_ocr_related else config.MAX_FULL_TEXT_PAPERS
        plans = summarizer.plan_summarization(
            paper_md, ranked, client=clients.deepseek, max_full_text=max_ft
        )
        cache.save("summarization_plan", plans)
        n_full = sum(1 for p in plans if p["method"] == "full_text")
        print(f"  {n_full} full-text, {len(plans) - n_full} abstract-only")
    else:
        print("\n[Stage 7/9] Summarization plan cache hit.")
        plans = cache.load("summarization_plan")

    # ── Stage 8: Generate Summaries (DeepSeek) ──────────────────────────────────
    if force_rerun or not cache.exists("summaries"):
        print("\n[Stage 8/9] Generating related work summaries...")
        related_cache = cache.dir / "related"
        summaries = summarizer.build_all_summaries(
            paper_md, plans, metadata_map, cache_dir=related_cache, client=clients.deepseek, ocr_engine=ocr_engine
        )
        cache.save("summaries", summaries)
        print(f"  Summarized {len(summaries)} papers")
    else:
        print("\n[Stage 8/9] Summary cache hit.")
        summaries = cache.load("summaries")

    # ── Stage 9a / 9b: Review or Dimensional Scoring (DeepSeek) ────────────────
    if score_dimensions:
        print("\n[Stage 9/9] Scoring paper on 7 quality dimensions...")
        model_dir = cache.dir / "models"
        scoring_model = scorer.DimensionalScoringModel(model_dir=model_dir)
        scores, final_score = scorer.score_paper(
            paper_md, summaries, client=clients.deepseek, model=scoring_model
        )
        output_md = scorer.format_scores_markdown(
            scores, final_score, model=scoring_model, venue=venue, year=2026
        )
        cache.save("dimension_scores", {"scores": scores, "final_score": final_score})
        for dim in scorer.DIMENSIONS:
            print(f"  {scorer.DIMENSION_LABELS[dim]}: {scores[dim]}/10")
        print(f"  Final score: {final_score:.2f}/10")
    else:
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

    # ── Batch mode ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    papers = []
    for md_path in md_files:
        stem = md_path.stem
        out_path = (output_dir / f"{stem}.md") if output_dir else None
        if out_path and out_path.exists():
            print(f"  Skipping {stem} — output already exists.")
            continue
        papers.append(dict(
            paper_id=stem,
            pdf_path=Path("placeholder.pdf"),
            markdown_path=md_path,
            output_path=str(out_path) if out_path else None,
            **pipeline_kwargs,
        ))

    if not papers:
        print("All papers already have output files. Nothing to do.")
        return

    print(f"\nBatch: {len(papers)} papers, {args.workers or 'default'} workers.\n")
    results = run_pipeline_batch(papers, max_workers=args.workers, **api_kwargs)
    print(f"\nDone. {len(results)} reviews written.")


if __name__ == "__main__":
    main()
