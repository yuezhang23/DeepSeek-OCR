#!/usr/bin/env python3
"""
Agentic Academic Paper Reviewer — CLI entry point.

Usage:
    python -m paper_reviewer.main --pdf paper.pdf [--venue ICLR] [--output review.md]
                                   [--markdown existing.mmd] [--force-rerun] [--skip-ocr-related]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ICLR-style peer review for a research paper PDF."
    )
    parser.add_argument("--pdf", help="Path to the input paper PDF.")
    parser.add_argument("--venue", default="ICLR", help="Target venue (default: ICLR).")
    parser.add_argument("--output", default=None, help="Output path for the review markdown.")
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
        "--markdown",
        default=None,
        help="Path to an existing .mmd markdown file for the paper. Skips OCR entirely.",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var or config.py).",
    )
    parser.add_argument(
        "--tavily-api-key",
        default=None,
        help="Tavily API key (defaults to TAVILY_API_KEY env var or config.py).",
    )
    return parser.parse_args()


def run_pipeline(
    pdf_path: Path,
    venue: str = "ICLR",
    output_path: Path | None = None,
    force_rerun: bool = False,
    skip_ocr_related: bool = False,
    markdown_path: Path | None = None,
    anthropic_api_key: str | None = None,
    tavily_api_key: str | None = None,
) -> str:
    """Execute all pipeline stages with per-stage caching. Returns review file path."""
    # ensure review_pipeline is on the path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from review_pipeline import config
    from review_pipeline.cache import StageCache
    from review_pipeline import ocr, query_gen, search, arxiv_client, relevance, summarizer, reviewer

    import anthropic
    from tavily import TavilyClient

    # Use provided API keys or fall back to config
    anthropic_key = anthropic_api_key or config.ANTHROPIC_API_KEY
    tavily_key = tavily_api_key or config.TAVILY_API_KEY

    claude = anthropic.Anthropic(api_key=anthropic_key)
    tavily = TavilyClient(api_key=tavily_key)

    # Initialize cache object before using it
    paper_stem = pdf_path.stem
    cache = StageCache(paper_stem, config.CACHE_DIR)

    # ── Stage 2: OCR ────────────────────────────────────────────────────────────
    if markdown_path is not None:
        print(f"\n[Stage 2/9] Using provided markdown file: {markdown_path}")
        with open(markdown_path, 'r') as f:
            paper_md = f.read()
        cache.save("ocr", paper_md)
    elif force_rerun or not cache.exists("ocr"):
        print("\n[Stage 2/9] Converting PDF to Markdown (DeepSeek OCR-2)...")
        paper_md = ocr.convert_pdf_to_markdown(pdf_path)
        cache.save("ocr", paper_md)
        print(f"  Saved markdown ({len(paper_md):,} chars)")
    else:
        print("\n[Stage 2/9] OCR cache hit — loading existing markdown.")
        paper_md = cache.load("ocr")

    # ── Stage 3: Query Generation ────────────────────────────────────────────────
    if force_rerun or not cache.exists("queries"):
        print("\n[Stage 3/9] Generating arXiv search queries...")
        queries = query_gen.generate_search_queries(paper_md, venue=venue, client=claude)
        cache.save("queries", queries)
        print(f"  Generated {len(queries)} queries")
    else:
        print("\n[Stage 3/9] Query cache hit.")
        queries = cache.load("queries")

    # ── Stage 4: Tavily Search ───────────────────────────────────────────────────
    if force_rerun or not cache.exists("search_results"):
        print(f"\n[Stage 4/9] Searching arXiv via Tavily ({len(queries)} queries)...")
        results = search.run_searches(queries, client=tavily)
        cache.save("search_results", results)
        print(f"  Found {len(results)} unique results")
    else:
        print("\n[Stage 4/9] Search cache hit.")
        results = cache.load("search_results")

    arxiv_ids = search.extract_arxiv_ids(results)
    print(f"  Extracted {len(arxiv_ids)} arXiv IDs")

    # ── Stage 5: ArXiv Metadata ──────────────────────────────────────────────────
    if force_rerun or not cache.exists("arxiv_metadata"):
        print(f"\n[Stage 5/9] Fetching metadata for {len(arxiv_ids)} papers from arXiv...")
        metadata_map = arxiv_client.fetch_metadata(arxiv_ids)
        cache.save("arxiv_metadata", metadata_map)
        print(f"  Fetched metadata for {len(metadata_map)} papers")
    else:
        print("\n[Stage 5/9] arXiv metadata cache hit.")
        metadata_map = cache.load("arxiv_metadata")

    # ── Stage 6: Relevance Ranking ───────────────────────────────────────────────
    if force_rerun or not cache.exists("ranked_papers"):
        print(f"\n[Stage 6/9] Evaluating relevance of {len(metadata_map)} papers...")
        ranked = relevance.evaluate_relevance(
            paper_md, metadata_map, top_k=config.TOP_K_PAPERS, client=claude
        )
        cache.save("ranked_papers", ranked)
        print(f"  Top-{len(ranked)} papers selected")
    else:
        print("\n[Stage 6/9] Relevance ranking cache hit.")
        ranked = cache.load("ranked_papers")

    # ── Stage 7: Summarization Plan ──────────────────────────────────────────────
    if force_rerun or not cache.exists("summarization_plan"):
        print("\n[Stage 7/9] Planning summarization strategy...")
        max_ft = 0 if skip_ocr_related else config.MAX_FULL_TEXT_PAPERS
        plans = summarizer.plan_summarization(paper_md, ranked, max_full_text=max_ft, client=claude)
        cache.save("summarization_plan", plans)
        n_full = sum(1 for p in plans if p["method"] == "full_text")
        print(f"  {n_full} full-text, {len(plans) - n_full} abstract-only")
    else:
        print("\n[Stage 7/9] Summarization plan cache hit.")
        plans = cache.load("summarization_plan")

    # ── Stage 8: Generate Summaries ──────────────────────────────────────────────
    if force_rerun or not cache.exists("summaries"):
        print("\n[Stage 8/9] Generating related work summaries...")
        related_cache = config.CACHE_DIR / paper_stem / "related"
        summaries = summarizer.build_all_summaries(
            paper_md, plans, metadata_map, cache_dir=related_cache, client=claude
        )
        cache.save("summaries", summaries)
        print(f"  Summarized {len(summaries)} papers")
    else:
        print("\n[Stage 8/9] Summary cache hit.")
        summaries = cache.load("summaries")

    # ── Stage 9: Generate Review ─────────────────────────────────────────────────
    print(f"\n[Stage 9/9] Generating {venue} 2026 review...")
    _, review_md = reviewer.generate_review(
        paper_md, summaries, venue=venue, year=2026, client=claude
    )
    cache.save("review", review_md)

    if output_path is None:
        output_path = pdf_path.parent / f"{paper_stem}_review_{venue.lower()}.md"
    else:
        output_path = Path(output_path + "/review_c.md")

    output_path.write_text(review_md, encoding="utf-8")
    print(f"\nReview written to: {output_path}")
    return str(output_path)


def main():
    args = parse_args()
    pdf_path = Path(args.pdf)


    markdown_path = None
    if args.markdown:
        markdown_path = Path(args.markdown)
        # if not markdown_path.exists():
        #     print(f"Error: Markdown file not found: {markdown_path}", file=sys.stderr)
        #     sys.exit(1)
    elif not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output = run_pipeline(
        pdf_path=pdf_path,
        venue=args.venue,
        output_path=Path(args.output) if args.output else None,
        force_rerun=args.force_rerun,
        skip_ocr_related=args.skip_ocr_related,
        markdown_path=Path(args.markdown) if args.markdown else None,
        anthropic_api_key=args.anthropic_api_key,
        tavily_api_key=args.tavily_api_key,
    )
    print(f"\nDone. Review saved to: {output}")


if __name__ == "__main__":
    main()
