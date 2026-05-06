"""
Stage 4: Execute Tavily searches and extract arXiv IDs from results.
"""
from __future__ import annotations

import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tavily import TavilyClient

from review_pipeline import config

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")


def _search_one(
    query: str,
    max_results: int,
    client: TavilyClient,
    rate_semaphore: threading.Semaphore,
) -> list[dict]:
    with rate_semaphore:
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_domains=["arxiv.org"],
            )
            time.sleep(0.25)  # ~4 req/s per worker slot; free tier is 1 req/s total
        except Exception as exc:
            logger.warning("Tavily search failed for query %r: %s", query, exc)
            return []

    items = []
    for item in response.get("results", []):
        url = item.get("url", "")
        if url:
            items.append({
                "url": url,
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "query": query,
            })
    return items


def run_searches(
    queries: list[str],
    max_results_per_query: int = None,
    client: Optional[TavilyClient] = None,
    api_key: Optional[str] = None,
    max_workers: int = None,
) -> list[dict]:
    """Run each query against Tavily in parallel and return deduplicated results.

    max_workers controls concurrency; defaults to config.TAVILY_SEARCH_WORKERS.
    A shared semaphore ensures we don't exceed Tavily's rate limit.
    """
    max_results_per_query = max_results_per_query or config.MAX_TAVILY_RESULTS
    max_workers = max_workers or config.TAVILY_SEARCH_WORKERS
    client = client or TavilyClient(api_key=api_key or config.TAVILY_API_KEY)

    # One slot = one in-flight request at a time to respect free-tier 1 req/s
    rate_semaphore = threading.Semaphore(max_workers)

    seen_urls: set[str] = set()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_search_one, q, max_results_per_query, client, rate_semaphore): q
            for q in queries
        }
        for fut in as_completed(futures):
            for item in fut.result():
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    results.append(item)

    return results


def extract_arxiv_ids(results: list[dict]) -> list[str]:
    """Parse arXiv IDs from result URLs, stripping version suffixes.

    Returns a deduplicated list of canonical IDs like '2401.00001'.
    """
    seen: set[str] = set()
    ids: list[str] = []

    for item in results:
        url = item.get("url", "")
        m = _ARXIV_ID_RE.search(url)
        if m:
            raw_id = m.group(1)
            canonical = re.sub(r"v\d+$", "", raw_id)  # strip version
            if canonical not in seen:
                seen.add(canonical)
                ids.append(canonical)

    return ids
