"""
Stage 4: Execute Tavily searches and extract arXiv IDs from results.
"""
from __future__ import annotations

import re
import time
import logging
from typing import Optional
from tavily import TavilyClient

from review_pipeline import config

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")


def run_searches(
    queries: list[str],
    max_results_per_query: int = None,
    client: Optional[TavilyClient] = None,
) -> list[dict]:
    """Run each query against Tavily and return deduplicated results.

    Each result dict has keys: url, title, content, score, query.
    """
    max_results_per_query = max_results_per_query or config.MAX_TAVILY_RESULTS
    client = client or TavilyClient(api_key=TAVILY_API_KEY)

    seen_urls: set[str] = set()
    results: list[dict] = []

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(1)  # stay within free-tier rate limits
        try:
            response = client.search(
                query=query,
                max_results=max_results_per_query,
                search_depth="basic",
                include_domains=["arxiv.org"],
            )
            for item in response.get("results", []):
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(
                        {
                            "url": url,
                            "title": item.get("title", ""),
                            "content": item.get("content", ""),
                            "score": item.get("score", 0.0),
                            "query": query,
                        }
                    )
        except Exception as exc:
            logger.warning("Tavily search failed for query %r: %s", query, exc)

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
