"""
Stage 5: Fetch arXiv paper metadata and download PDFs.

Primary engine: the `arxiv` Python library (export.arxiv.org).
Fallback engine: Semantic Scholar Graph API — used when the arXiv endpoint
returns nothing (typically HTTP 429 rate-limit) or skips specific IDs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TypedDict

import arxiv

logger = logging.getLogger(__name__)

_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_FIELDS = "title,abstract,authors,year,publicationDate,externalIds"
_S2_BATCH_SIZE = 500
_S2_RETRIES = 3


class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str      # ISO date string
    categories: list[str]
    pdf_url: str


def _fetch_from_arxiv(arxiv_ids: list[str]) -> dict[str, PaperMetadata]:
    """Primary path: query export.arxiv.org via the `arxiv` library."""
    client = arxiv.Client(
        page_size=min(len(arxiv_ids), 100),
        delay_seconds=3.0,
        num_retries=3,
    )
    search = arxiv.Search(id_list=arxiv_ids)
    metadata: dict[str, PaperMetadata] = {}
    try:
        for paper in client.results(search):
            arxiv_id = paper.get_short_id().split("v")[0]  # strip version
            metadata[arxiv_id] = PaperMetadata(
                arxiv_id=arxiv_id,
                title=paper.title,
                authors=[a.name for a in paper.authors],
                abstract=paper.summary.replace("\n", " ").strip(),
                published=paper.published.date().isoformat() if paper.published else "",
                categories=paper.categories,
                pdf_url=paper.pdf_url or "",
            )
    except Exception as exc:
        logger.warning("arXiv engine failed: %s", exc)
    return metadata


def _fetch_from_semantic_scholar(arxiv_ids: list[str]) -> dict[str, PaperMetadata]:
    """Fallback path: Semantic Scholar `paper/batch` lookup by ARXIV:<id> key."""
    import requests

    metadata: dict[str, PaperMetadata] = {}
    for start in range(0, len(arxiv_ids), _S2_BATCH_SIZE):
        chunk = arxiv_ids[start:start + _S2_BATCH_SIZE]
        body = {"ids": [f"ARXIV:{aid}" for aid in chunk]}

        results = None
        for attempt in range(1, _S2_RETRIES + 1):
            try:
                r = requests.post(
                    _S2_BATCH_URL, params={"fields": _S2_FIELDS}, json=body, timeout=30,
                )
                if r.status_code == 429:
                    sleep_s = 2 ** attempt
                    logger.warning(
                        "Semantic Scholar 429 (attempt %d/%d); sleeping %ds.",
                        attempt, _S2_RETRIES, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                results = r.json()
                break
            except Exception as exc:
                logger.warning(
                    "Semantic Scholar request failed (attempt %d/%d): %s",
                    attempt, _S2_RETRIES, exc,
                )
                if attempt < _S2_RETRIES:
                    time.sleep(2 ** attempt)

        if not results:
            continue

        for aid, item in zip(chunk, results):
            if not item:
                continue
            year = item.get("year")
            published = item.get("publicationDate") or (f"{year}-01-01" if year else "")
            metadata[aid] = PaperMetadata(
                arxiv_id=aid,
                title=item.get("title") or "",
                authors=[a.get("name", "") for a in (item.get("authors") or [])],
                abstract=(item.get("abstract") or "").replace("\n", " ").strip(),
                published=published,
                categories=[],  # S2 doesn't expose arXiv categories
                pdf_url=f"https://arxiv.org/pdf/{aid}.pdf",
            )
    return metadata


def fetch_metadata(arxiv_ids: list[str]) -> dict[str, PaperMetadata]:
    """Fetch metadata for a list of arXiv IDs.

    Tries arXiv first; for any IDs still missing (e.g. arXiv 429 rate-limit),
    falls back to Semantic Scholar's `paper/batch` endpoint with ARXIV:<id>
    keys. Returns a dict keyed by canonical arXiv ID; unresolved IDs are
    logged and skipped.
    """
    if not arxiv_ids:
        return {}

    unique_ids = list(dict.fromkeys(arxiv_ids))  # deduplicate, preserve order
    metadata = _fetch_from_arxiv(unique_ids)

    missing = [aid for aid in unique_ids if aid not in metadata]
    if missing:
        logger.info(
            "arXiv returned %d/%d papers; falling back to Semantic Scholar for %d.",
            len(metadata), len(unique_ids), len(missing),
        )
        metadata.update(_fetch_from_semantic_scholar(missing))

    still_missing = set(unique_ids) - set(metadata.keys())
    if still_missing:
        logger.warning(
            "Could not fetch metadata for %d papers after fallback: %s",
            len(still_missing), still_missing,
        )
    return metadata


def download(arxiv_id: str, dest_dir: Path, pdf_url: str = None) -> Path:
    # client.download(arxiv_id, dest_dir="pdfs", pdf_url)
    """Download the PDF for an arXiv paper.

    Returns the path to the saved PDF. Retries up to 3 times on failure.
    Raises RuntimeError if download fails after all retries.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)


    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path
    
    import requests
    url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"Downloading PDF for {arxiv_id} from {url}...")
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)

    return dest_path

if __name__ == "__main__":
    # Example usage
    test_ids = ["2405.18881", "2307.06350", "2506.16853"]
    metadata = fetch_metadata(test_ids)

    for arxiv_id in metadata.keys():
        try:
            pdf_url = metadata[arxiv_id]['pdf_url']
            pad_path = download(arxiv_id, dest_dir="pdfs", pdf_url=pdf_url)

        except RuntimeError as exc:
            logger.error("Error downloading PDF for %s: %s", arxiv_id, exc)