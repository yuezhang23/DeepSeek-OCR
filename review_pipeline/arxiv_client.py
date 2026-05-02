"""
Stage 5: Fetch arXiv paper metadata and download PDFs.

Uses the `arxiv` Python library which handles rate-limiting automatically.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TypedDict

import arxiv

logger = logging.getLogger(__name__)


class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str      # ISO date string
    categories: list[str]
    pdf_url: str


def fetch_metadata(arxiv_ids: list[str]) -> dict[str, PaperMetadata]:
    """Fetch metadata for a list of arXiv IDs.

    Returns a dict keyed by canonical arXiv ID. Papers that cannot be
    fetched are logged and skipped.
    """
    if not arxiv_ids:
        return {}

    unique_ids = list(dict.fromkeys(arxiv_ids))  # deduplicate, preserve order

    client = arxiv.Client(
        page_size=min(len(unique_ids), 100),
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(id_list=unique_ids)
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
        logger.warning("Error fetching arXiv metadata: %s", exc)

    missing = set(unique_ids) - set(metadata.keys())
    if missing:
        logger.warning("Could not fetch metadata for %d papers: %s", len(missing), missing)

    return metadata


def download_pdf(arxiv_id: str, dest_dir: Path, filename: str = None) -> Path:
    """Download the PDF for an arXiv paper.

    Returns the path to the saved PDF. Retries up to 3 times on failure.
    Raises RuntimeError if download fails after all retries.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = filename or f"{arxiv_id.replace('/', '_')}.pdf"
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    client = arxiv.Client(num_retries=3, delay_seconds=3.0)
    search = arxiv.Search(id_list=[arxiv_id])

    for attempt in range(3):
        try:
            paper = next(client.results(search))
            paper.download_pdf(dirpath=str(dest_dir), filename=filename)
            return dest_path
        except StopIteration:
            raise RuntimeError(f"arXiv paper not found: {arxiv_id}")
        except Exception as exc:
            logger.warning("PDF download attempt %d failed for %s: %s", attempt + 1, arxiv_id, exc)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"Failed to download PDF for {arxiv_id} after 3 attempts.")
