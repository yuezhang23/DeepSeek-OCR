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
            print(f"Fetched metadata for {arxiv_id}:/n{metadata[arxiv_id].keys()}")
    except Exception as exc:
        logger.warning("Error fetching arXiv metadata: %s", exc)

    missing = set(unique_ids) - set(metadata.keys())
    if missing:
        logger.warning("Could not fetch metadata for %d papers: %s", len(missing), missing)
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