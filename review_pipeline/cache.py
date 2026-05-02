import json
from pathlib import Path
from typing import Any


_STAGE_FILES = {
    "ocr": "paper.mmd",
    "queries": "queries.json",
    "search_results": "search_results.json",
    "arxiv_metadata": "arxiv_metadata.json",
    "ranked_papers": "ranked_papers.json",
    "summarization_plan": "summarization_plan.json",
    "summaries": "summaries.json",
    "review": "review.md",
}

_TEXT_STAGES = {"ocr", "review"}


class StageCache:
    def __init__(self, paper_stem: str, base_dir: Path):
        self.dir = base_dir / paper_stem
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        return self.dir / _STAGE_FILES[stage]

    def exists(self, stage: str) -> bool:
        return self.path(stage).exists()

    def load(self, stage: str) -> Any:
        p = self.path(stage)
        if stage in _TEXT_STAGES:
            return p.read_text(encoding="utf-8")
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, stage: str, data: Any) -> None:
        p = self.path(stage)
        if stage in _TEXT_STAGES:
            p.write_text(data, encoding="utf-8")
        else:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
