"""
PDF → Markdown conversion via DeepSeek OCR-2 (HuggingFace Transformers backend).

Wraps the existing HF script as a library. The model is loaded lazily and
cached in-process so repeated calls within a pipeline run pay the load cost
only once.
"""
import re
import importlib.util
import sys
import tempfile
import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "DeepSeek-OCR-vllm"


def _load_backend_convert_pdf_to_markdown():
    module_path = _BACKEND_DIR / "run_dpsk_ocr_pdf.py"
    spec = importlib.util.spec_from_file_location("review_pipeline_run_dpsk_ocr_pdf", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OCR backend from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_pdf_to_markdown


_convert_pdf_to_markdown = _load_backend_convert_pdf_to_markdown()


def convert_pdf_to_markdown(
    paper_id: str,
    pdf_path: str | Path,
    ocr_output_path: str | Path,
) -> str:
    return _convert_pdf_to_markdown(paper_id, pdf_path, ocr_output_path)

