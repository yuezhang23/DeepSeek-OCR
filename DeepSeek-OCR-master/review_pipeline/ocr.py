"""
PDF → Markdown conversion via DeepSeek OCR-2 (vLLM backend).

The backend module is loaded lazily — importing this module does NOT
initialize the GPU model.  Call build_ocr_engine() once to warm up the
LLM, then pass the returned engine to convert_pdf_to_markdown() so the
model is reused across papers.
"""
import importlib.util
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "DeepSeek-OCR-vllm"

_backend_module = None


def _load_backend():
    global _backend_module
    if _backend_module is not None:
        return _backend_module
    module_path = _BACKEND_DIR / "run_dpsk_ocr_pdf.py"
    spec = importlib.util.spec_from_file_location("review_pipeline_run_dpsk_ocr_pdf", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OCR backend from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _backend_module = module
    return module


def build_ocr_engine():
    """Load the backend and warm up the vLLM engine. Returns (llm, sampling_params).

    Call this once before your processing loop and pass the result to
    convert_pdf_to_markdown() to avoid re-initializing the GPU model.
    """
    backend = _load_backend()
    return backend.build_llm()


def convert_pdf_to_markdown(
    paper_id: str,
    pdf_path,
    ocr_output_path,
    ocr_engine=None,
) -> tuple:
    """Convert a PDF to markdown. ocr_engine is (llm, sampling_params) from build_ocr_engine()."""
    backend = _load_backend()
    if ocr_engine is not None:
        llm, sampling_params = ocr_engine
        return backend.convert_pdf_to_markdown(paper_id, pdf_path, ocr_output_path, llm=llm, sampling_params=sampling_params)
    return backend.convert_pdf_to_markdown(paper_id, pdf_path, ocr_output_path)
