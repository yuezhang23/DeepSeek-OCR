"""
PDF → Markdown conversion via DeepSeek OCR-2 (HuggingFace Transformers backend).

Wraps the existing HF script as a library. The model is loaded lazily and
cached in-process so repeated calls within a pipeline run pay the load cost
only once.
"""
import re
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm


from review_pipeline import config

_MODEL_CACHE: dict = {}  # {model_name: (tokenizer, model)}

_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
_PAGE_SPLIT = "\n<--- Page Split --->\n"
_EOS_TOKEN = "<｜end▁of▁sentence｜>"


def _get_model(model_name: str, dtype: str):
    key = (model_name, dtype)
    if key not in _MODEL_CACHE:
        _ensure_hf_on_path()
        from run_dpsk_ocr2_pdf import build_model
        tokenizer, model = build_model(model_name, dtype)
        _MODEL_CACHE[key] = (tokenizer, model)
    return _MODEL_CACHE[key]


def _ensure_hf_on_path():
    hf_dir = str(config.HF_OCR_DIR)
    if hf_dir not in sys.path:
        sys.path.insert(0, hf_dir)


def _clean_page_content(content: str, page_index: int, image_dir: Optional[str]) -> str:
    """Strip EOS tokens and replace bounding-box annotations with markdown."""
    content = content.replace(_EOS_TOKEN, "")

    _ensure_hf_on_path()
    from run_dpsk_ocr2_pdf import re_match

    matches_ref, matches_images, matches_other = re_match(content)

    # Replace image bounding boxes with markdown image links
    for idx, match_image in enumerate(matches_images):
        if image_dir:
            img_path = f"{image_dir}/{page_index}_{idx}.jpg"
        else:
            img_path = f"images/{page_index}_{idx}.jpg"
        content = content.replace(match_image, f"![]({img_path})\n")

    # Remove other annotation tags, clean up whitespace artifacts
    for match_other in matches_other:
        content = content.replace(match_other, "")

    content = (
        content
        .replace("\\coloneqq", ":=")
        .replace("\\eqqcolon", "=:")
        .replace("\n\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
    )
    return content


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    model_name: str = None,
    dtype: str = None,
    dpi: int = 144,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
) -> str:
    """Convert a PDF to a markdown string using DeepSeek OCR-2.

    The model is loaded once and reused across calls in the same process.
    Returns the full concatenated markdown for all pages.
    """
    model_name = model_name or config.OCR_MODEL_NAME
    dtype = dtype or config.OCR_DTYPE
    pdf_path = Path(pdf_path)

    _ensure_hf_on_path()
    from run_dpsk_ocr2_pdf import pdf_to_images_high_quality, get_text_from_infer_result

    tokenizer, model = _get_model(model_name, dtype)

    print(f"Loading PDF: {pdf_path}")
    images = pdf_to_images_high_quality(str(pdf_path), dpi=dpi)
    print(f"  {len(images)} pages found")

    page_contents = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_index, image in enumerate(tqdm(images, desc="OCR pages")):
            page_image_path = os.path.join(tmp_dir, f"page_{page_index}.png")
            image.save(page_image_path)

            page_output_dir = os.path.join(tmp_dir, f"out_{page_index}")
            os.makedirs(page_output_dir, exist_ok=True)

            result = model.infer(
                tokenizer,
                prompt=_PROMPT,
                image_file=page_image_path,
                output_path=page_output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
            )

            raw = get_text_from_infer_result(result)
            cleaned = _clean_page_content(raw, page_index, image_dir=None)
            page_contents.append(cleaned + _PAGE_SPLIT)

    return "".join(page_contents)
