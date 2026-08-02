"""
ingestion.py
------------
Document ingestion module. Accepts:
  - PDFs (OCR'd via pytesseract + pdf2image, same approach as the original app)
  - Raw .txt files
  - Domain-specific Hugging Face datasets ("archives") via the `datasets` library

Every ingestion function returns a list of dicts with a consistent shape:
    {"source": <str>, "page_number": <int|None>, "text_content": <str>}

This common shape is what the chunking module expects downstream, regardless
of which ingestion path produced it.
"""

import os
from typing import List, Dict, Optional

import pytesseract
from pdf2image import convert_from_bytes

# --- Configure these for your machine (see README) ---
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD", r"F:\tesseract ocr\tesseract.EXE"
)
POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\poppler\Library\bin")


def ingest_pdf(file_bytes: bytes, source_name: str = "uploaded.pdf", dpi: int = 300) -> List[Dict]:
    """OCR a PDF (as raw bytes) into a list of per-page text records."""
    images = convert_from_bytes(file_bytes, dpi=dpi, fmt="jpeg", poppler_path=POPPLER_PATH)
    records = []
    for i, page_image in enumerate(images):
        if page_image.mode != "RGB":
            page_image = page_image.convert("RGB")
        text = pytesseract.image_to_string(page_image)
        records.append({
            "source": source_name,
            "page_number": i + 1,
            "text_content": text,
        })
    return records


def ingest_txt(file_bytes: bytes, source_name: str = "uploaded.txt", encoding: str = "utf-8") -> List[Dict]:
    """Load a raw text file. Returned as a single 'page' record."""
    text = file_bytes.decode(encoding, errors="replace")
    return [{
        "source": source_name,
        "page_number": None,
        "text_content": text,
    }]


def ingest_txt_path(path: str, encoding: str = "utf-8") -> List[Dict]:
    """Convenience wrapper for loading a .txt file directly from disk."""
    with open(path, "rb") as f:
        return ingest_txt(f.read(), source_name=os.path.basename(path), encoding=encoding)


def ingest_hf_dataset(
    dataset_name: str,
    text_column: str,
    split: str = "train",
    config_name: Optional[str] = None,
    max_records: Optional[int] = None,
) -> List[Dict]:
    """
    Load a domain-specific Hugging Face dataset ("archive") and convert its
    text column into ingestion records.

    Example:
        ingest_hf_dataset("squad", text_column="context", split="train", max_records=500)
    """
    from datasets import load_dataset  # imported lazily; optional dependency

    ds = load_dataset(dataset_name, config_name, split=split) if config_name else load_dataset(dataset_name, split=split)
    if max_records is not None:
        ds = ds.select(range(min(max_records, len(ds))))

    records = []
    for i, row in enumerate(ds):
        text = row.get(text_column, "")
        if not text:
            continue
        records.append({
            "source": f"{dataset_name}:{split}#{i}",
            "page_number": None,
            "text_content": text,
        })
    return records


def ingest(path_or_bytes, source_name: str, kind: str, **kwargs) -> List[Dict]:
    """
    Unified entry point.
    kind: "pdf" | "txt" | "hf"
    For "pdf"/"txt", path_or_bytes should be raw bytes.
    For "hf", path_or_bytes is ignored; pass dataset_name/text_column/split via kwargs.
    """
    if kind == "pdf":
        return ingest_pdf(path_or_bytes, source_name=source_name, **kwargs)
    elif kind == "txt":
        return ingest_txt(path_or_bytes, source_name=source_name, **kwargs)
    elif kind == "hf":
        return ingest_hf_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown ingestion kind: {kind}")
