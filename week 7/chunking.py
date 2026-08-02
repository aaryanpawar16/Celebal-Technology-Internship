"""
chunking.py
-----------
Clean chunking methodology for splitting unstructured text into manageable,
overlapping chunks. Supports:
  - "char"     : fixed-size character windows with overlap (fast, simple)
  - "sentence" : sentence-aware chunking that packs whole sentences up to a
                 target size, so chunks don't cut sentences in half

Output: list of dicts, each with a unique id, source, page number, and the
chunk text -- ready to be embedded.
"""

import re
from typing import List, Dict, Optional


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no external NLP dependency required)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital/number,
    # while keeping the punctuation attached to the sentence.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_char(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Fixed-size character chunking with overlap (same approach as the original app)."""
    if not text:
        return []
    chunks, start, text_len = [], 0, len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end >= text_len:
            break
    return chunks


def chunk_sentence_aware(text: str, chunk_size: int = 1000, overlap_sentences: int = 1) -> List[str]:
    """
    Packs whole sentences into chunks up to ~chunk_size characters.
    Keeps `overlap_sentences` sentences of overlap between consecutive chunks
    so context isn't lost at chunk boundaries.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            # carry the last N sentences forward for overlap
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(
    records: List[Dict],
    chunk_size: int = 1000,
    overlap: int = 200,
    method: str = "sentence",
) -> List[Dict]:
    """
    Turn ingestion records (from ingestion.py) into a flat list of chunks.
    method: "char" or "sentence"
    """
    chunks = []
    cid = 0
    for rec in records:
        source = rec.get("source")
        page_no = rec.get("page_number")
        text = rec.get("text_content") or ""

        if method == "sentence":
            pieces = chunk_sentence_aware(text, chunk_size=chunk_size, overlap_sentences=max(1, overlap // 200))
        else:
            pieces = chunk_char(text, chunk_size=chunk_size, overlap=overlap)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            cid += 1
            chunks.append({
                "id": cid,
                "source": source,
                "page_number": page_no,
                "chunk_text": piece,
            })
    return chunks
