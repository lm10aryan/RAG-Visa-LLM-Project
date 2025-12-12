from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

MIN_PARAGRAPH_SIZE = 200
MAX_PARAGRAPH_SIZE = 800
H1B_KEYWORDS = ("h-1b", "h1b")

PROCESSED_DATA_PATH = Path("data/processed/h1b_chunks.json")


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs and text.strip():
        return [text.strip()]
    return paragraphs


def _is_header(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    first_line = lines[0] if lines else paragraph
    words = first_line.split()
    if len(words) <= 10 and any(first_line.lower().startswith(prefix) for prefix in ("h-1b", "cap", "how", "when", "fees")):
        return True
    return bool(len(paragraph) < 140 and paragraph.endswith(":"))


def _merge_short_paragraphs(paragraphs: List[str]) -> List[str]:
    merged: List[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            para = f"{buffer}\n{para}"
            buffer = ""

        if len(para) < MIN_PARAGRAPH_SIZE:
            buffer = para
            continue

        merged.append(para)

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]}\n{buffer}"
        else:
            merged.append(buffer)
    return merged


def _split_long_paragraph(paragraph: str, target_size: int) -> List[str]:
    if len(paragraph) <= MAX_PARAGRAPH_SIZE:
        return [paragraph]

    effective_target = max(target_size, 400)
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= effective_target:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Dict[str, str], target_size: int = 600) -> List[Dict[str, str]]:
    """
    Chunk scraped USCIS pages while preserving semantic boundaries.
    """

    paragraphs = _split_paragraphs(doc.get("content", ""))
    if not paragraphs:
        return []

    prepared: List[str] = []
    pending_header: str | None = None

    for para in paragraphs:
        if _is_header(para):
            pending_header = para
            continue

        if pending_header:
            para = f"{pending_header}\n{para}"
            pending_header = None

        prepared.append(para)

    merged_paragraphs = _merge_short_paragraphs(prepared)

    chunks: List[Dict[str, str]] = []
    chunk_index = 0
    for para in merged_paragraphs:
        for chunk_text in _split_long_paragraph(para, target_size):
            lower_text = chunk_text.lower()
            if not any(keyword in lower_text for keyword in H1B_KEYWORDS):
                continue

            chunks.append(
                {
                    "source_url": doc.get("url", ""),
                    "source_title": doc.get("title", ""),
                    "text": chunk_text.strip(),
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    return chunks


def save_chunks(chunks: List[Dict[str, str]]) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_DATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)


__all__ = ["chunk_document", "save_chunks"]
