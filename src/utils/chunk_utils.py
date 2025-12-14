from __future__ import annotations

import re
from typing import Dict
from urllib.parse import urlparse


def make_chunk_id(chunk: Dict, fallback_index: int) -> str:
    """
    Build a stable chunk identifier using the source URL and index.
    """

    url = chunk.get("source_url") or ""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "root"
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-") or "chunk"
    chunk_index = chunk.get("chunk_index", fallback_index)
    return f"{slug}-{chunk_index}-{fallback_index}"


__all__ = ["make_chunk_id"]

