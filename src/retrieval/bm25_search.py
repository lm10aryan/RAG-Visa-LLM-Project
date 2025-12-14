from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

class BM25Searcher:
    """BM25 keyword search over document chunks."""

    def __init__(self, chunks: List[Dict]) -> None:
        self.chunks: List[Dict] = []
        self.tokenized_chunks: List[List[str]] = []

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if not text:
                continue
            normalized_chunk = dict(chunk)
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Chunk at index {idx} missing chunk_id")
            normalized_chunk["chunk_id"] = chunk_id
            tokens = self._tokenize(text)
            self.chunks.append(normalized_chunk)
            self.tokenized_chunks.append(tokens)

        if not self.chunks:
            raise ValueError("No valid chunks provided for BM25 indexing.")

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text with stopword removal for better BM25."""
        STOPWORDS = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'how'
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        # Remove stopwords
        tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens


__all__ = ["BM25Searcher"]
