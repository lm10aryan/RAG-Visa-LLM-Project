from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from src.embeddings.embedder import load_embedding_model
from src.utils.chunk_utils import make_chunk_id


class RAGRetriever:
    """
    Wrapper around the sentence-transformer encoder for cosine-similarity search.
    """

    def __init__(
        self,
        embeddings_path: str = "embeddings.npy",
        metadata_path: str = "chunks_metadata.pkl",
    ) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.model = load_embedding_model()
        self.embeddings = np.load(self.embeddings_path)

        with self.metadata_path.open("rb") as handle:
            self.chunks = pickle.load(handle)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Chunks ({len(self.chunks)}) and embeddings ({self.embeddings.shape[0]}) mismatch."
            )

        self.chunk_ids = []
        for idx, chunk in enumerate(self.chunks):
            chunk_id = chunk.get("chunk_id") or make_chunk_id(chunk, idx)
            chunk["chunk_id"] = chunk_id
            self.chunk_ids.append(chunk_id)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[dict] = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk_id = self.chunk_ids[idx]
            chunk["chunk_id"] = chunk_id
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results


__all__ = ["RAGRetriever"]
