from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np

from src.embeddings.embedder import load_embedding_model


class RAGRetriever:
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

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.chunks[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query H1B knowledge base via cosine similarity.")
    parser.add_argument("query", type=str, help="Natural language question about H1B topics.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return.")
    args = parser.parse_args()

    retriever = RAGRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    for idx, result in enumerate(results, start=1):
        print(f"\nResult {idx} (score={result['score']:.3f})")
        print(f"Source: {result['source_title']} — {result['source_url']}")
        print(result["text"][:500])


if __name__ == "__main__":
    main()

