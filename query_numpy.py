from __future__ import annotations

import argparse

from src.retrieval.semantic_retriever import RAGRetriever


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
