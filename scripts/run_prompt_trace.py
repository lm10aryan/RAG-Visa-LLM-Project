from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.visa_rag import VisaRAG


def format_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "  (no sources returned)"

    lines = []
    for idx, source in enumerate(sources, 1):
        title = source.get("title", "")
        url = source.get("url", "")
        relevance = source.get("relevance")
        citation = source.get("citation_quality")
        chunk_id = source.get("chunk_id")
        preview = source.get("text", "").strip().replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")

        lines.append(
            f"  [{idx}] {title}\n"
            f"      url: {url}\n"
            f"      relevance: {relevance}\n"
            f"      chunk_id: {chunk_id}\n"
            f"      citation_quality: {citation}\n"
            f"      text: {preview}"
        )
    return "\n".join(lines)


def format_hybrid_results(results: List[Dict[str, Any]], heading: str) -> str:
    if not results:
        return f"{heading}\n  (no results)"

    lines = [heading]
    for idx, result in enumerate(results, 1):
        hybrid_score = float(result.get("hybrid_score", 0.0))
        lines.append(
            "  "
            + " | ".join(
                [
                    f"[{idx}] chunk={result.get('chunk_id')}",
                    f"title={result.get('source_title', 'USCIS')}",
                    f"hybrid={hybrid_score:.3f}",
                    f"semantic_rank={result.get('semantic_rank')}",
                    f"bm25_rank={result.get('bm25_rank')}",
                ]
            )
        )
    return "\n".join(lines)


def format_rerank(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "  (reranker returned no results)"

    lines = []
    for idx, chunk in enumerate(results, 1):
        final_score = float(chunk.get("final_score", 0.0))
        lines.append(
            "  "
            + " | ".join(
                [
                    f"[{idx}] chunk={chunk.get('chunk_id')}",
                    f"final={final_score:.3f}",
                    f"authority={float(chunk.get('authority_bonus', 0.0)):.3f}",
                    f"completeness={float(chunk.get('completeness_bonus', 0.0)):.3f}",
                    f"answer={float(chunk.get('answer_bonus', 0.0)):.3f}",
                    f"recency={float(chunk.get('recency_bonus', 0.0)):.3f}",
                    f"alignment={float(chunk.get('query_alignment_bonus', 0.0)):.3f}",
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace a VisaRAG query with enhanced prompts, routing, and source diagnostics."
        )
    )
    parser.add_argument("question", type=str, help="Natural language H-1B question to test.")
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable query enhancement to test raw question handling.",
    )
    parser.add_argument(
        "--no-citation-validation",
        action="store_true",
        help="Skip citation validation for quicker debugging.",
    )
    args = parser.parse_args()

    rag = VisaRAG(validate_citations=not args.no_citation_validation)

    enhance_query = not args.no_enhance and rag.enhancement_mode != "off"
    enhanced_query = (
        rag.query_enhancer.enhance(args.question, mode=rag.enhancement_mode)
        if enhance_query
        else args.question
    )
    retrieval_query = enhanced_query

    hybrid_results = rag.hybrid.search(retrieval_query, top_k=rag.tier2.rerank_depth)
    reranked_results = rag.reranker.rerank(
        hybrid_results,
        retrieval_query,
        mode=rag.tier2.rerank_mode,
    )

    result = rag.query(
        args.question,
        enhance=not args.no_enhance,
        enhanced_query=enhanced_query,
    )

    print("=" * 70)
    print("VisaRAG Prompt Trace")
    print("=" * 70)
    print(f"Question:        {args.question}")
    print(f"Enhanced query:  {result.get('enhanced_query')}")
    print(f"Was enhanced?:   {result.get('query_enhanced')}")
    print(f"Tier used:       {result.get('tier')} ({result.get('method')})")
    print(f"Confidence:      {result.get('confidence')} ({result.get('confidence_level')})")
    print("\nAnswer:\n")
    print(result.get("answer", "(no answer)"))

    print("\nRetrieval steps:")
    print(format_hybrid_results(hybrid_results[:5], "Hybrid fusion (top 5):"))
    print("\nReranker scoring (top 5):")
    print(format_rerank(reranked_results[:5]))

    print("\nSources:")
    print(format_sources(result.get("sources", [])))


if __name__ == "__main__":
    main()
