from __future__ import annotations

from typing import Dict, List, Optional

from src.evaluation.confidence_scorer import ConfidenceScorer
from src.reasoning.model_manager import ModelManager
from src.retrieval.hybrid_ranker import HybridRanker
from src.retrieval.reranker import Reranker


class Tier2Handler:
    """Hybrid retrieval + re-ranking handler."""

    def __init__(
        self,
        hybrid_ranker: HybridRanker,
        reranker: Reranker,
        rerank_depth: int = 10,
        max_context: int = 10,
        rerank_mode: str = "smart",
        model_manager: Optional[ModelManager] = None,
    ) -> None:
        self.hybrid = hybrid_ranker
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.max_context = max_context
        self.confidence_scorer = ConfidenceScorer()
        self.rerank_mode = rerank_mode
        self.model_manager = model_manager or ModelManager()

    def handle(self, question: str, enhanced_query: Optional[str] = None) -> Optional[Dict]:
        retrieval_query = enhanced_query or question
        hybrid_results = self.hybrid.search(retrieval_query, top_k=self.rerank_depth)
        if not hybrid_results:
            return None

        reranked = self.reranker.rerank(
            hybrid_results,
            question,
            mode=self.rerank_mode,
        )
        top_results = reranked[: self.max_context]
        answer = self._compose_answer(question, top_results)

        sources = []
        for chunk in top_results:
            sources.append(
                {
                    "title": chunk.get("source_title", "USCIS"),
                    "url": chunk.get("source_url"),
                    "relevance": round(chunk.get("final_score", chunk.get("hybrid_score", 0.0)), 3),
                    "text": chunk.get("text", ""),
                    "chunk_id": chunk.get("chunk_id"),
                    "scraped_at": chunk.get("scraped_at"),
                }
            )

        confidence_result = self.confidence_scorer.calculate_confidence(
            answer=answer,
            sources=sources,
            tier=2,
        )

        return {
            "tier": 2,
            "method": "tier2_hybrid",
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence_result.overall, 2),
            "confidence_level": confidence_result.level,
            "confidence_reasoning": confidence_result.reasoning,
            "confidence_factors": confidence_result.factors,
        }

    def _compose_answer(self, question: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "Please refer to the cited sources for more details."

        context_parts: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            if text:
                context_parts.append(f"[Source {idx}]\n{text}")

        context = "\n\n".join(context_parts)
        return self.model_manager.generate_response(question, context)


__all__ = ["Tier2Handler"]
