from __future__ import annotations

from typing import Dict, List

from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.semantic_retriever import RAGRetriever
from src.utils.chunk_utils import make_chunk_id


class HybridRanker:
    """Combine semantic embeddings and BM25 keyword search results."""

    def __init__(
        self,
        semantic_searcher: RAGRetriever,
        bm25_searcher: BM25Searcher,
        rrf_k: int = 60,
    ) -> None:
        self.semantic = semantic_searcher
        self.bm25 = bm25_searcher
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, fusion_method: str = "rrf") -> List[Dict]:
        semantic_results = self._prepare_semantic_results(query)
        bm25_results = self.bm25.search(query, top_k=20)

        if fusion_method == "rrf":
            combined = self._reciprocal_rank_fusion(semantic_results, bm25_results)
        else:
            combined = self._weighted_fusion(semantic_results, bm25_results, semantic_weight=0.6)

        combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined[:top_k]

    def _prepare_semantic_results(self, query: str) -> List[Dict]:
        results = self.semantic.retrieve(query, top_k=20)
        prepared: List[Dict] = []

        for idx, result in enumerate(results):
            chunk = dict(result)
            chunk_id = chunk.get("chunk_id") or make_chunk_id(chunk, idx)
            chunk["chunk_id"] = chunk_id
            chunk["semantic_rank"] = idx + 1
            prepared.append(chunk)

        return prepared

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
    ) -> List[Dict]:
        semantic_ranks = {r["chunk_id"]: r["semantic_rank"] for r in semantic_results}
        bm25_ranks = {r["chunk_id"]: idx + 1 for idx, r in enumerate(bm25_results)}

        all_chunk_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        chunk_map = self._build_chunk_map(semantic_results, bm25_results)

        combined: List[Dict] = []
        for chunk_id in all_chunk_ids:
            sem_rank = semantic_ranks.get(chunk_id, 100)
            bm_rank = bm25_ranks.get(chunk_id, 100)
            rrf_score = (1.0 / (self.rrf_k + sem_rank)) + (1.0 / (self.rrf_k + bm_rank))

            chunk = dict(chunk_map[chunk_id])
            chunk["hybrid_score"] = rrf_score
            chunk["semantic_rank"] = sem_rank if sem_rank < 100 else None
            chunk["bm25_rank"] = bm_rank if bm_rank < 100 else None

            combined.append(chunk)

        return combined

    def _weighted_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        semantic_weight: float = 0.6,
    ) -> List[Dict]:
        semantic_scores = [r["score"] for r in semantic_results] or [1.0]
        bm25_scores = [r["bm25_score"] for r in bm25_results] or [1.0]
        sem_max = max(semantic_scores) or 1.0
        bm_max = max(bm25_scores) or 1.0

        sem_map = {r["chunk_id"]: r["score"] / sem_max for r in semantic_results}
        bm_map = {r["chunk_id"]: r["bm25_score"] / bm_max for r in bm25_results}
        all_chunk_ids = set(sem_map.keys()) | set(bm_map.keys())
        chunk_map = self._build_chunk_map(semantic_results, bm25_results)

        combined: List[Dict] = []
        for chunk_id in all_chunk_ids:
            sem_score = sem_map.get(chunk_id, 0.0)
            bm_score = bm_map.get(chunk_id, 0.0)
            hybrid_score = semantic_weight * sem_score + (1 - semantic_weight) * bm_score

            chunk = dict(chunk_map[chunk_id])
            chunk["hybrid_score"] = hybrid_score
            chunk["semantic_score_norm"] = sem_score
            chunk["bm25_score_norm"] = bm_score
            combined.append(chunk)

        return combined

    def _build_chunk_map(self, semantic_results: List[Dict], bm25_results: List[Dict]) -> Dict[str, Dict]:
        chunk_map: Dict[str, Dict] = {}
        for result in semantic_results + bm25_results:
            chunk_map[result["chunk_id"]] = result
        return chunk_map


__all__ = ["HybridRanker"]

