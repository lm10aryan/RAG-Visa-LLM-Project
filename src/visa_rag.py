from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.enhancement.query_enhancer import QueryEnhancer
from src.evaluation.citation_validator import CitationValidator
from src.reasoning.model_manager import ModelManager
from src.reasoning.tier1_handler import Tier1Handler
from src.reasoning.tier2_handler import Tier2Handler
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.hybrid_ranker import HybridRanker
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_retriever import RAGRetriever
from src.utils.chunk_utils import make_chunk_id


class VisaRAG:
    """Two-tier question answering system for H-1B knowledge base."""

    def __init__(
        self,
        facts_db_path: str = "data/structured/h1b_facts.db",
        embeddings_path: str = "embeddings.npy",
        metadata_path: str = "chunks_metadata.pkl",
        enhancement_mode: str = "balanced",
        validate_citations: bool = True,
    ) -> None:
        self.semantic = RAGRetriever(embeddings_path=embeddings_path, metadata_path=metadata_path)

        chunks = self._load_chunks(metadata_path)
        self.bm25 = BM25Searcher(chunks)
        self.hybrid = HybridRanker(self.semantic, self.bm25)
        self.reranker = Reranker()
        self.tier1 = Tier1Handler(facts_db_path)
        self.model_manager = ModelManager()
        self.tier2 = Tier2Handler(self.hybrid, self.reranker, model_manager=self.model_manager)
        self.query_log: List[Dict] = []
        self.query_enhancer = QueryEnhancer()
        self.enhancement_mode = enhancement_mode
        self.validate_citations = validate_citations
        self.citation_validator = CitationValidator()

    def query(
        self, question: str, enhance: bool = True, enhanced_query: Optional[str] = None
    ) -> Dict:
        if enhanced_query is None:
            if enhance and self.enhancement_mode != "off":
                enhanced_query = self.query_enhancer.enhance(
                    question, mode=self.enhancement_mode
                )
            else:
                enhanced_query = question

        tier1_result = self.tier1.handle(question)
        if tier1_result:
            tier1_result["query_enhanced"] = enhanced_query != question
            tier1_result["enhanced_query"] = enhanced_query
            if self.validate_citations and tier1_result.get("sources"):
                validated_sources = self.citation_validator.validate_sources(
                    tier1_result["sources"]
                )
                tier1_result["sources"] = validated_sources
                tier1_result["citation_quality"] = self.citation_validator.get_citation_quality_score(
                    validated_sources
                )

            routing_info = {"tier": 1, "reason": "structured_fact_match"}
            self._log_query(question, tier1_result, routing_info)
            return tier1_result

        tier2_result = self.tier2.handle(question, enhanced_query=enhanced_query) or {
            "tier": 2,
            "method": "tier2_hybrid",
            "answer": "Unable to find a confident answer. Please refine the question.",
            "sources": [],
            "confidence": 0.2,
        }
        tier2_result["method"] = "hybrid_rag"
        tier2_result["query_enhanced"] = enhanced_query != question
        tier2_result["enhanced_query"] = enhanced_query

        if self.validate_citations and tier2_result.get("sources"):
            validated_sources = self.citation_validator.validate_sources(tier2_result["sources"])
            tier2_result["sources"] = validated_sources
            tier2_result["citation_quality"] = self.citation_validator.get_citation_quality_score(
                validated_sources
            )

        routing_info = {"tier": 2, "reason": "tier1_no_match"}
        self._log_query(question, tier2_result, routing_info)
        return tier2_result

    def query_all_methods(self, question: str) -> Dict:
        semantic_only = self.semantic.retrieve(question, top_k=5)
        bm25_only = self.bm25.search(question, top_k=5)
        hybrid = self.hybrid.search(question, top_k=5)
        tier1_result = self.tier1.handle(question)
        tier2_result = self.tier2.handle(question)

        return {
            "semantic_only": semantic_only,
            "bm25_only": bm25_only,
            "hybrid": hybrid,
            "tier1": tier1_result,
            "tier2": tier2_result,
        }

    def _load_chunks(self, metadata_path: str) -> List[Dict]:
        path = Path(metadata_path)
        with path.open("rb") as handle:
            chunks = pickle.load(handle)
        normalized = []
        for idx, chunk in enumerate(chunks):
            if not chunk.get("chunk_id"):
                chunk["chunk_id"] = make_chunk_id(chunk, idx)
            normalized.append(chunk)
        return normalized

    def _log_query(self, question: str, result: Dict, routing: Dict) -> None:
        self.query_log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "routed_tier": routing["tier"],
                "actual_tier": result["tier"],
                "confidence": result.get("confidence"),
                "method": result.get("method"),
                "query_enhanced": result.get("query_enhanced"),
            }
        )


__all__ = ["VisaRAG"]
