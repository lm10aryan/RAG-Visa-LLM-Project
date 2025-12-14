from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.visa_rag import VisaRAG


rag = VisaRAG()


def test_tier1_routing() -> None:
    fact_questions = [
        "What is the H1B cap?",
        "How much does H1B cost?",
        "What is the ACWIA fee?",
    ]
    for question in fact_questions:
        result = rag.query(question)
        assert result["tier"] == 1, f"Expected Tier 1 for {question}"
        assert result["confidence"] >= 0.7


def test_tier2_routing() -> None:
    complex_questions = [
        "Am I eligible for H1B with a CS degree?",
        "Can I change employers on H1B?",
        "How do I apply for H1B status?",
    ]
    for question in complex_questions:
        result = rag.query(question)
        assert result["tier"] == 2, f"Expected Tier 2 for {question}"
        assert len(result["sources"]) > 0


def test_hybrid_retrieval_mentions_cap() -> None:
    diagnostics = rag.query_all_methods("What is the H1B cap?")
    hybrid_chunks = diagnostics["hybrid"]
    assert hybrid_chunks, "Hybrid search returned no results"
    top_text = hybrid_chunks[0]["text"].lower()
    assert "65,000" in top_text or "65000" in top_text
