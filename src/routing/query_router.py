from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from difflib import SequenceMatcher


@dataclass
class FactRecord:
    question: str
    fact: str
    category: str


class QueryRouter:
    """Routes queries to Tier 1 (structured facts) or Tier 2 (hybrid RAG)."""

    def __init__(self, facts_db_path: str) -> None:
        self.facts = self._load_facts_db(facts_db_path)
        self.fact_patterns = [
            r"^what is ",
            r"^what are ",
            r"^tell me ",
            r"how much",
            r"how many",
            r"\$\d+",
        ]
        self.fact_keywords = [
            "cap",
            "fee",
            "cost",
            "price",
            "number",
            "amount",
            "date",
            "when",
        ]
        self.complex_indicators = ["am i", "can i", "should i", "how do i", "what if"]

    def route(self, query: str) -> Dict:
        """
        Determine which tier should handle this query.
        """

        normalized = (query or "").strip()
        if not normalized:
            return {
                "tier": 2,
                "confidence": 0.0,
                "reasoning": "Empty query defaults to Tier 2",
            }

        query_lower = normalized.lower()

        if self._has_exact_match(query_lower):
            return {
                "tier": 1,
                "confidence": 1.0,
                "reasoning": "Exact match in facts database",
            }

        if self._matches_direct_fact_pattern(query_lower):
            return {
                "tier": 1,
                "confidence": 0.95,
                "reasoning": "Keyword heuristic strongly indicates fact lookup",
            }

        tier1_score = self._calculate_tier1_score(query_lower)
        if tier1_score >= 0.7:
            return {
                "tier": 1,
                "confidence": round(tier1_score, 2),
                "reasoning": "Fact-based question pattern detected",
            }

        return {
            "tier": 2,
            "confidence": round(max(0.1, 1.0 - tier1_score), 2),
            "reasoning": "Complex or multi-part question requires retrieval",
        }

    def _has_exact_match(self, query: str) -> bool:
        for fact in self.facts:
            similarity = self._string_similarity(query, fact.question.lower())
            if similarity > 0.7:
                return True
        return False

    def _matches_direct_fact_pattern(self, query: str) -> bool:
        required_pairs = [
            ("what", "cap"),
            ("what", "fee"),
            ("what", "cost"),
            ("how much", ""),
            ("how many", ""),
        ]
        for pair in required_pairs:
            first, second = pair
            if first and first in query and (not second or second in query):
                return True
        return False

    def _calculate_tier1_score(self, query: str) -> float:
        score = 0.0
        for pattern in self.fact_patterns:
            if re.search(pattern, query):
                score += 0.3
        for keyword in self.fact_keywords:
            if keyword in query:
                score += 0.2
        for indicator in self.complex_indicators:
            if indicator in query:
                score -= 0.3
        return max(0.0, min(1.0, score))

    def _string_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _load_facts_db(self, db_path: str) -> List[FactRecord]:
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Facts DB not found at {db_path}")

        connection = sqlite3.connect(path)
        cursor = connection.cursor()
        cursor.execute("SELECT question, fact, category FROM facts")
        rows = cursor.fetchall()
        connection.close()

        return [FactRecord(question=row[0], fact=row[1], category=row[2]) for row in rows]


__all__ = ["QueryRouter"]
