from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from difflib import SequenceMatcher

CATEGORY_KEYWORDS = {
    "cap": ["cap", "lottery", "quota", "limit", "65", "65000"],
    "registration": ["register", "registration", "account"],
    "timeline": ["date", "when", "window", "deadline", "selection"],
    "fees": ["fee", "cost", "price", "$", "usd"],
    "filing": ["file", "form", "petition", "lca"],
    "employment": ["work", "employment", "years", "validity", "start"],
    "cap-gap": ["cap-gap", "opt", "f-1", "extension"],
}


class Tier1Handler:
    """Structured SQLite fact lookup."""

    def __init__(self, facts_db_path: str) -> None:
        self.db_path = Path(facts_db_path)
        self.facts = self._load_facts()

    def handle(self, question: str) -> Optional[Dict]:
        if not question:
            return None
        best_fact, best_score = self._match_fact(question)
        if not best_fact:
            return None

        return {
            "tier": 1,
            "method": "tier1_db",
            "answer": best_fact["fact"],
            "confidence": min(1.0, best_score + 0.2),
            "confidence_level": "high" if best_score >= 0.5 else "medium",
            "confidence_reasoning": "Matched structured fact entry",
            "confidence_factors": {"match_score": best_score},
            "sources": [
                {
                    "title": best_fact["source_title"],
                    "url": best_fact["source_url"],
                    "text": best_fact["fact"],
                    "relevance": 1.0,
                }
            ],
            "metadata": best_fact.get("metadata", {}),
        }

    def _match_fact(self, question: str) -> tuple[Optional[Dict], float]:
        normalized = question.lower()
        best = None
        best_score = 0.0

        for fact in self.facts:
            score = self._score_fact(normalized, fact)
            if score > best_score:
                best = fact
                best_score = score

        threshold = 0.35
        if best_score < threshold:
            return None, best_score
        return best, best_score

    def _score_fact(self, question: str, fact: Dict) -> float:
        similarity = SequenceMatcher(None, question, fact["question"].lower()).ratio()
        keyword_overlap = self._keyword_overlap(question, fact["question"])
        category_bonus = self._category_bonus(question, fact["category"])
        return min(1.0, 0.6 * similarity + 0.25 * keyword_overlap + category_bonus)

    def _keyword_overlap(self, question: str, fact_question: str) -> float:
        q_tokens = set(question.split())
        f_tokens = set(fact_question.lower().split())
        if not q_tokens or not f_tokens:
            return 0.0
        overlap = len(q_tokens & f_tokens) / len(q_tokens)
        return overlap

    def _category_bonus(self, question: str, category: str) -> float:
        keywords = CATEGORY_KEYWORDS.get(category, [])
        bonus = 0.0
        for keyword in keywords:
            if keyword in question:
                bonus += 0.05
        return min(0.2, bonus)

    def _load_facts(self) -> List[Dict]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Facts DB missing: {self.db_path}")

        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT question, category, fact, metadata, source_url, source_title FROM facts")
        rows = cursor.fetchall()
        connection.close()

        facts = []
        for row in rows:
            metadata = json.loads(row[3]) if row[3] else {}
            facts.append(
                {
                    "question": row[0],
                    "category": row[1],
                    "fact": row[2],
                    "metadata": metadata,
                    "source_url": row[4],
                    "source_title": row[5],
                }
            )
        return facts


__all__ = ["Tier1Handler"]
