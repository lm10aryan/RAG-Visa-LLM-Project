from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConfidenceResult:
    overall: float
    level: str
    reasoning: str
    factors: Dict[str, float]


class ConfidenceScorer:
    """Calculate multi-factor confidence for Tier 2 answers."""

    def calculate_confidence(
        self,
        answer: str,
        sources: List[Dict],
        tier: int = 2,
    ) -> ConfidenceResult:
        factors = {
            "source_agreement": self._check_source_agreement(sources),
            "source_quality": self._check_source_quality(sources),
            "retrieval_quality": self._check_retrieval_quality(sources),
            "answer_completeness": self._check_answer_completeness(answer),
            "tier_bonus": 1.0 if tier == 1 else 0.0,
        }

        overall = min(1.0, sum(factors.values()) / len(factors))
        if overall >= 0.8:
            level = "high"
            reasoning = "Multiple authoritative sources agree"
        elif overall >= 0.6:
            level = "medium"
            reasoning = "Good sources but some uncertainty"
        else:
            level = "low"
            reasoning = "Limited or conflicting information"

        return ConfidenceResult(
            overall=overall,
            level=level,
            reasoning=reasoning,
            factors=factors,
        )

    def _check_source_agreement(self, sources: List[Dict]) -> float:
        if len(sources) < 2:
            return 0.5
        scores = [
            s.get("relevance") or s.get("final_score") or 0.0
            for s in sources[:2]
        ]
        if abs(scores[0] - scores[1]) < 0.1:
            return 0.9
        if abs(scores[0] - scores[1]) < 0.2:
            return 0.7
        return 0.5

    def _check_source_quality(self, sources: List[Dict]) -> float:
        if not sources:
            return 0.0
        official = sum(1 for s in sources if "uscis.gov" in (s.get("url", "").lower()))
        return official / len(sources)

    def _check_retrieval_quality(self, sources: List[Dict]) -> float:
        if not sources:
            return 0.0
        top_score = sources[0].get("relevance") or sources[0].get("final_score") or 0.0
        return min(1.0, max(0.0, top_score))

    def _check_answer_completeness(self, answer: str) -> float:
        if not answer:
            return 0.0
        score = 0.5
        if re.search(r"\d+", answer):
            score += 0.2
        if any(term in answer.lower() for term in ["must", "requires", "is", "are", "will", "can"]):
            score += 0.2
        if not any(word in answer.lower() for word in ["might", "possibly", "perhaps", "unclear"]):
            score += 0.1
        return min(1.0, score)


__all__ = ["ConfidenceScorer", "ConfidenceResult"]

