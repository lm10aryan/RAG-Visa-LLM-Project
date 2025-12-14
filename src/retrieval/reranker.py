from __future__ import annotations

from datetime import datetime
import re
from typing import Dict, List, Optional


class Reranker:
    """Re-rank hybrid results using authority and completeness signals."""

    def rerank(self, results: List[Dict], query: str, mode: str = "smart") -> List[Dict]:
        if mode == "simple":
            return self._simple_rerank(results, query)

        query_type = self._classify_query_type(query)
        reranked: List[Dict] = []
        for result in results:
            chunk = dict(result)
            base_score = chunk.get("hybrid_score", chunk.get("score", 0.0))
            bonuses = self._calculate_smart_bonuses(chunk, query, query_type)
            final_score = base_score + sum(bonuses.values())
            chunk.update(bonuses)
            chunk["final_score"] = final_score
            reranked.append(chunk)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        self._apply_contradiction_penalty(reranked)
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    def _simple_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        simple: List[Dict] = []
        for result in results:
            chunk = dict(result)
            base_score = chunk.get("hybrid_score", chunk.get("score", 0.0))
            authority_bonus = self._authority_score(chunk.get("source_url", ""))
            completeness_bonus = self._completeness_score(chunk.get("text", ""))
            answer_bonus = self._answer_quality_score(chunk.get("text", ""), query)
            chunk["final_score"] = base_score + authority_bonus + completeness_bonus + answer_bonus
            chunk["authority_bonus"] = authority_bonus
            chunk["completeness_bonus"] = completeness_bonus
            chunk["answer_bonus"] = answer_bonus
            simple.append(chunk)
        simple.sort(key=lambda x: x["final_score"], reverse=True)
        return simple

    def _calculate_smart_bonuses(self, chunk: Dict, query: str, query_type: str) -> Dict[str, float]:
        authority = self._authority_score(chunk.get("source_url", ""))
        completeness = self._completeness_score(chunk.get("text", ""))
        answer_quality = self._answer_quality_score(chunk.get("text", ""), query)
        recency = self._recency_bonus(chunk)
        query_alignment = self._query_alignment_bonus(chunk.get("text", ""), query_type)

        weights = self._query_weights(query_type)
        return {
            "authority_bonus": authority * weights["authority"],
            "completeness_bonus": completeness * weights["completeness"],
            "answer_bonus": answer_quality * weights["answer"],
            "recency_bonus": recency * weights["recency"],
            "query_alignment_bonus": query_alignment * weights["alignment"],
        }

    def _authority_score(self, url: str) -> float:
        url = url or ""
        if "uscis.gov" in url:
            return 0.15
        if "state.gov" in url:
            return 0.1
        return 0.0

    def _completeness_score(self, text: str) -> float:
        length = len(text or "")
        if length > 600:
            return 0.1
        if length > 400:
            return 0.05
        return 0.0

    def _answer_quality_score(self, text: str, query: str) -> float:
        text_lower = (text or "").lower()
        query_lower = (query or "").lower()
        bonus = 0.0

        answer_patterns = [" is ", " are ", " must ", " requires ", r"\d+"]
        for pattern in answer_patterns:
            if pattern.startswith(r"\d+"):
                if re.search(pattern, text_lower):
                    bonus += 0.02
            elif pattern.strip() in text_lower:
                bonus += 0.02

        for keyword in query_lower.split():
            if keyword and keyword in text_lower:
                bonus += 0.01

        return min(0.05, bonus)

    def _recency_bonus(self, chunk: Dict) -> float:
        timestamp = chunk.get("scraped_at") or chunk.get("last_updated")
        if not timestamp:
            return 0.0
        try:
            scraped_time = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0
        days_old = (datetime.utcnow() - scraped_time).days
        if days_old <= 30:
            return 0.1
        if days_old <= 180:
            return 0.05
        return 0.0

    def _query_alignment_bonus(self, text: str, query_type: str) -> float:
        text_lower = text.lower()
        if query_type == "fact":
            return 0.05 if re.search(r"\d+", text_lower) else 0.0
        if query_type == "process":
            return 0.05 if any(word in text_lower for word in ["step", "process", "procedure"]) else 0.0
        return 0.02 if text_lower else 0.0

    def _query_weights(self, query_type: str) -> Dict[str, float]:
        if query_type == "fact":
            return {"authority": 0.4, "completeness": 0.2, "answer": 0.2, "recency": 0.1, "alignment": 0.1}
        if query_type == "process":
            return {"authority": 0.2, "completeness": 0.4, "answer": 0.2, "recency": 0.1, "alignment": 0.1}
        return {"authority": 0.3, "completeness": 0.25, "answer": 0.2, "recency": 0.15, "alignment": 0.1}

    def _classify_query_type(self, query: str) -> str:
        lower = query.lower()
        if any(lower.startswith(prefix) for prefix in ["what", "how much", "how many"]):
            return "fact"
        if lower.startswith("how do") or "process" in lower or "steps" in lower:
            return "process"
        return "general"

    def _apply_contradiction_penalty(self, chunks: List[Dict]) -> None:
        if len(chunks) < 2:
            return
        numbers = []
        for chunk in chunks[:3]:
            value = self._extract_numeric_value(chunk.get("text", ""))
            if value is not None:
                numbers.append((chunk, value))
        if len(numbers) < 2:
            return
        base_value = numbers[0][1]
        for chunk, value in numbers[1:]:
            if value != base_value:
                penalty = 0.1
                chunk["contradiction_penalty"] = penalty
                chunk["final_score"] -= penalty
                numbers[0][0]["final_score"] -= penalty / 2
                numbers[0][0]["contradiction_penalty"] = penalty / 2

    def _extract_numeric_value(self, text: str) -> Optional[int]:
        match = re.search(r"(\d{2,6})", text or "")
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None


__all__ = ["Reranker"]
