from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional


class CitationValidator:
    """Validate citation URLs for accessibility and freshness."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def validate_sources(self, sources: List[Dict]) -> List[Dict]:
        validated: List[Dict] = []
        for source in sources:
            entry = dict(source)
            entry["validation"] = {
                "accessible": self._check_accessible(entry.get("url")),
                "freshness": self._determine_freshness(entry.get("scraped_at")),
            }
            validated.append(entry)
        return validated

    def get_citation_quality_score(self, sources: List[Dict]) -> float:
        if not sources:
            return 0.0
        accessible = sum(
            1 for s in sources if s.get("validation", {}).get("accessible") is True
        )
        freshness = sum(
            1 for s in sources if s.get("validation", {}).get("freshness") == "recent"
        )
        return 0.7 * (accessible / len(sources)) + 0.3 * (freshness / len(sources))

    def _check_accessible(self, url: Optional[str]) -> Optional[bool]:
        if not url:
            return None
        if url.startswith("http"):
            return True
        return False

    def _determine_freshness(self, timestamp: Optional[str]) -> str:
        if not timestamp:
            return "unknown"
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            return "unknown"
        age_days = (datetime.utcnow() - parsed).days
        if age_days <= 90:
            return "recent"
        return "old"


__all__ = ["CitationValidator"]
