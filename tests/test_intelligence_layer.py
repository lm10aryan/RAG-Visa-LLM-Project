from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.visa_rag import VisaRAG


def test_full_intelligence_stack() -> None:
    rag = VisaRAG(enhancement_mode="balanced", validate_citations=False)
    result = rag.query("What is H1B cap?")
    assert result.get("confidence_level") in {"high", "medium", "low"}
    assert "confidence_factors" in result
    assert "query_enhanced" in result
    assert result["query_enhanced"] is True
