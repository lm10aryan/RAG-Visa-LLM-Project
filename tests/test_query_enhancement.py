from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.enhancement.query_enhancer import QueryEnhancer


def test_abbreviation_expansion() -> None:
    enhancer = QueryEnhancer()
    query = "What is H1B cap?"
    enhanced = enhancer.enhance(query, mode="light")
    assert "H-1B" in enhanced
    assert "specialty" in enhanced


def test_number_normalization_aggressive() -> None:
    enhancer = QueryEnhancer()
    query = "Is the fee $460?"
    enhanced = enhancer.enhance(query, mode="aggressive")
    assert "460 dollars" in enhanced
    assert "$460" in enhanced
