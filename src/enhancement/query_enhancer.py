from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EnhancementStats:
    original_length: int
    enhanced_length: int
    expansion_ratio: float
    added_terms: int


class QueryEnhancer:
    """Apply abbreviation expansion, synonym injection, and number normalization."""

    def __init__(self) -> None:
        self.abbreviations: Dict[str, str] = {
            "h1b": "H-1B H1B specialty occupation",
            "h-1b": "H-1B H1B specialty occupation",
            "opt": "OPT Optional Practical Training",
            "cpt": "CPT Curricular Practical Training",
            "lca": "LCA Labor Condition Application",
            "uscis": "USCIS United States Citizenship Immigration Services",
            "dhs": "DHS Department Homeland Security",
            "i-129": "I-129 Form petition nonimmigrant worker",
            "cap-gap": "cap-gap extension F-1 OPT H-1B",
        }
        self.synonyms: Dict[str, List[str]] = {
            "cap": ["cap", "limit", "quota", "ceiling", "maximum"],
            "fee": ["fee", "cost", "price", "charge", "payment"],
            "eligible": ["eligible", "qualify", "requirement", "criteria"],
            "apply": ["apply", "petition", "file", "submit"],
            "process": ["process", "procedure", "steps", "timeline"],
            "change": ["change", "transfer", "switch", "move"],
            "extend": ["extend", "renewal", "extension", "continue"],
        }

    def enhance(self, query: str, mode: str = "balanced") -> str:
        normalized = query.strip()
        if not normalized:
            return normalized

        if mode == "light":
            return self._expand_abbreviations(normalized)

        if mode == "balanced":
            enhanced = self._expand_abbreviations(normalized)
            enhanced = self._add_key_synonyms(enhanced)
            return enhanced

        if mode == "aggressive":
            enhanced = self._expand_abbreviations(normalized)
            enhanced = self._add_all_synonyms(enhanced)
            enhanced = self._normalize_numbers(enhanced)
            return enhanced

        return normalized

    def _expand_abbreviations(self, query: str) -> str:
        lowered = query.lower()
        enhanced = query
        for abbr, expansion in self.abbreviations.items():
            if abbr in lowered:
                enhanced = f"{enhanced} {expansion}"
        return enhanced

    def _add_key_synonyms(self, query: str) -> str:
        lowered = query.lower()
        enhanced = query
        for key_term, synonyms in self.synonyms.items():
            if key_term in lowered:
                enhanced = f"{enhanced} {' '.join(synonyms[:2])}"
        return enhanced

    def _add_all_synonyms(self, query: str) -> str:
        lowered = query.lower()
        enhanced = query
        for key_term, synonyms in self.synonyms.items():
            if key_term in lowered:
                enhanced = f"{enhanced} {' '.join(synonyms)}"
        return enhanced

    def _normalize_numbers(self, query: str) -> str:
        enhanced = query
        dollars = re.findall(r"\$(\d+)", query)
        for amount in dollars:
            enhanced = f"{enhanced} {amount} dollars"

        numbers = re.findall(r"\b(\d{4,})\b", query)
        for num in numbers:
            comma_version = f"{num[:-3]},{num[-3:]}"
            enhanced = f"{enhanced} {comma_version}"
        return enhanced

    def get_enhancement_stats(self, original: str, enhanced: str) -> EnhancementStats:
        orig_tokens = original.split()
        enhanced_tokens = enhanced.split()
        return EnhancementStats(
            original_length=len(orig_tokens),
            enhanced_length=len(enhanced_tokens),
            expansion_ratio=(len(enhanced_tokens) / len(orig_tokens)) if orig_tokens else 1.0,
            added_terms=len(enhanced_tokens) - len(orig_tokens),
        )


__all__ = ["QueryEnhancer", "EnhancementStats"]

