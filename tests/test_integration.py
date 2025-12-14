from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.visa_rag import VisaRAG


def main() -> None:
    print("🧪 Testing Visa RAG System\n")

    rag = VisaRAG()
    test_cases = [
        ("What is the H1B cap?", 1, "65,000"),
        ("How much does H1B cost?", 1, "$"),
        ("Am I eligible with CS degree?", 2, "degree"),
        ("Can I change employers?", 2, "employ"),
    ]

    passed = 0
    for question, expected_tier, expected_keyword in test_cases:
        print(f"Q: {question}")
        result = rag.query(question)
        tier_ok = result["tier"] == expected_tier
        keyword_ok = expected_keyword.lower() in result["answer"].lower()

        if tier_ok and keyword_ok:
            print(f"   ✅ Tier {result['tier']}, contains '{expected_keyword}'")
            passed += 1
        else:
            print(f"   ❌ Failed (tier={result['tier']}, keyword={keyword_ok})")

        print(f"   Answer: {result['answer'][:160]}...\n")

    print("=" * 50)
    print(f"Passed: {passed}/{len(test_cases)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
