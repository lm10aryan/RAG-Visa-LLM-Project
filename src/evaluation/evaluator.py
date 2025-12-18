from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

from src.reasoning.model_manager import ModelManager
from src.reasoning.tier2_handler import Tier2Handler
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.semantic_retriever import RAGRetriever
from src.utils.chunk_utils import make_chunk_id
from src.visa_rag import VisaRAG


class Evaluator:
    """Evaluate multiple system variants against the shared question set."""

    def __init__(self, dataset_path: str = "data/evaluation/test_questions.json") -> None:
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Test dataset not found: {dataset_path}")

        with dataset_file.open("r", encoding="utf-8") as handle:
            self.test_questions = json.load(handle)

        self.model_manager = ModelManager()
        self.semantic = RAGRetriever()

        with Path("chunks_metadata.pkl").open("rb") as handle:
            self.chunks = pickle.load(handle)
        for idx, chunk in enumerate(self.chunks):
            chunk.setdefault("chunk_id", make_chunk_id(chunk, idx))

        self.bm25 = BM25Searcher(self.chunks)
        self.hybrid_rag = VisaRAG(enhancement_mode="off", validate_citations=False)
        self.enhanced_rag = VisaRAG(enhancement_mode="balanced", validate_citations=True)

        self.variant_map: Dict[str, Callable[[str, Dict], Dict]] = {
            "baseline": self._baseline_variant,
            "semantic_only": self._semantic_only_variant,
            "bm25_only": self._bm25_only_variant,
            "hybrid": self._hybrid_variant,
            "enhanced": self._enhanced_variant,
        }

    def run_evaluation(self, variants: List[str], max_questions: int | None = None) -> Dict:
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "variants": {},
        }

        for name in variants:
            if name not in self.variant_map:
                raise ValueError(f"Unknown variant: {name}")
            print(f"\n▶️  Evaluating variant: {name}")
            results["variants"][name] = self._run_variant(name, self.variant_map[name], max_questions)
        return results

    def _run_variant(self, name: str, runner: Callable[[str, Dict], Dict], max_questions: int | None) -> Dict:
        variant_results = {
            "questions": [],
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "avg_response_time": 0.0,
            "by_category": {},
            "by_difficulty": {},
        }
        total_time = 0.0

        questions = self.test_questions[:max_questions] if max_questions else self.test_questions

        for idx, test_q in enumerate(questions, 1):
            question = test_q["question"]
            ground_truth = test_q["ground_truth"]
            print(f"   [{idx}/{len(self.test_questions)}] {question}")

            start = time.time()
            try:
                response = runner(question, test_q)
            except Exception as exc:  # noqa: BLE001
                print(f"      ❌ Error: {exc}")
                variant_results["questions"].append(
                    {
                        "id": test_q["id"],
                        "question": question,
                        "category": test_q["category"],
                        "difficulty": test_q["difficulty"],
                        "error": str(exc),
                        "correct": False,
                        "ground_truth": ground_truth,
                    }
                )
                variant_results["total"] += 1
                continue

            elapsed = time.time() - start
            total_time += elapsed

            answer_text = response.get("answer", response) if isinstance(response, dict) else response
            preview = (answer_text or "").strip().replace("\n", " ")
            if preview:
                print(f"      ↪︎ Answer preview: {preview[:120]}" + ("…" if len(preview) > 120 else ""))

            is_correct = self._check_correctness(answer_text, ground_truth)

            entry = {
                "id": test_q["id"],
                "question": question,
                "category": test_q["category"],
                "difficulty": test_q["difficulty"],
                "response": answer_text,
                "correct": is_correct,
                "response_time": elapsed,
                "sources": response.get("sources", []) if isinstance(response, dict) else [],
                "confidence_level": response.get("confidence_level"),
                "confidence": response.get("confidence"),
                "citation_quality": response.get("citation_quality"),
                "ground_truth": ground_truth,
            }

            variant_results["questions"].append(entry)
            variant_results["total"] += 1
            if is_correct:
                variant_results["correct"] += 1

            category_bucket = variant_results["by_category"].setdefault(
                test_q["category"],
                {"correct": 0, "total": 0},
            )
            category_bucket["total"] += 1
            if is_correct:
                category_bucket["correct"] += 1

            difficulty_bucket = variant_results["by_difficulty"].setdefault(
                test_q["difficulty"],
                {"correct": 0, "total": 0},
            )
            difficulty_bucket["total"] += 1
            if is_correct:
                difficulty_bucket["correct"] += 1

        if variant_results["total"]:
            variant_results["accuracy"] = variant_results["correct"] / variant_results["total"]
            variant_results["avg_response_time"] = total_time / variant_results["total"]
        for stats in variant_results["by_category"].values():
            if stats["total"]:
                stats["accuracy"] = stats["correct"] / stats["total"]
            else:
                stats["accuracy"] = 0.0
        for stats in variant_results["by_difficulty"].values():
            if stats["total"]:
                stats["accuracy"] = stats["correct"] / stats["total"]
            else:
                stats["accuracy"] = 0.0
        return variant_results

    def _check_correctness(self, response: str, ground_truth: Dict) -> bool:
        text = (response or "").lower()
        for term in ground_truth.get("must_include", []):
            if term.lower() not in text:
                variations = ground_truth.get("acceptable_variations", [])
                if not any(var.lower() in text for var in variations):
                    return False
        for banned in ground_truth.get("must_not_include", []):
            if banned.lower() in text:
                return False
        return True

    def _baseline_variant(self, question: str, _: Dict) -> Dict:
        return {"answer": self.model_manager.generate_response(question, context=""), "sources": []}

    def _semantic_only_variant(self, question: str, _: Dict) -> Dict:
        results = self.semantic.retrieve(question, top_k=3)
        context = self._format_context(results)
        answer = self.model_manager.generate_response(question, context)
        return {"answer": answer, "sources": results, "citation_quality": 0.0}

    def _bm25_only_variant(self, question: str, _: Dict) -> Dict:
        results = self.bm25.search(question, top_k=3)
        context = self._format_context(results)
        answer = self.model_manager.generate_response(question, context)
        return {"answer": answer, "sources": results, "citation_quality": 0.0}

    def _hybrid_variant(self, question: str, _: Dict) -> Dict:
        return self.hybrid_rag.query(question, enhance=False)

    def _enhanced_variant(self, question: str, _: Dict) -> Dict:
        return self.enhanced_rag.query(question, enhance=True)

    def _format_context(self, results: List[Dict]) -> str:
        excerpts = []
        for idx, chunk in enumerate(results, 1):
            text = chunk.get("text", "")
            excerpts.append(f"[Source {idx}]\n{text}")
        return "\n\n".join(excerpts)


__all__ = ["Evaluator"]
