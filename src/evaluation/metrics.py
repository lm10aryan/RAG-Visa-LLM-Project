from __future__ import annotations

from statistics import mean
from typing import Dict, List


class MetricsCalculator:
    """Compute accuracy, calibration, and retrieval metrics for each variant."""

    def calculate_all_metrics(self, results: Dict) -> Dict:
        metrics = {"overall": {}, "by_variant": {}, "ablation": {}}
        variants = results.get("variants", {})
        for name, data in variants.items():
            metrics["by_variant"][name] = self._calculate_variant_metrics(data)
        if variants:
            metrics["overall"] = self._overall_summary(metrics["by_variant"])
            metrics["ablation"] = self._ablation_contributions(metrics["overall"])
        return metrics

    def _calculate_variant_metrics(self, data: Dict) -> Dict:
        questions = data.get("questions", [])
        metrics = {
            "accuracy": {
                "overall": data.get("accuracy", 0.0),
                "by_category": {
                    cat: stats.get("accuracy", 0.0)
                    for cat, stats in data.get("by_category", {}).items()
                },
            },
            "response_time": {
                "avg": data.get("avg_response_time", 0.0),
                "total": sum(q.get("response_time", 0.0) for q in questions),
            },
            "confidence_calibration": self._confidence_metrics(questions),
            "citation_quality": self._citation_metrics(questions),
            "retrieval_precision": self._retrieval_precision(questions),
        }
        return metrics

    def _confidence_metrics(self, questions: List[Dict]) -> Dict:
        buckets = {"high": {"correct": 0, "total": 0}, "medium": {"correct": 0, "total": 0}, "low": {"correct": 0, "total": 0}}
        for entry in questions:
            level = (entry.get("confidence_level") or "").lower()
            if level in buckets:
                buckets[level]["total"] += 1
                if entry.get("correct"):
                    buckets[level]["correct"] += 1
        metrics = {}
        for level, stats in buckets.items():
            if stats["total"]:
                metrics[level] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "count": stats["total"],
                }
        return metrics

    def _citation_metrics(self, questions: List[Dict]) -> Dict:
        sources_present = [bool(entry.get("sources")) for entry in questions]
        coverage = sum(sources_present) / len(sources_present) if sources_present else 0.0
        quality_scores = [entry.get("citation_quality") for entry in questions if entry.get("citation_quality") is not None]
        quality = mean(quality_scores) if quality_scores else 0.0
        return {"coverage": coverage, "quality": quality}

    def _retrieval_precision(self, questions: List[Dict]) -> Dict:
        hits_top1 = 0
        hits_top3 = 0
        total_with_sources = 0

        for entry in questions:
            sources = entry.get("sources") or []
            if not sources:
                continue

            total_with_sources += 1
            ground_truth = entry.get("ground_truth", {})
            must_terms = [term.lower() for term in ground_truth.get("must_include", [])]

            def contains_terms(text: str) -> bool:
                lowered = text.lower()
                return all(term in lowered for term in must_terms) if must_terms else False

            if contains_terms(sources[0].get("text", "")):
                hits_top1 += 1
            if any(contains_terms(source.get("text", "")) for source in sources[:3]):
                hits_top3 += 1

        precision = {
            "p_at_1": (hits_top1 / total_with_sources) if total_with_sources else 0.0,
            "p_at_3": (hits_top3 / total_with_sources) if total_with_sources else 0.0,
            "evaluated": total_with_sources,
        }
        return precision

    def _overall_summary(self, by_variant: Dict[str, Dict]) -> Dict:
        summary = {"details": {}, "best": None}
        best_acc = -1.0
        for name, metrics in by_variant.items():
            accuracy = metrics["accuracy"]["overall"]
            summary["details"][name] = {
                "accuracy": accuracy,
                "avg_time": metrics["response_time"]["avg"],
            }
            if accuracy > best_acc:
                best_acc = accuracy
                summary["best"] = name
        return summary

    def _ablation_contributions(self, overall: Dict) -> Dict:
        baseline = overall["details"].get("baseline", {}).get("accuracy", 0.0)
        semantic = overall["details"].get("semantic_only", {}).get("accuracy", baseline)
        hybrid = overall["details"].get("hybrid", {}).get("accuracy", semantic)
        enhanced = overall["details"].get("enhanced", {}).get("accuracy", hybrid)
        return {
            "semantic_contribution": max(0.0, semantic - baseline),
            "hybrid_contribution": max(0.0, hybrid - semantic),
            "enhancement_contribution": max(0.0, enhanced - hybrid),
        }


__all__ = ["MetricsCalculator"]

