from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualizer import Visualizer


def main() -> None:
    print("=" * 70)
    print("H1B KNOWLEDGE BASE — DAY 4 EVALUATION")
    print("=" * 70)

    dataset_path = Path("data/evaluation/test_questions.json")
    if not dataset_path.exists():
        print("📝 Test dataset missing. Generating default question set...")
        from scripts.generate_test_dataset import generate_test_dataset

        generate_test_dataset()

    evaluator = Evaluator(str(dataset_path))

    variants = ["baseline", "semantic_only", "bm25_only", "hybrid", "enhanced"]
    results = evaluator.run_evaluation(variants)

    results_dir = Path("data/evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / f"results_{results['timestamp'].replace(':', '').replace('-', '')}.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\n🗂  Saved raw outputs to {raw_path}")

    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(results)

    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"📊 Metrics written to {metrics_path}")

    visualizer = Visualizer()
    visualizer.generate_all_plots(metrics)

    print("\nSummary:")
    for variant, stats in metrics.get("overall", {}).get("details", {}).items():
        print(f"  - {variant:12s}: {stats['accuracy']*100:5.1f}% accuracy, {stats['avg_time']:.2f}s avg time")
    best = metrics.get("overall", {}).get("best")
    if best:
        print(f"\n🏆 Best variant: {best}")


if __name__ == "__main__":
    main()
