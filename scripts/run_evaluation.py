from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.metrics import MetricsCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or reprocess the H1B evaluation pipeline.")
    parser.add_argument(
        "--results-file",
        type=str,
        help="Use an existing results_*.json file to regenerate metrics and plots (skips running variants).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Limit the number of questions to evaluate per variant when running fresh results.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip visualization generation (useful in limited environments).",
    )
    return parser.parse_args()


def update_manifest(
    run_id: str,
    results_path: Path,
    metrics_path: Path,
    variants,
    mode: str,
    dataset_path: Path | None,
    question_count: int,
    max_questions: int | None,
) -> None:
    manifest_path = Path("data/evaluation/run_history.json")
    manifest = {"runs": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    entry = {
        "id": run_id,
        "results_file": str(results_path),
        "metrics_file": str(metrics_path),
        "variants": variants,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "question_count": question_count,
        "max_questions": max_questions,
        "last_updated": datetime.utcnow().isoformat(),
        "mode": mode,
    }

    existing = next((item for item in manifest["runs"] if item["id"] == run_id), None)
    if existing:
        existing.update(entry)
    else:
        manifest["runs"].append(entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("H1B KNOWLEDGE BASE — DAY 4 EVALUATION")
    print("=" * 70)

    dataset_path = Path("data/evaluation/test_questions.json")
    if not args.results_file and not dataset_path.exists():
        print("📝 Test dataset missing. Generating default question set...")
        from scripts.generate_test_dataset import generate_test_dataset

        generate_test_dataset()

    variants = ["baseline", "semantic_only", "bm25_only", "hybrid", "enhanced"]

    results_dir = Path("data/evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    mode = "reprocess" if args.results_file else "fresh"
    if args.results_file:
        raw_path = Path(args.results_file)
        if not raw_path.exists():
            raise FileNotFoundError(f"Results file not found: {raw_path}")
        print(f"🔁 Reprocessing cached results from {raw_path}")
        results = json.loads(raw_path.read_text())
    else:
        from src.evaluation.evaluator import Evaluator

        evaluator = Evaluator(str(dataset_path))
        results = evaluator.run_evaluation(variants, max_questions=args.max_questions)
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

    if args.skip_plots:
        print("⚠️ Plot generation skipped (flag enabled).")
    else:
        mpl_dir = ROOT / "data" / "evaluation" / "results"
        try:
            mpl_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
            from src.evaluation.visualizer import Visualizer

            visualizer = Visualizer()
            visualizer.generate_all_plots(metrics)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Plot generation skipped due to error: {exc}")

    question_count = len(next(iter(results["variants"].values()))["questions"])
    update_manifest(
        run_id=results["timestamp"],
        results_path=raw_path,
        metrics_path=metrics_path,
        variants=variants,
        mode=mode,
        dataset_path=dataset_path if dataset_path.exists() else None,
        question_count=question_count,
        max_questions=args.max_questions,
    )

    print("\nSummary:")
    for variant, stats in metrics.get("overall", {}).get("details", {}).items():
        print(f"  - {variant:12s}: {stats['accuracy']*100:5.1f}% accuracy, {stats['avg_time']:.2f}s avg time")
    best = metrics.get("overall", {}).get("best")
    if best:
        print(f"\n🏆 Best variant: {best}")


if __name__ == "__main__":
    main()
