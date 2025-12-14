from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Generate comparison plots for evaluation results."""

    def __init__(self, output_dir: str = "data/evaluation/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self, metrics: Dict) -> None:
        self.plot_overall_accuracy(metrics, self.output_dir / "01_overall_accuracy.png")
        self.plot_category_breakdown(metrics, self.output_dir / "02_category_breakdown.png")
        self.plot_ablation(metrics, self.output_dir / "03_ablation_study.png")
        self.plot_confidence(metrics, self.output_dir / "04_confidence_calibration.png")
        self.plot_response_time(metrics, self.output_dir / "05_response_times.png")

    def plot_overall_accuracy(self, metrics: Dict, path: Path) -> None:
        variants = metrics.get("by_variant", {})
        if not variants:
            return
        names = list(variants.keys())
        accuracies = [variants[name]["accuracy"]["overall"] * 100 for name in names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, accuracies, color="#2ecc71")
        for bar, value in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}%", ha="center")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Overall Accuracy by Variant")
        plt.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def plot_category_breakdown(self, metrics: Dict, path: Path) -> None:
        variants = metrics.get("by_variant", {})
        if not variants:
            return
        categories = ["fact", "requirement", "complex"]
        x = np.arange(len(categories))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (name, stats) in enumerate(variants.items()):
            accs = [stats["accuracy"]["by_category"].get(cat, 0.0) * 100 for cat in categories]
            ax.bar(x + idx * width, accs, width=width, label=name)

        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels([cat.title() for cat in categories])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Question Category")
        ax.legend()
        plt.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def plot_ablation(self, metrics: Dict, path: Path) -> None:
        ablation = metrics.get("ablation")
        if not ablation:
            return
        components = ["semantic_contribution", "hybrid_contribution", "enhancement_contribution"]
        values = [ablation.get(comp, 0.0) * 100 for comp in components]
        labels = ["Semantic RAG gain", "Hybrid gain", "Intelligence gain"]

        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative = 0
        colors = ["#3498db", "#9b59b6", "#f1c40f"]
        for value, label, color in zip(values, labels, colors):
            ax.barh(0, value, left=cumulative, color=color, label=label)
            ax.text(cumulative + value / 2, 0, f"+{value:.1f}%", ha="center", va="center", color="white", fontweight="bold")
            cumulative += value

        ax.set_xlabel("Accuracy (%)")
        ax.set_yticks([])
        ax.set_title("Ablation Study")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def plot_confidence(self, metrics: Dict, path: Path) -> None:
        enhanced = metrics.get("by_variant", {}).get("enhanced")
        if not enhanced:
            return
        calibration = enhanced.get("confidence_calibration", {})
        if not calibration:
            return
        levels = ["low", "medium", "high"]
        accs = [calibration.get(level, {}).get("accuracy", 0.0) * 100 for level in levels]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(levels, accs, marker="o", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Actual Accuracy (%)")
        ax.set_title("Confidence Calibration (Enhanced Variant)")
        plt.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def plot_response_time(self, metrics: Dict, path: Path) -> None:
        variants = metrics.get("by_variant", {})
        if not variants:
            return
        names = list(variants.keys())
        times = [variants[name]["response_time"]["avg"] for name in names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, times, color="#95a5a6")
        for bar, val in zip(bars, times):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}s", va="center")
        ax.set_xlabel("Average Response Time (s)")
        ax.set_title("Response Time by Variant")
        plt.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


__all__ = ["Visualizer"]

