# Task: Update Evaluation Modules for 5-Level Difficulty System

## Context

I have an H1B visa RAG (Retrieval-Augmented Generation) evaluation system. I recently expanded my test dataset from 30 to 100 questions and changed the difficulty labeling system.

**Old system:** `easy`, `medium`, `hard` (3 levels)
**New system:** `L1_fact`, `L2_definition`, `L3_conditional`, `L4_multi_step`, `L5_edge_case` (5 levels)

The evaluation modules need to be updated to track and visualize accuracy by these new difficulty levels.

---

## Files to Modify

All files are in the project root: `RAG-Visa-LLM-Project-main/`

### 1. `src/evaluation/evaluator.py`
### 2. `src/evaluation/metrics.py`
### 3. `src/evaluation/visualizer.py`

---

## Specific Changes Required

### FILE 1: `src/evaluation/evaluator.py`

**Change A:** In the `_run_variant` method, add `"by_difficulty": {}` to the `variant_results` dict (around line 63-70).

**Change B:** After the `category_bucket` tracking block (around line 129-135), add similar tracking for difficulty:

```python
difficulty_bucket = variant_results["by_difficulty"].setdefault(
    test_q["difficulty"],
    {"correct": 0, "total": 0},
)
difficulty_bucket["total"] += 1
if is_correct:
    difficulty_bucket["correct"] += 1
```

**Change C:** After the loop where `by_category` accuracy is calculated (around line 140-144), add the same for `by_difficulty`:

```python
for stats in variant_results["by_difficulty"].values():
    if stats["total"]:
        stats["accuracy"] = stats["correct"] / stats["total"]
    else:
        stats["accuracy"] = 0.0
```

---

### FILE 2: `src/evaluation/metrics.py`

**Change A:** In the `_calculate_variant_metrics` method, add `by_difficulty` to the accuracy dict:

```python
"accuracy": {
    "overall": data.get("accuracy", 0.0),
    "by_category": {
        cat: stats.get("accuracy", 0.0)
        for cat, stats in data.get("by_category", {}).items()
    },
    "by_difficulty": {
        diff: stats.get("accuracy", 0.0)
        for diff, stats in data.get("by_difficulty", {}).items()
    },
},
```

---

### FILE 3: `src/evaluation/visualizer.py`

**Change A:** In `generate_all_plots`, add a call to the new difficulty chart method. Update the numbering:

```python
def generate_all_plots(self, metrics: Dict) -> None:
    self.plot_overall_accuracy(metrics, self.output_dir / "01_overall_accuracy.png")
    self.plot_category_breakdown(metrics, self.output_dir / "02_category_breakdown.png")
    self.plot_difficulty_breakdown(metrics, self.output_dir / "03_difficulty_breakdown.png")
    self.plot_ablation(metrics, self.output_dir / "04_ablation_study.png")
    self.plot_confidence(metrics, self.output_dir / "05_confidence_calibration.png")
    self.plot_response_time(metrics, self.output_dir / "06_response_times.png")
```

**Change B:** Add a new method `plot_difficulty_breakdown`. Place it after `plot_category_breakdown`:

```python
def plot_difficulty_breakdown(self, metrics: Dict, path: Path) -> None:
    """Plot accuracy by difficulty level for each variant."""
    variants = metrics.get("by_variant", {})
    if not variants:
        return
    
    # Define difficulty levels in order (easy to hard)
    difficulty_levels = ["L1_fact", "L2_definition", "L3_conditional", "L4_multi_step", "L5_edge_case"]
    display_labels = ["L1\nFact", "L2\nDefinition", "L3\nConditional", "L4\nMulti-step", "L5\nEdge Case"]
    
    x = np.arange(len(difficulty_levels))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"]
    
    for idx, (name, stats) in enumerate(variants.items()):
        by_diff = stats.get("accuracy", {}).get("by_difficulty", {})
        accs = [by_diff.get(level, 0.0) * 100 for level in difficulty_levels]
        bars = ax.bar(x + idx * width, accs, width=width, label=name, color=colors[idx % len(colors)])
    
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by Difficulty Level")
    ax.legend(loc="upper right")
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% target')
    
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
```

---

## Verification Steps

After making changes, please verify:

1. **Syntax check:** Run `python -m py_compile src/evaluation/evaluator.py src/evaluation/metrics.py src/evaluation/visualizer.py`

2. **Import check:** Run `python -c "from src.evaluation.evaluator import Evaluator; from src.evaluation.metrics import MetricsCalculator; from src.evaluation.visualizer import Visualizer; print('All imports OK')"`

3. **Show me the diff:** After each file change, show me what you changed so I can verify.

---

## Important Notes

- Do NOT modify any other logic (retrieval, variants, correctness checking)
- Keep all existing functionality intact
- The 5 difficulty levels should appear in this order in charts: L1_fact → L2_definition → L3_conditional → L4_multi_step → L5_edge_case
- If you encounter any errors, show me the error and ask before proceeding

---

## Expected Output

When done, confirm:
- [ ] `evaluator.py` updated with `by_difficulty` tracking
- [ ] `metrics.py` updated with `by_difficulty` in accuracy dict
- [ ] `visualizer.py` updated with new `plot_difficulty_breakdown` method
- [ ] All syntax checks pass
- [ ] All imports work

Then tell me: "Ready to run evaluation. Use: `python scripts/run_evaluation.py`"
