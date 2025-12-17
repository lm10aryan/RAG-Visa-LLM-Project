# RAG-Visa-LLM-Project

## Evaluation Pipeline

Use the updated CLI wrapper to either run a fresh Groq-powered evaluation or reprocess cached results without new API calls.

```bash
cd /Users/aryanthepade/Desktop/Coding/network-final-project
source venv/bin/activate

# Reprocess an existing run (skips Groq/model downloads)
python scripts/run_evaluation.py \
  --results-file data/evaluation/results/results_20251217T210508.965475.json \
  --skip-plots

# Run a fresh evaluation (optional --max-questions to limit cost)
python scripts/run_evaluation.py --max-questions 10
```

- `--results-file` tells the script to reuse a saved `results_*.json`.
- `--max-questions` throttles the number of prompts per variant when running fresh evaluations.
- `--skip-plots` avoids Matplotlib generation when the environment lacks a writable cache.

Every run records metadata in `data/evaluation/run_history.json`, so downstream tooling or a frontend can list past runs, their artifacts, and configuration. Raw per-question outputs live under `data/evaluation/results/`, and aggregated metrics are always written to `data/evaluation/results/metrics.json`.
