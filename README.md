# RAG-Visa-LLM-Project

## Prompt trace helper
Use the prompt trace helper to exercise the RAG pipeline end-to-end with visibility into query enhancement, routing, and citations:

```bash
python scripts/run_prompt_trace.py "What is the H-1B cap season start date?"
```

Flags:
- `--no-enhance` to disable query rewriting
- `--no-citation-validation` to skip citation checks for faster iterations

The helper now also prints how the retrieval stages connect:

- **Hybrid fusion** shows the top chunks after combining semantic and BM25 scores.
- **Reranker scoring** breaks down the bonuses applied for authority, completeness, query alignment, and recency.
- **Sources** displays the final context sent to the generator and any citation validation results.

## Two-tier routing at a glance
The system answers questions in two passes:

1) **Tier 1: Structured facts.** If a question strongly matches a curated SQLite fact (e.g., filing windows, lottery caps, fee amounts), the `Tier1Handler` returns that fact directly with a confidence boost and validated citations. This prevents LLM hallucinations when the answer is a known, unambiguous data point.
2) **Tier 2: Hybrid RAG.** When no structured fact meets the confidence threshold, the query falls back to the hybrid retriever (semantic + BM25 fusion) and reranker, which assembles contextual chunks for generation.

Recent changes restored Tier 1 routing with stricter thresholds and citation validation, so structured answers are only used when there is a high-quality match, otherwise Tier 2 handles open-ended questions.
