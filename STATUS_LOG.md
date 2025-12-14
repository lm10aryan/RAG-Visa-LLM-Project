# Project Status Log

_Last updated: Fri, 12 Dec 2025_

## Day 1 — H1B Data Collection Pipeline
- Set up project structure (`src`, `data/raw|processed|structured`, `scripts`) and requirements.
- Implemented USCIS-only scraper with retry/delay + offline fallbacks (`src/ingestion/h1b_scraper.py`).
- Built semantic chunker preserving headers/context (`src/ingestion/semantic_chunker.py`); generated 94 H1B chunks.
- Seeded SQLite facts DB with 21 question-driven facts (`src/db/facts_builder.py`).
- Added embedding generation with MiniLM + saved `embeddings.npy` and `chunks_metadata.pkl`.
- Created build script (`scripts/build_knowledge_base.py`) and CLI retriever (`query_numpy.py`).
- Outputs generated:
  - `data/raw/h1b_pages.json`
  - `data/processed/h1b_chunks.json`
  - `data/structured/h1b_facts.db`
  - `embeddings.npy`, `chunks_metadata.pkl`

## Day 2 — 2-Tier Hybrid RAG System
- Added utilities for stable chunk IDs (`src/utils/chunk_utils.py`).
- Routing:
  - Implemented query router with fuzzy DB match + heuristics (`src/routing/query_router.py`).
- Retrieval:
  - Refactored semantic retriever (`src/retrieval/semantic_retriever.py`).
  - Added BM25 search (`src/retrieval/bm25_search.py`) and hybrid ranker with RRF + weighted fusion (`src/retrieval/hybrid_ranker.py`).
  - Added re-ranking layer (authority/completeness/answer bonuses) (`src/retrieval/reranker.py`).
- Reasoning tiers:
  - Tier 1 handler for fact lookups (`src/reasoning/tier1_handler.py`).
  - Tier 2 handler combining hybrid search + answer synthesis (`src/reasoning/tier2_handler.py`).
- Orchestrator:
  - `VisaRAG` exposes unified `.query()` and comparison logging (`src/visa_rag.py`).
- Structured facts injected into chunk metadata for hybrid coverage (cap numbers, fees).
- Tests:
  - `tests/test_tier_system.py` (router accuracy + hybrid cap validation) — **PASS**.
  - `tests/test_integration.py` — **PASS** (Tier 1/2 answers verified).

## Day 3 — Intelligence Layer
- Query Enhancement:
  - Added `QueryEnhancer` for abbreviation, synonym, and numeric expansion (`src/enhancement/query_enhancer.py`).
  - Integrated enhancement pipeline + stats into `VisaRAG` (original query for routing, enhanced for retrieval).
- Confidence Scoring:
  - Multi-factor confidence via `ConfidenceScorer` (agreement, source quality, retrieval strength, completeness).
  - Tier 1 + Tier 2 responses now include `confidence_level`, reasoning, and factor breakdown.
- Advanced Re-Ranking:
  - Smart reranker with recency bonuses (using `scraped_at`), query-type weighting, and contradiction penalties.
- Citation Validation:
  - Added lightweight validator to annotate sources with accessibility/freshness + aggregate citation quality scores.
  - Both tiers now support validation toggles.
- Tests:
  - `tests/test_query_enhancement.py` — abbreviation + number normalization coverage.
  - `tests/test_intelligence_layer.py` — verifies enhancement flag + confidence metadata.
  - Existing tier + integration tests still pass with new features.

## Commands Run (latest session)
```
python scripts/build_knowledge_base.py
pytest tests/test_tier_system.py
pytest tests/test_query_enhancement.py
pytest tests/test_intelligence_layer.py
python tests/test_integration.py
```

## Current Artifacts
- 94+ semantic chunks (USCIS-only) + 21 structured fact chunks (with `scraped_at` + chunk IDs).
- SQLite DB with 21 fact entries (question-driven).
- Embeddings: 384-D vectors aligned with enhanced metadata.
- Query logs + confidence/citation annotations stored during `VisaRAG` usage.

## Day 4 — Evaluation Framework
- Dataset:
  - `scripts/generate_test_dataset.py` produces 30 benchmark questions (fact, requirement, complex).
  - `data/evaluation/test_questions.json` stores questions with ground-truth answers + tier expectations.
- Harness & Metrics:
  - `src/evaluation/evaluator.py` runs baseline, semantic-only, BM25-only, hybrid, and enhanced variants.
  - `src/evaluation/model_manager.py` provides deterministic text generation for baseline/ablation variants.
  - `src/evaluation/metrics.py` computes accuracy, calibration, citation coverage, and retrieval precision.
  - `src/evaluation/visualizer.py` renders comparison plots (saved under `data/evaluation/figures/`).
  - Orchestration via `scripts/run_evaluation.py` (auto-generates dataset if missing, runs pipeline, saves metrics).
- Dependencies updated with `matplotlib` for plotting.

## Commands Run (latest session)
```
python3 scripts/generate_test_dataset.py
python3 scripts/run_evaluation.py      # run when ready; may take time
pytest tests/test_tier_system.py
pytest tests/test_query_enhancement.py
pytest tests/test_intelligence_layer.py
python tests/test_integration.py
```

## Current Artifacts Snapshot
- Knowledge base assets unchanged (chunks/DB/embeddings).
- Evaluation assets new:
  - `data/evaluation/test_questions.json`
  - `data/evaluation/results/` (raw results + metrics JSON after running)
  - `data/evaluation/figures/` (plots after running visualizer)

## Next Steps
1. Execute `python scripts/run_evaluation.py` to produce full metrics + figures (takes ~30–45 minutes).
2. Summarize findings (accuracy table, confidence calibration, ablation takeaway) for Day 4 report.
3. Begin prepping Day 5 write-up/slides/demo based on collected metrics.
