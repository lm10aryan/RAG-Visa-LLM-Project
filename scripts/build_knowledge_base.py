from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.db.facts_builder import create_facts_database  # noqa: E402
from src.embeddings.embedder import generate_embeddings  # noqa: E402
from src.ingestion.h1b_scraper import scrape_h1b_pages  # noqa: E402
from src.ingestion.semantic_chunker import chunk_document, save_chunks  # noqa: E402

OFFLINE_HTML_DIR = Path("data/manual/html")

H1B_PAGE_VARIANTS = [
    [
        "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/how-do-i-apply-for-h-1b-status",
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/how-do-i-apply-for-h-1b-status",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/h-1b-fiscal-year-fy-2024-cap-season",
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-fiscal-year-fy-2024-cap-season",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/h-1b-electronic-registration-process",
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-electronic-registration-process",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/faqs-for-individuals-in-h-1b-nonimmigrant-status",
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/faqs-for-individuals-in-h-1b-nonimmigrant-status",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/employment-authorization-for-certain-h-4-dependent-spouses",
    ],
    [
        "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/extension-of-post-completion-optional-practical-training-opt-and-f-1-status-for-eligible-students",
    ],
    [
        "https://www.uscis.gov/forms/filing-fees",
        "https://www.uscis.gov/forms/h-and-l-filing-fees",
    ],
]


def build_pipeline() -> None:
    print("🚀 Starting H1B knowledge base build...")
    pages = scrape_h1b_pages(H1B_PAGE_VARIANTS, offline_dir=OFFLINE_HTML_DIR)
    print(f"✅ Scraped {len(pages)} USCIS H1B pages")

    print("✂️  Chunking documents semantically...")
    chunks = []
    for page in pages:
        chunks.extend(chunk_document(page))

    if len(chunks) < 30:
        raise RuntimeError(f"Insufficient chunks: produced {len(chunks)} < 30 target.")

    save_chunks(chunks)
    print(f"✅ Created and saved {len(chunks)} H1B-focused chunks")

    print("🗄️  Building structured facts database...")
    create_facts_database()
    print("✅ Facts database populated with 20 entries")

    print("🧮 Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    np.save("embeddings.npy", embeddings)

    with open("chunks_metadata.pkl", "wb") as handle:
        pickle.dump(chunks, handle)

    print(f"✅ Saved {embeddings.shape[0]} embeddings ({embeddings.shape[1]} dimensions)")
    print("🎉 Knowledge base build complete!")


if __name__ == "__main__":
    build_pipeline()
