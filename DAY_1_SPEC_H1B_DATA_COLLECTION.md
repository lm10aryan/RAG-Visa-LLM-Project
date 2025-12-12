# DAY 1 SPEC: H1B Data Collection System

**Goal:** Build a focused H1B visa data collection pipeline from scratch

**Timeline:** 3-4 hours

**Deliverables:**
- 30+ H1B-specific document chunks
- High-quality, focused data (not mixed visa types)
- Semantic chunking (preserves context)
- 20 structured H1B facts in SQLite database
- Embeddings generated and ready for retrieval

---

## Part 1: H1B Web Scraper (90 minutes)

### **What We're Building:**
A targeted web scraper that collects ONLY H1B-related content from official USCIS sources.

### **Why This Approach:**
- **Focused data** > Mixed data (current system had F1, H1B, everything mixed)
- **Official sources only** (no forums, no blogs - only USCIS.gov)
- **Respectful scraping** (1 second delay, proper headers, no rate limit violations)

### **Target Pages (Top Priority):**
```
1. https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations
2. https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season
3. https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/how-do-i-apply-for-h-1b-status
4. https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-fiscal-year-fy-2024-cap-season
5. https://www.uscis.gov/forms/h-and-l-filing-fees
```

### **Implementation Requirements:**

**File:** `src/ingestion/h1b_scraper.py`

**Core Functions:**
```python
def scrape_uscis_page(url: str) -> dict:
    """
    Scrape a single USCIS page
    
    Returns:
        {
            'url': str,
            'title': str,
            'content': str,  # Clean text, no HTML
            'scraped_at': datetime,
            'success': bool
        }
    """
    
def scrape_h1b_pages(urls: List[str]) -> List[dict]:
    """
    Scrape multiple pages with delay
    
    - 1 second delay between requests
    - Proper User-Agent header
    - Retry logic (max 3 attempts)
    - Error handling (skip failed pages, don't crash)
    """
```

**Requirements:**
- Use `requests` library for HTTP
- Use `BeautifulSoup` for HTML parsing
- Extract only main content (skip navigation, footer, ads)
- Clean text: remove extra whitespace, normalize newlines
- Save raw scraped data to: `data/raw/h1b_pages.json`

**Error Handling:**
- If page fails: log error, continue to next page
- If all pages fail: raise error with clear message
- Network timeout: 10 seconds max

**Output Format:**
```json
[
    {
        "url": "https://uscis.gov/...",
        "title": "H-1B Specialty Occupations",
        "content": "The H-1B program applies to employers...",
        "scraped_at": "2024-12-11T21:45:00",
        "success": true
    }
]
```

---

## Part 2: Semantic Chunking (60 minutes)

### **What We're Building:**
Smart chunking that preserves context (not just dumb 800-character splits).

### **Why This Approach:**
- **Context matters** - "65,000" without context is useless
- **Semantic boundaries** - Split at paragraph/section breaks, not mid-sentence
- **Header preservation** - Keep section headers with their content

### **Implementation Requirements:**

**File:** `src/ingestion/semantic_chunker.py`

**Strategy:**
```
1. Split by paragraphs (not fixed character count)
2. If paragraph > 800 chars → split at sentence boundary
3. If paragraph < 200 chars → merge with next paragraph
4. Keep section headers with their paragraphs
5. Add metadata: source_url, source_title, chunk_index
```

**Core Function:**
```python
def chunk_document(doc: dict, target_size: int = 600) -> List[dict]:
    """
    Chunk a document semantically
    
    Args:
        doc: Output from scraper {'url', 'title', 'content'}
        target_size: Target chunk size in characters (soft limit)
    
    Returns:
        List of chunks with metadata
    """
```

**Chunk Format:**
```python
{
    'text': str,              # The actual chunk text
    'source_url': str,        # Original page URL
    'source_title': str,      # Page title
    'chunk_id': str,          # Unique ID: "h1b_001"
    'chunk_index': int,       # Position in document
    'char_count': int,        # Length
    'created_at': datetime
}
```

**Output:**
- Save to: `data/processed/h1b_chunks.json`
- Also save metadata to: `chunks_metadata.pkl` (for compatibility with existing code)

**Quality Checks:**
- Minimum chunk size: 200 chars (discard smaller)
- Maximum chunk size: 1000 chars (split larger)
- No orphaned headers (headers always have content)
- No duplicate chunks

---

## Part 3: Structured Facts Database (45 minutes)

### **What We're Building:**
SQLite database with 20 critical H1B facts for Tier 1 (instant, 100% accurate lookup).

### **Why This Approach:**
- Questions like "What is H1B cap?" don't need RAG
- Direct database lookup = instant + perfect accuracy
- Forms the foundation of our 2-tier system

### **Implementation Requirements:**

**File:** `src/ingestion/db_builder.py`

**Database Schema:**
```sql
CREATE TABLE h1b_facts (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,           -- 'cap', 'fees', 'dates', 'requirements'
    question TEXT NOT NULL,            -- "What is the H1B cap?"
    answer TEXT NOT NULL,              -- "65,000 regular + 20,000 advanced degree"
    source_url TEXT NOT NULL,
    last_verified DATE NOT NULL
);

CREATE INDEX idx_question ON h1b_facts(question);
CREATE INDEX idx_category ON h1b_facts(category);
```

**20 Critical Facts to Include:**

**CAP (5 facts):**
1. What is the H1B cap? → 65,000 regular cap
2. What is the advanced degree cap? → 20,000 additional
3. When does cap season start? → March (registration period)
4. When do selected petitions start? → October 1st
5. Are there cap-exempt employers? → Yes (universities, nonprofits, research orgs)

**FEES (5 facts):**
1. What is the base filing fee? → $460 (I-129 form)
2. What is the fraud prevention fee? → $500
3. What is the ACWIA fee? → $750 or $1,500 (depends on company size)
4. What is premium processing fee? → $2,500 for 15-day processing
5. Total typical cost? → $1,710 to $2,710 (employer pays)

**REQUIREMENTS (5 facts):**
1. Education requirement? → Bachelor's degree or equivalent
2. Job requirement? → Specialty occupation (theoretical/technical expertise)
3. Can I apply myself? → No, employer must petition
4. How long is H1B valid? → 3 years initially, extendable to 6 years total
5. Can I change employers? → Yes, through H1B transfer

**PROCESS (5 facts):**
1. How long does processing take? → 2-4 months (regular), 15 days (premium)
2. Can I work while petition pending? → Only if H1B transfer with valid status
3. What form is used? → Form I-129 (Petition for Nonimmigrant Worker)
4. What is LCA? → Labor Condition Application (required before I-129)
5. Can I apply for green card on H1B? → Yes, H1B allows dual intent

**Implementation:**
```python
def create_facts_database(db_path: str = "data/structured/h1b_facts.db"):
    """Create and populate facts database"""
    
def lookup_fact(question: str) -> dict:
    """
    Exact match lookup for Tier 1 routing
    
    Returns:
        {
            'question': str,
            'answer': str,
            'source_url': str,
            'confidence': 1.0  # Always 1.0 for DB lookups
        }
    """
```

---

## Part 4: Generate Embeddings (45 minutes)

### **What We're Building:**
Convert chunks into 384-dimensional vectors using sentence-transformers.

### **Why This Approach:**
- **sentence-transformers** is industry standard for semantic search
- **all-MiniLM-L6-v2** is fast, good quality, and free
- **384 dimensions** is sweet spot (not too sparse, not too slow)

### **Implementation Requirements:**

**File:** `src/ingestion/embedder.py`

**Core Function:**
```python
def generate_embeddings(chunks: List[dict]) -> np.ndarray:
    """
    Generate embeddings for all chunks
    
    Args:
        chunks: List of chunk dicts from semantic_chunker
    
    Returns:
        numpy array of shape (N, 384)
    """
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Batch processing for efficiency
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity
        show_progress_bar=True,
        batch_size=32
    )
    
    return embeddings
```

**Output Files:**
- `embeddings.npy` - numpy array (N, 384)
- `chunks_metadata.pkl` - list of chunk dicts (same order as embeddings)

**Validation:**
```python
# Sanity checks
assert embeddings.shape[0] == len(chunks)
assert embeddings.shape[1] == 384
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # Check normalization
```

---

## Part 5: Integration & Testing (30 minutes)

### **What We're Building:**
Master script that runs the entire pipeline and validates output.

### **Implementation Requirements:**

**File:** `scripts/build_knowledge_base.py`

```python
def main():
    """
    Run complete data collection pipeline
    
    Steps:
    1. Scrape H1B pages
    2. Chunk documents semantically
    3. Build structured facts database
    4. Generate embeddings
    5. Save all outputs
    6. Validate everything
    """
    
    print("🕷️  Step 1: Scraping H1B pages...")
    pages = scrape_h1b_pages(H1B_URLS)
    print(f"✅ Scraped {len(pages)} pages")
    
    print("✂️  Step 2: Chunking documents...")
    chunks = []
    for page in pages:
        chunks.extend(chunk_document(page))
    print(f"✅ Created {len(chunks)} chunks")
    
    print("🗄️  Step 3: Building facts database...")
    create_facts_database()
    print(f"✅ Database created with 20 facts")
    
    print("🧮 Step 4: Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    np.save('embeddings.npy', embeddings)
    with open('chunks_metadata.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ Saved {embeddings.shape[0]} embeddings")
    
    print("✅ Knowledge base built successfully!")
    print(f"   - {len(chunks)} H1B chunks")
    print(f"   - 20 structured facts")
    print(f"   - 384D embeddings")
```

**Validation Checks:**
```python
# Must pass all these:
assert len(chunks) >= 30, "Need at least 30 H1B chunks"
assert all('h1b' in c['text'].lower() or 'h-1b' in c['text'].lower() 
           for c in chunks[:5]), "First 5 chunks should mention H1B"
assert os.path.exists('embeddings.npy')
assert os.path.exists('chunks_metadata.pkl')
assert os.path.exists('data/structured/h1b_facts.db')
```

---

## Expected Outputs (End of Day 1)

**Files Created:**
```
data/
├── raw/
│   └── h1b_pages.json           # 5 scraped pages
├── processed/
│   └── h1b_chunks.json          # 30-50 chunks
└── structured/
    └── h1b_facts.db             # SQLite with 20 facts

embeddings.npy                    # (N, 384) numpy array
chunks_metadata.pkl               # List of N chunk dicts
```

**Quality Metrics:**
- Total chunks: 30-50 (H1B focused)
- Average chunk size: 500-700 characters
- H1B coverage: 100% (all chunks mention H1B)
- Structured facts: 20 (all verified)
- Embedding dimensions: 384

**Test Command:**
```python
# Should work after Day 1:
from query_numpy import RAGRetriever

retriever = RAGRetriever()
results = retriever.retrieve("What is the H1B cap?", top_k=3)

# Expected: Top result mentions "65,000"
print(results[0]['text'][:200])
```

---

## Success Criteria

✅ **Must Have:**
- 30+ H1B chunks created
- All chunks are H1B-related (not mixed visa types)
- Embeddings generated successfully
- Database has 20 facts
- Test query retrieves relevant chunks

✅ **Nice to Have:**
- 50+ chunks (more is better)
- Chunks preserve context (headers + content together)
- High-quality text (no HTML artifacts)
- Source URLs all valid

---

## Common Issues & Solutions

**Issue:** "Too few chunks (<30)"
**Solution:** Scrape 2-3 additional H1B pages from USCIS

**Issue:** "Chunks too small (< 300 chars)"
**Solution:** Adjust merging logic in semantic chunker

**Issue:** "Embedding generation slow"
**Solution:** Normal - takes ~1 minute for 50 chunks on CPU

**Issue:** "Some chunks don't mention H1B"
**Solution:** Filter during chunking - skip generic immigration content

---

## Next Steps (Day 2)

After Day 1 completes:
- Build query router (classifies questions → Tier 1 or Tier 2)
- Implement Tier 1 (structured database lookup)
- Implement Tier 2 (hybrid RAG retrieval)
- Create initial test dataset (30 questions)

---

## Notes for Claude Code

When implementing this:
1. Create all 5 files mentioned above
2. Run the build script: `python scripts/build_knowledge_base.py`
3. Test with: `python query_numpy.py "What is H1B cap?"`
4. Report back: number of chunks created + sample chunk

**Focus on quality over quantity** - 30 great H1B chunks > 100 mixed visa chunks.
