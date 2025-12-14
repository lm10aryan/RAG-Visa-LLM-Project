# DAY 2 SPEC: 2-Tier Intelligent RAG System

**Goal:** Build intelligent routing + hybrid retrieval system that outperforms single-method approaches

**Timeline:** 3-4 hours

**Deliverables:**
- Query router (classifies Tier 1 vs Tier 2)
- BM25 keyword search implementation
- Hybrid fusion ranker (semantic + BM25)
- Smart re-ranking layer
- Complete 2-tier system with unified interface
- Comparison logging (for ablation study)

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ QUERY ROUTER                        │
│ Classifies: Fact vs Complex         │
└─────────────┬───────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
[TIER 1]          [TIER 2]
     │                 │
┌────▼─────┐     ┌─────▼──────────────────────┐
│ SQLITE   │     │ HYBRID RETRIEVAL           │
│ LOOKUP   │     │                            │
│          │     │ 1. Semantic Search         │
│ Direct   │     │    └─ Top 20 (score_sem)  │
│ answer   │     │                            │
│ 100%     │     │ 2. BM25 Search             │
│ accurate │     │    └─ Top 20 (score_bm25) │
│          │     │                            │
│          │     │ 3. Fusion                  │
│          │     │    └─ Combine scores       │
│          │     │    └─ Top 10 candidates   │
│          │     │                            │
│          │     │ 4. Re-Ranking              │
│          │     │    └─ Authority bonus      │
│          │     │    └─ Completeness bonus   │
│          │     │    └─ Top 3 final         │
│          │     │                            │
│          │     │ 5. LLM Generation          │
│          │     │    └─ Context + Query      │
└────┬─────┘     └─────┬──────────────────────┘
     │                 │
     └────────┬────────┘
              ▼
     Formatted Response
```

---

## Part 1: Query Router (45 minutes)

### **What It Does:**
Intelligently routes questions to the right tier based on question type.

### **Classification Logic:**

**Tier 1 (Structured DB) - Fact Questions:**
```
Patterns:
├─ Starts with "What is/are"
├─ Contains exact terms: "cap", "fee", "cost", "$"
├─ Asks for numbers: "how much", "how many"
└─ Single-fact questions

Examples:
├─ "What is the H1B cap?" → Tier 1
├─ "How much does H1B cost?" → Tier 1
├─ "What are the filing fees?" → Tier 1
└─ "How many H1Bs are issued?" → Tier 1
```

**Tier 2 (Hybrid RAG) - Complex Questions:**
```
Patterns:
├─ "Am I eligible..."
├─ "Can I..." (permission/possibility)
├─ "How do I..." (process questions)
├─ Multi-part questions
└─ Scenario-based questions

Examples:
├─ "Am I eligible with CS degree?" → Tier 2
├─ "Can I change employers on H1B?" → Tier 2
├─ "How do I apply for H1B?" → Tier 2
└─ "I'm on F1 OPT, can I get H1B?" → Tier 2
```

### **Implementation:**

**File:** `src/routing/query_router.py`

```python
class QueryRouter:
    """Routes queries to appropriate tier"""
    
    def __init__(self, facts_db_path: str):
        self.facts_db = self._load_facts_db(facts_db_path)
        
        # Tier 1 indicators
        self.fact_patterns = [
            r'^what is ',
            r'^what are ',
            r'how much',
            r'how many',
            r'\$\d+',  # Dollar amounts
        ]
        
        self.fact_keywords = [
            'cap', 'fee', 'cost', 'price', 
            'number', 'amount', 'date', 'when'
        ]
    
    def route(self, query: str) -> dict:
        """
        Determine which tier should handle this query
        
        Returns:
            {
                'tier': 1 or 2,
                'confidence': float (0-1),
                'reasoning': str (why this tier)
            }
        """
        query_lower = query.lower().strip()
        
        # Check for exact DB match first
        if self._has_exact_match(query_lower):
            return {
                'tier': 1,
                'confidence': 1.0,
                'reasoning': 'Exact match in facts database'
            }
        
        # Check patterns
        tier1_score = self._calculate_tier1_score(query_lower)
        
        if tier1_score > 0.7:
            return {
                'tier': 1,
                'confidence': tier1_score,
                'reasoning': 'Fact-based question pattern detected'
            }
        else:
            return {
                'tier': 2,
                'confidence': 1.0 - tier1_score,
                'reasoning': 'Complex question requiring context retrieval'
            }
    
    def _has_exact_match(self, query: str) -> bool:
        """Check if query matches a fact in DB"""
        # Fuzzy match against fact questions
        for fact_question in self.facts_db.keys():
            similarity = self._string_similarity(query, fact_question.lower())
            if similarity > 0.85:
                return True
        return False
    
    def _calculate_tier1_score(self, query: str) -> float:
        """Score likelihood this is a Tier 1 question"""
        score = 0.0
        
        # Check patterns
        for pattern in self.fact_patterns:
            if re.search(pattern, query):
                score += 0.3
        
        # Check keywords
        for keyword in self.fact_keywords:
            if keyword in query:
                score += 0.2
        
        # Penalize for complex indicators
        complex_indicators = ['am i', 'can i', 'should i', 'how do i']
        for indicator in complex_indicators:
            if indicator in query:
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple similarity metric (can use difflib)"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
```

### **Testing Router:**
```python
test_queries = [
    ("What is the H1B cap?", 1),
    ("How much does H1B cost?", 1),
    ("Am I eligible for H1B?", 2),
    ("Can I change employers?", 2),
    ("What is premium processing fee?", 1),
    ("How do I apply for H1B?", 2),
]

router = QueryRouter('data/structured/h1b_facts.db')
for query, expected_tier in test_queries:
    result = router.route(query)
    print(f"Query: {query}")
    print(f"  → Tier {result['tier']} (expected {expected_tier})")
    print(f"  → Confidence: {result['confidence']:.2f}")
    print(f"  → Reasoning: {result['reasoning']}\n")
```

---

## Part 2: BM25 Keyword Search (45 minutes)

### **What It Does:**
Ranks chunks by exact keyword matching using BM25 algorithm.

### **Why BM25:**
```
Example Query: "What is H1B cap?"

BM25 scoring factors:
1. Term Frequency (TF): How often "H1B" and "cap" appear in chunk
2. Inverse Document Frequency (IDF): How rare these terms are across all chunks
3. Document Length: Normalize by chunk length

Result: Chunks with BOTH "H1B" AND "cap" score highest
```

### **Implementation:**

**File:** `src/retrieval/bm25_search.py`

```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict

class BM25Searcher:
    """BM25 keyword search over document chunks"""
    
    def __init__(self, chunks: List[Dict]):
        """
        Initialize BM25 index
        
        Args:
            chunks: List of chunk dicts with 'text' field
        """
        self.chunks = chunks
        
        # Tokenize all chunks
        self.tokenized_chunks = [
            self._tokenize(chunk['text']) 
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        print(f"✅ BM25 index built for {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search using BM25 keyword matching
        
        Args:
            query: User query string
            top_k: Number of results to return
        
        Returns:
            List of dicts with chunk + BM25 score
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                result = self.chunks[idx].copy()
                result['bm25_score'] = float(scores[idx])
                result['chunk_index'] = int(idx)
                results.append(result)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase + split on whitespace
        
        Could be enhanced with:
        - Stemming (running → run)
        - Stop word removal (the, a, an)
        - N-grams (H-1B as single token)
        """
        # Basic tokenization
        tokens = text.lower().split()
        
        # Remove punctuation
        tokens = [
            ''.join(c for c in token if c.isalnum() or c == '-')
            for token in tokens
        ]
        
        # Filter empty tokens
        tokens = [t for t in tokens if t]
        
        return tokens
```

### **Testing BM25:**
```python
# Load chunks
with open('chunks_metadata.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Initialize BM25
bm25_searcher = BM25Searcher(chunks)

# Test query
results = bm25_searcher.search("What is H1B cap?", top_k=5)

for i, result in enumerate(results, 1):
    print(f"\n{i}. BM25 Score: {result['bm25_score']:.3f}")
    print(f"   Text: {result['text'][:150]}...")
    print(f"   Source: {result['source_url']}")
```

---

## Part 3: Hybrid Fusion (60 minutes)

### **What It Does:**
Combines semantic and BM25 scores to get best of both worlds.

### **Fusion Strategy:**

**Reciprocal Rank Fusion (RRF):**
```
Instead of combining raw scores, combine ranks:

Semantic results:
1. Chunk A (score: 0.89)
2. Chunk C (score: 0.76)
3. Chunk B (score: 0.65)

BM25 results:
1. Chunk C (score: 12.5)
2. Chunk A (score: 10.2)
3. Chunk D (score: 8.1)

RRF score:
Chunk A: 1/(1+1) + 1/(1+2) = 0.833
Chunk C: 1/(1+2) + 1/(1+1) = 0.833
Chunk B: 1/(1+3) + 0 = 0.250
Chunk D: 0 + 1/(1+3) = 0.250

Final ranking: Chunk A, Chunk C (tied), then B, D
```

**Why RRF > Weighted Average:**
- Don't need to normalize scores (different scales)
- More robust to score distribution differences
- Proven to work well in research (TREC competitions)

### **Implementation:**

**File:** `src/retrieval/hybrid_ranker.py`

```python
class HybridRanker:
    """Combines semantic and BM25 search results"""
    
    def __init__(
        self, 
        semantic_searcher,
        bm25_searcher,
        k: int = 60  # RRF constant (typical value)
    ):
        self.semantic = semantic_searcher
        self.bm25 = bm25_searcher
        self.k = k
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        fusion_method: str = 'rrf'  # 'rrf' or 'weighted'
    ) -> List[Dict]:
        """
        Hybrid search combining semantic + BM25
        
        Args:
            query: User query
            top_k: Final number of results
            fusion_method: 'rrf' (recommended) or 'weighted'
        
        Returns:
            List of chunks ranked by hybrid score
        """
        # Get results from both methods
        semantic_results = self.semantic.retrieve(query, top_k=20)
        bm25_results = self.bm25.search(query, top_k=20)
        
        # Combine using chosen method
        if fusion_method == 'rrf':
            combined = self._reciprocal_rank_fusion(
                semantic_results, 
                bm25_results
            )
        else:
            combined = self._weighted_fusion(
                semantic_results,
                bm25_results,
                semantic_weight=0.6
            )
        
        # Sort by hybrid score and return top-k
        combined.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return combined[:top_k]
    
    def _reciprocal_rank_fusion(
        self, 
        semantic_results: List[Dict],
        bm25_results: List[Dict]
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Score = 1/(k + rank_semantic) + 1/(k + rank_bm25)
        """
        # Build maps: chunk_id → rank
        semantic_ranks = {
            r['chunk_id']: i + 1 
            for i, r in enumerate(semantic_results)
        }
        bm25_ranks = {
            r['chunk_id']: i + 1 
            for i, r in enumerate(bm25_results)
        }
        
        # Get all unique chunks
        all_chunk_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        combined = []
        chunk_map = {r['chunk_id']: r for r in semantic_results + bm25_results}
        
        for chunk_id in all_chunk_ids:
            # Get ranks (or large number if not present)
            sem_rank = semantic_ranks.get(chunk_id, 100)
            bm25_rank = bm25_ranks.get(chunk_id, 100)
            
            # RRF formula
            rrf_score = (1.0 / (self.k + sem_rank)) + (1.0 / (self.k + bm25_rank))
            
            # Get chunk data
            chunk = chunk_map[chunk_id].copy()
            chunk['hybrid_score'] = rrf_score
            chunk['semantic_rank'] = sem_rank if sem_rank < 100 else None
            chunk['bm25_rank'] = bm25_rank if bm25_rank < 100 else None
            
            combined.append(chunk)
        
        return combined
    
    def _weighted_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        semantic_weight: float = 0.6
    ) -> List[Dict]:
        """
        Simple weighted average of normalized scores
        
        Score = α * score_semantic + (1-α) * score_bm25
        """
        # Normalize scores to [0, 1]
        semantic_scores = [r['score'] for r in semantic_results]
        bm25_scores = [r['bm25_score'] for r in bm25_results]
        
        sem_max = max(semantic_scores) if semantic_scores else 1.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        
        # Build chunk maps with normalized scores
        sem_map = {
            r['chunk_id']: r['score'] / sem_max
            for r in semantic_results
        }
        bm25_map = {
            r['chunk_id']: r['bm25_score'] / bm25_max
            for r in bm25_results
        }
        
        # Combine
        all_chunk_ids = set(sem_map.keys()) | set(bm25_map.keys())
        chunk_map = {r['chunk_id']: r for r in semantic_results + bm25_results}
        
        combined = []
        for chunk_id in all_chunk_ids:
            sem_score = sem_map.get(chunk_id, 0.0)
            bm25_score = bm25_map.get(chunk_id, 0.0)
            
            hybrid_score = (semantic_weight * sem_score + 
                          (1 - semantic_weight) * bm25_score)
            
            chunk = chunk_map[chunk_id].copy()
            chunk['hybrid_score'] = hybrid_score
            chunk['semantic_score_norm'] = sem_score
            chunk['bm25_score_norm'] = bm25_score
            
            combined.append(chunk)
        
        return combined
```

### **Testing Hybrid:**
```python
# Initialize all components
semantic = RAGRetriever()  # From Day 1
bm25 = BM25Searcher(chunks)
hybrid = HybridRanker(semantic, bm25)

# Compare all three methods
query = "What is the H1B cap?"

print("=== SEMANTIC ONLY ===")
sem_results = semantic.retrieve(query, top_k=3)
for r in sem_results:
    print(f"{r['score']:.3f}: {r['text'][:100]}...")

print("\n=== BM25 ONLY ===")
bm25_results = bm25.search(query, top_k=3)
for r in bm25_results:
    print(f"{r['bm25_score']:.3f}: {r['text'][:100]}...")

print("\n=== HYBRID (RRF) ===")
hybrid_results = hybrid.search(query, top_k=3, fusion_method='rrf')
for r in hybrid_results:
    print(f"{r['hybrid_score']:.3f}: {r['text'][:100]}...")
```

---

## Part 4: Re-Ranking Layer (45 minutes)

### **What It Does:**
Takes top-10 from hybrid search and re-ranks by multiple quality signals.

### **Re-Ranking Factors:**

**1. Source Authority:**
```python
if 'uscis.gov' in url:
    authority_bonus = +0.15
elif 'state.gov' in url:
    authority_bonus = +0.10
else:
    authority_bonus = 0.0
```

**2. Chunk Completeness:**
```python
# Longer chunks likely more complete
if len(chunk) > 600:
    completeness_bonus = +0.10
elif len(chunk) > 400:
    completeness_bonus = +0.05
else:
    completeness_bonus = 0.0
```

**3. Answer Indicators:**
```python
# Chunk contains answer patterns?
answer_patterns = ['is', 'are', 'must', 'requires', r'\d+']
if has_answer_pattern(chunk):
    answer_bonus = +0.05
```

### **Implementation:**

**File:** `src/retrieval/reranker.py`

```python
class Reranker:
    """Re-rank hybrid results by quality signals"""
    
    def rerank(
        self, 
        results: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Apply re-ranking to improve top results
        
        Args:
            results: Hybrid search results with 'hybrid_score'
            query: Original query (for context)
        
        Returns:
            Re-ranked results with 'final_score'
        """
        reranked = []
        
        for result in results:
            # Start with hybrid score
            base_score = result['hybrid_score']
            
            # Calculate bonuses
            authority = self._authority_score(result.get('source_url', ''))
            completeness = self._completeness_score(result['text'])
            answer_quality = self._answer_quality_score(result['text'], query)
            
            # Final score
            final_score = base_score + authority + completeness + answer_quality
            
            result['final_score'] = final_score
            result['authority_bonus'] = authority
            result['completeness_bonus'] = completeness
            result['answer_bonus'] = answer_quality
            
            reranked.append(result)
        
        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        return reranked
    
    def _authority_score(self, url: str) -> float:
        """Bonus for authoritative sources"""
        if 'uscis.gov' in url:
            return 0.15
        elif 'state.gov' in url:
            return 0.10
        elif '.gov' in url:
            return 0.05
        return 0.0
    
    def _completeness_score(self, text: str) -> float:
        """Bonus for detailed chunks"""
        length = len(text)
        if length > 600:
            return 0.10
        elif length > 400:
            return 0.05
        return 0.0
    
    def _answer_quality_score(self, text: str, query: str) -> float:
        """Bonus if chunk likely contains answer"""
        score = 0.0
        
        # Contains numbers (good for factual queries)
        if re.search(r'\d+', text):
            score += 0.03
        
        # Contains answer verbs
        answer_verbs = ['is', 'are', 'must', 'requires', 'allows']
        if any(verb in text.lower() for verb in answer_verbs):
            score += 0.02
        
        return score
```

---

## Part 5: Tier Handlers (45 minutes)

### **Tier 1 Handler:**

**File:** `src/reasoning/tier1_handler.py`

```python
class Tier1Handler:
    """Handles fact-based questions via database lookup"""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def handle(self, query: str) -> Dict:
        """
        Look up fact in database
        
        Returns:
            {
                'answer': str,
                'source_url': str,
                'confidence': 1.0,
                'tier': 1,
                'method': 'database_lookup'
            }
        """
        # Try exact match first
        result = self._exact_lookup(query)
        
        if result:
            return {
                'answer': result['answer'],
                'source_url': result['source_url'],
                'confidence': 1.0,
                'tier': 1,
                'method': 'database_exact_match'
            }
        
        # Try fuzzy match
        result = self._fuzzy_lookup(query)
        
        if result:
            return {
                'answer': result['answer'],
                'source_url': result['source_url'],
                'confidence': 0.9,
                'tier': 1,
                'method': 'database_fuzzy_match'
            }
        
        # No match - return None (will fallback to Tier 2)
        return None
    
    def _exact_lookup(self, query: str) -> Optional[Dict]:
        """Exact question match"""
        self.cursor.execute(
            "SELECT answer, source_url FROM h1b_facts WHERE LOWER(question) = LOWER(?)",
            (query.strip(),)
        )
        result = self.cursor.fetchone()
        if result:
            return {'answer': result[0], 'source_url': result[1]}
        return None
    
    def _fuzzy_lookup(self, query: str) -> Optional[Dict]:
        """Fuzzy match using keyword overlap"""
        # Get all facts
        self.cursor.execute("SELECT question, answer, source_url FROM h1b_facts")
        facts = self.cursor.fetchall()
        
        # Score each fact by keyword overlap
        query_keywords = set(query.lower().split())
        best_match = None
        best_score = 0.0
        
        for fact_q, fact_a, fact_url in facts:
            fact_keywords = set(fact_q.lower().split())
            overlap = len(query_keywords & fact_keywords)
            score = overlap / max(len(query_keywords), len(fact_keywords))
            
            if score > best_score and score > 0.5:  # 50% overlap threshold
                best_score = score
                best_match = {'answer': fact_a, 'source_url': fact_url}
        
        return best_match
```

### **Tier 2 Handler:**

**File:** `src/reasoning/tier2_handler.py`

```python
class Tier2Handler:
    """Handles complex questions via hybrid RAG"""
    
    def __init__(self, hybrid_ranker, reranker, model_manager):
        self.hybrid_ranker = hybrid_ranker
        self.reranker = reranker
        self.model_manager = model_manager
    
    def handle(self, query: str, top_k: int = 3) -> Dict:
        """
        Process query through hybrid RAG pipeline
        
        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'confidence': float,
                'tier': 2,
                'method': 'hybrid_rag'
            }
        """
        # Step 1: Hybrid search
        candidates = self.hybrid_ranker.search(query, top_k=10)
        
        # Step 2: Re-rank
        reranked = self.reranker.rerank(candidates, query)
        
        # Step 3: Select top-k
        top_chunks = reranked[:top_k]
        
        # Step 4: Format context for LLM
        context = self._format_context(top_chunks)
        
        # Step 5: Generate response
        answer = self.model_manager.generate_response(query, context)
        
        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(top_chunks)
        
        return {
            'answer': answer,
            'sources': self._format_sources(top_chunks),
            'confidence': confidence,
            'tier': 2,
            'method': 'hybrid_rag',
            'retrieval_scores': {
                'semantic': [c.get('semantic_rank') for c in top_chunks],
                'bm25': [c.get('bm25_rank') for c in top_chunks],
                'hybrid': [c['hybrid_score'] for c in top_chunks],
                'final': [c['final_score'] for c in top_chunks]
            }
        }
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context for LLM"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk['source_title']}\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(context_parts)
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information for citation"""
        return [
            {
                'title': chunk['source_title'],
                'url': chunk['source_url'],
                'relevance': chunk['final_score'],
                'text_preview': chunk['text'][:200]
            }
            for chunk in chunks
        ]
    
    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """
        Estimate confidence based on retrieval quality
        
        High confidence: Top chunks have high scores + from official sources
        Low confidence: Low scores or disagreement
        """
        if not chunks:
            return 0.0
        
        # Average final score
        avg_score = sum(c['final_score'] for c in chunks) / len(chunks)
        
        # Source quality
        official_count = sum(1 for c in chunks if 'uscis.gov' in c['source_url'])
        source_quality = official_count / len(chunks)
        
        # Combined confidence
        confidence = 0.6 * avg_score + 0.4 * source_quality
        
        return min(1.0, confidence)
```

---

## Part 6: Unified Interface (30 minutes)

### **Master Class:**

**File:** `src/visa_rag.py`

```python
class VisaRAG:
    """
    Main interface for Visa Navigator RAG system
    
    Handles routing, retrieval, and response generation
    """
    
    def __init__(self, data_mode: str = 'official'):
        """
        Initialize RAG system
        
        Args:
            data_mode: 'official' (USCIS only) or 'extended' (multi-source)
        """
        self.data_mode = data_mode
        
        # Load components
        self.router = QueryRouter('data/structured/h1b_facts.db')
        self.tier1 = Tier1Handler('data/structured/h1b_facts.db')
        
        # Load chunks and build retrievers
        chunks = self._load_chunks(data_mode)
        semantic = RAGRetriever()
        bm25 = BM25Searcher(chunks)
        hybrid_ranker = HybridRanker(semantic, bm25)
        reranker = Reranker()
        model_manager = ModelManager()
        
        self.tier2 = Tier2Handler(hybrid_ranker, reranker, model_manager)
        
        # Logging for evaluation
        self.query_log = []
    
    def query(self, question: str) -> Dict:
        """
        Main query interface
        
        Args:
            question: User's question
        
        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'confidence': float,
                'tier_used': int,
                'method': str
            }
        """
        # Route query
        routing = self.router.route(question)
        
        # Try Tier 1 first if routed there
        if routing['tier'] == 1:
            result = self.tier1.handle(question)
            if result:
                # Log for evaluation
                self._log_query(question, result, routing)
                return result
        
        # Fallback to Tier 2 (or if routed there)
        result = self.tier2.handle(question)
        
        # Log for evaluation
        self._log_query(question, result, routing)
        
        return result
    
    def query_all_methods(self, question: str) -> Dict:
        """
        Run query through ALL methods for comparison
        
        Used for evaluation/ablation study
        
        Returns:
            {
                'semantic_only': {...},
                'bm25_only': {...},
                'hybrid': {...},
                'tier1': {...},
                'tier2': {...}
            }
        """
        # This will be used in Day 4 evaluation
        pass
    
    def _load_chunks(self, mode: str) -> List[Dict]:
        """Load appropriate chunk dataset"""
        if mode == 'official':
            with open('chunks_metadata.pkl', 'rb') as f:
                return pickle.load(f)
        elif mode == 'extended':
            # For Day 5 - multi-source feature
            with open('chunks_metadata_extended.pkl', 'rb') as f:
                return pickle.load(f)
    
    def _log_query(self, question: str, result: Dict, routing: Dict):
        """Log query for evaluation"""
        self.query_log.append({
            'timestamp': datetime.now(),
            'question': question,
            'routed_tier': routing['tier'],
            'actual_tier': result['tier'],
            'confidence': result['confidence'],
            'method': result['method']
        })
```

---

## Part 7: Testing & Validation (30 minutes)

### **Test Suite:**

**File:** `tests/test_tier_system.py`

```python
def test_tier1_routing():
    """Test that fact questions go to Tier 1"""
    rag = VisaRAG()
    
    fact_questions = [
        "What is the H1B cap?",
        "How much does H1B cost?",
        "What is the filing fee?",
    ]
    
    for q in fact_questions:
        result = rag.query(q)
        assert result['tier'] == 1, f"Expected Tier 1 for: {q}"
        assert result['confidence'] >= 0.9

def test_tier2_routing():
    """Test that complex questions go to Tier 2"""
    rag = VisaRAG()
    
    complex_questions = [
        "Am I eligible for H1B with CS degree?",
        "Can I change employers on H1B?",
        "How do I apply for H1B?",
    ]
    
    for q in complex_questions:
        result = rag.query(q)
        assert result['tier'] == 2, f"Expected Tier 2 for: {q}"
        assert len(result['sources']) > 0

def test_hybrid_vs_single():
    """Test that hybrid outperforms single methods"""
    # This validates our core claim
    
    test_query = "What is the H1B cap?"
    
    # Get results from all methods
    semantic_chunks = semantic_searcher.retrieve(test_query, top_k=3)
    bm25_chunks = bm25_searcher.search(test_query, top_k=3)
    hybrid_chunks = hybrid_ranker.search(test_query, top_k=3)
    
    # Check: Hybrid top result should contain "65,000"
    assert "65000" in hybrid_chunks[0]['text'] or "65,000" in hybrid_chunks[0]['text']
    
    print("✅ Hybrid correctly retrieved cap information")
```

### **Integration Test:**

```python
# test_integration.py

def main():
    print("🧪 Testing Visa RAG System\n")
    
    rag = VisaRAG()
    
    test_cases = [
        # Tier 1 tests
        ("What is the H1B cap?", 1, "65,000"),
        ("How much does H1B cost?", 1, "$"),
        
        # Tier 2 tests
        ("Am I eligible with CS degree?", 2, "bachelor"),
        ("Can I change employers?", 2, "transfer"),
    ]
    
    passed = 0
    for question, expected_tier, expected_keyword in test_cases:
        print(f"Q: {question}")
        result = rag.query(question)
        
        # Check tier
        tier_ok = result['tier'] == expected_tier
        
        # Check answer contains expected keyword
        keyword_ok = expected_keyword.lower() in result['answer'].lower()
        
        if tier_ok and keyword_ok:
            print(f"   ✅ Tier {result['tier']}, contains '{expected_keyword}'")
            passed += 1
        else:
            print(f"   ❌ Failed (tier={result['tier']}, keyword={keyword_ok})")
        
        print(f"   Answer: {result['answer'][:100]}...\n")
    
    print(f"\n{'='*50}")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
```

---

## Expected Outputs (End of Day 2)

### **Files Created:**
```
src/
├─ routing/
│  └─ query_router.py
├─ retrieval/
│  ├─ bm25_search.py
│  ├─ hybrid_ranker.py
│  └─ reranker.py
├─ reasoning/
│  ├─ tier1_handler.py
│  └─ tier2_handler.py
└─ visa_rag.py

tests/
├─ test_tier_system.py
└─ test_integration.py
```

### **System Capabilities:**
✅ Automatically routes questions to appropriate tier
✅ Tier 1: 100% accuracy on fact questions (via DB)
✅ Tier 2: Hybrid retrieval (semantic + BM25)
✅ Re-ranking by authority + completeness
✅ Unified interface: `rag.query(question)`
✅ Logging for evaluation

### **Performance Targets:**
- Tier 1 accuracy: 100% (database lookups)
- Tier 2 accuracy: 85%+ (hybrid RAG)
- Average response time: <3 seconds
- Citation coverage: 100% (all answers have sources)

---

## Success Criteria

✅ **Must Have:**
- Router correctly classifies 90%+ of test questions
- Tier 1 returns correct facts instantly
- Tier 2 retrieves relevant chunks (top-3 contain answer)
- System runs end-to-end without errors
- Can demonstrate improvement over single-method search

✅ **Nice to Have:**
- Query logging working (for Day 4 evaluation)
- Re-ranking provides measurable improvement
- Clean error handling (graceful degradation)

---

## Common Issues & Solutions

**Issue:** "BM25 scores much higher than semantic"
**Solution:** Scores are on different scales - use RRF fusion (rank-based)

**Issue:** "Router sends everything to Tier 2"
**Solution:** Check fact_patterns regex, might need tuning

**Issue:** "Hybrid worse than semantic alone"
**Solution:** Check BM25 tokenization, might need better preprocessing

**Issue:** "Re-ranking doesn't change order"
**Solution:** Bonuses too small, increase authority_bonus to 0.2

---

## Next Steps (Day 3)

After Day 2 completes, we'll add:
- Query enhancement (expand terms, handle abbreviations)
- Advanced re-ranking (recency scoring, completeness checks)
- Confidence scoring (high/medium/low reliability indicators)
- Citation validation (verify URLs, check text alignment)

---

## Notes for Claude Code

When implementing:

1. **Install BM25:**
   ```bash
   pip install rank-bm25
   ```

2. **Build in order:**
   - BM25 search (standalone)
   - Hybrid ranker (uses existing semantic + new BM25)
   - Re-ranker (uses hybrid output)
   - Router (standalone)
   - Tier handlers (use all above)
   - Unified interface (brings it together)

3. **Test incrementally:**
   - Test BM25 alone first
   - Test hybrid fusion
   - Test full pipeline

4. **Focus on working first, optimizing later:**
   - Get end-to-end working
   - Then tune fusion weights
   - Then optimize re-ranking bonuses

**The goal: A working 2-tier system that demonstrably outperforms single-method search.**
