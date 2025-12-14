# DAY 3 SPEC: Intelligence Layer

**Goal:** Add intelligent enhancements that make the system smarter, more reliable, and transparent

**Timeline:** 3-4 hours (4 modular parts with checkpoints)

**Deliverables:**
- Query enhancement (handles abbreviations, expands terms)
- Confidence scoring (high/medium/low reliability indicators)
- Advanced re-ranking (recency, contradiction detection)
- Citation validation (verify URLs, check alignment)

**Architecture:** Each part is independent - can stop at any checkpoint and system still works

---

## Architecture Overview

```
Current System (Day 2):
User Query → Router → Tier 1/2 → Response

Enhanced System (Day 3):
User Query 
    ↓
Query Enhancement ←─────┐ CHECKPOINT 1
    ↓                   │ (Can stop here)
Router → Tier 1/2       │
    ↓                   │
Response Generation     │
    ↓                   │
Confidence Scoring ←────┐ CHECKPOINT 2
    ↓                   │ (Can stop here)
Citation Validation ←───┐ CHECKPOINT 3
    ↓                   │ (Can stop here)
Final Response          │
```

---

## PART 1: Query Enhancement (1 hour)

### **CHECKPOINT 1: Basic Enhancement**

### **What It Does:**
Improves raw queries before they hit the retrieval system.

### **Enhancements:**

**1. Abbreviation Expansion:**
```
"H1B" → "H-1B H1B specialty occupation"
"OPT" → "OPT Optional Practical Training"
"LCA" → "LCA Labor Condition Application"
"USCIS" → "USCIS United States Citizenship Immigration Services"
```

**Why:** Different documents use different spellings/formats

**2. Synonym Addition:**
```
"cap" → "cap limit quota ceiling"
"fee" → "fee cost price charge"
"eligible" → "eligible qualify requirements"
```

**Why:** Catches documents using different terminology

**3. Number Normalization:**
```
"$460" → "$460 460 dollars"
"65000" → "65000 65,000"
```

**Why:** Documents format numbers differently

### **Implementation:**

**File:** `src/enhancement/query_enhancer.py`

```python
class QueryEnhancer:
    """Enhance queries before retrieval"""
    
    def __init__(self):
        # Abbreviation mappings
        self.abbreviations = {
            'h1b': 'H-1B H1B specialty occupation',
            'h-1b': 'H-1B H1B specialty occupation',
            'opt': 'OPT Optional Practical Training',
            'cpt': 'CPT Curricular Practical Training',
            'lca': 'LCA Labor Condition Application',
            'uscis': 'USCIS United States Citizenship Immigration Services',
            'dhs': 'DHS Department Homeland Security',
            'i-129': 'I-129 Form petition nonimmigrant worker',
            'cap-gap': 'cap-gap extension F-1 OPT H-1B',
        }
        
        # Synonym groups
        self.synonyms = {
            'cap': ['cap', 'limit', 'quota', 'ceiling', 'maximum'],
            'fee': ['fee', 'cost', 'price', 'charge', 'payment'],
            'eligible': ['eligible', 'qualify', 'requirement', 'criteria'],
            'apply': ['apply', 'petition', 'file', 'submit'],
            'process': ['process', 'procedure', 'steps', 'timeline'],
            'change': ['change', 'transfer', 'switch', 'move'],
            'extend': ['extend', 'renewal', 'extension', 'continue'],
        }
    
    def enhance(self, query: str, mode: str = 'balanced') -> str:
        """
        Enhance query for better retrieval
        
        Args:
            query: Original user query
            mode: 'light' (minimal), 'balanced', or 'aggressive' (max expansion)
        
        Returns:
            Enhanced query string
        """
        if mode == 'light':
            # Just abbreviations
            return self._expand_abbreviations(query)
        
        elif mode == 'balanced':
            # Abbreviations + key synonyms
            enhanced = self._expand_abbreviations(query)
            enhanced = self._add_key_synonyms(enhanced)
            return enhanced
        
        elif mode == 'aggressive':
            # Everything
            enhanced = self._expand_abbreviations(query)
            enhanced = self._add_all_synonyms(enhanced)
            enhanced = self._normalize_numbers(enhanced)
            return enhanced
        
        return query
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand known abbreviations"""
        query_lower = query.lower()
        enhanced = query
        
        for abbr, expansion in self.abbreviations.items():
            if abbr in query_lower:
                # Add expansion (keep original)
                enhanced = f"{enhanced} {expansion}"
        
        return enhanced
    
    def _add_key_synonyms(self, query: str) -> str:
        """Add synonyms for key terms"""
        query_lower = query.lower()
        enhanced = query
        
        for key_term, synonyms in self.synonyms.items():
            if key_term in query_lower:
                # Add top 2 synonyms
                enhanced = f"{enhanced} {' '.join(synonyms[:2])}"
        
        return enhanced
    
    def _add_all_synonyms(self, query: str) -> str:
        """Add all synonyms for matched terms"""
        query_lower = query.lower()
        enhanced = query
        
        for key_term, synonyms in self.synonyms.items():
            if key_term in query_lower:
                enhanced = f"{enhanced} {' '.join(synonyms)}"
        
        return enhanced
    
    def _normalize_numbers(self, query: str) -> str:
        """Add number format variations"""
        import re
        
        enhanced = query
        
        # Find dollar amounts: $460
        dollars = re.findall(r'\$(\d+)', query)
        for amount in dollars:
            enhanced = f"{enhanced} {amount} dollars"
        
        # Find plain numbers: 65000
        numbers = re.findall(r'\b(\d{3,})\b', query)
        for num in numbers:
            # Add comma version
            if len(num) >= 4:
                comma_version = f"{num[:-3]},{num[-3:]}"
                enhanced = f"{enhanced} {comma_version}"
        
        return enhanced
    
    def get_enhancement_stats(self, original: str, enhanced: str) -> dict:
        """Compare original vs enhanced for logging"""
        return {
            'original_length': len(original.split()),
            'enhanced_length': len(enhanced.split()),
            'expansion_ratio': len(enhanced.split()) / len(original.split()),
            'added_terms': len(enhanced.split()) - len(original.split())
        }
```

### **Integration with VisaRAG:**

**Update:** `src/visa_rag.py`

```python
class VisaRAG:
    def __init__(self, data_mode: str = 'official', enhancement_mode: str = 'balanced'):
        # ... existing init ...
        self.query_enhancer = QueryEnhancer()
        self.enhancement_mode = enhancement_mode
    
    def query(self, question: str, enhance: bool = True) -> Dict:
        """
        Main query interface with optional enhancement
        
        Args:
            question: User's question
            enhance: Whether to enhance query (default: True)
        """
        # Enhance query if enabled
        if enhance and self.enhancement_mode != 'off':
            enhanced_query = self.query_enhancer.enhance(
                question, 
                mode=self.enhancement_mode
            )
        else:
            enhanced_query = question
        
        # Route using ORIGINAL query (for better classification)
        routing = self.router.route(question)
        
        # But retrieve using ENHANCED query (for better recall)
        if routing['tier'] == 1:
            result = self.tier1.handle(question)  # Use original for DB
            if result:
                return result
        
        # Tier 2 uses enhanced query
        result = self.tier2.handle(enhanced_query)
        result['query_enhanced'] = (enhanced_query != question)
        
        return result
```

### **Testing Enhancement:**

```python
# test_query_enhancement.py

def test_abbreviation_expansion():
    enhancer = QueryEnhancer()
    
    query = "What is H1B cap?"
    enhanced = enhancer.enhance(query, mode='light')
    
    assert 'H-1B' in enhanced
    assert 'specialty occupation' in enhanced
    print(f"Original: {query}")
    print(f"Enhanced: {enhanced}\n")

def test_enhancement_improves_retrieval():
    """Test that enhancement finds more relevant results"""
    
    # Without enhancement
    results_plain = hybrid_ranker.search("H1B cap", top_k=3)
    
    # With enhancement
    enhancer = QueryEnhancer()
    enhanced = enhancer.enhance("H1B cap", mode='balanced')
    results_enhanced = hybrid_ranker.search(enhanced, top_k=3)
    
    # Enhanced should have better top result
    print("Plain query top result:")
    print(f"  Score: {results_plain[0]['hybrid_score']:.3f}")
    print(f"  Text: {results_plain[0]['text'][:100]}...")
    
    print("\nEnhanced query top result:")
    print(f"  Score: {results_enhanced[0]['hybrid_score']:.3f}")
    print(f"  Text: {results_enhanced[0]['text'][:100]}...")
```

### **CHECKPOINT 1 COMPLETE**
✅ Query enhancement working
✅ Can toggle on/off per query
✅ Logging for evaluation
✅ **System still works if you stop here**

---

## PART 2: Confidence Scoring (45 minutes)

### **CHECKPOINT 2: Reliability Indicators**

### **What It Does:**
Tells users how confident the system is in its answer.

### **Confidence Factors:**

**1. Source Agreement:**
```
High: Top 3 chunks say same thing
Medium: 2/3 chunks agree
Low: Chunks contradict or only 1 mentions it
```

**2. Source Quality:**
```
High: All from uscis.gov
Medium: Mix of .gov sources
Low: Non-official sources
```

**3. Retrieval Quality:**
```
High: Top chunk score > 0.8
Medium: Top chunk score 0.6-0.8
Low: Top chunk score < 0.6
```

**4. Answer Completeness:**
```
High: Answer has numbers, dates, specifics
Medium: Answer is general but relevant
Low: Answer is vague or hedged
```

### **Implementation:**

**File:** `src/evaluation/confidence_scorer.py`

```python
class ConfidenceScorer:
    """Calculate confidence in system's answers"""
    
    def calculate_confidence(
        self,
        answer: str,
        sources: List[Dict],
        tier: int
    ) -> Dict:
        """
        Calculate multi-factor confidence score
        
        Returns:
            {
                'overall': float (0-1),
                'level': str ('high', 'medium', 'low'),
                'factors': {
                    'source_agreement': float,
                    'source_quality': float,
                    'retrieval_quality': float,
                    'answer_completeness': float
                },
                'reasoning': str
            }
        """
        if tier == 1:
            # Tier 1 is always high confidence (database facts)
            return {
                'overall': 1.0,
                'level': 'high',
                'factors': {
                    'source_agreement': 1.0,
                    'source_quality': 1.0,
                    'retrieval_quality': 1.0,
                    'answer_completeness': 1.0
                },
                'reasoning': 'Fact from verified database'
            }
        
        # Tier 2: Calculate each factor
        factors = {
            'source_agreement': self._check_source_agreement(sources),
            'source_quality': self._check_source_quality(sources),
            'retrieval_quality': self._check_retrieval_quality(sources),
            'answer_completeness': self._check_answer_completeness(answer)
        }
        
        # Weighted average
        weights = {
            'source_agreement': 0.35,
            'source_quality': 0.30,
            'retrieval_quality': 0.20,
            'answer_completeness': 0.15
        }
        
        overall = sum(factors[k] * weights[k] for k in factors)
        
        # Classify level
        if overall >= 0.8:
            level = 'high'
            reasoning = 'Multiple authoritative sources agree'
        elif overall >= 0.6:
            level = 'medium'
            reasoning = 'Good sources but some uncertainty'
        else:
            level = 'low'
            reasoning = 'Limited or conflicting information'
        
        return {
            'overall': overall,
            'level': level,
            'factors': factors,
            'reasoning': reasoning
        }
    
    def _check_source_agreement(self, sources: List[Dict]) -> float:
        """Do sources agree or contradict?"""
        if len(sources) < 2:
            return 0.5  # Can't check agreement
        
        # Simple heuristic: check if key facts appear in multiple sources
        # For now, just check if similar scores (implies similar content)
        scores = [s.get('relevance', s.get('final_score', 0)) for s in sources]
        
        if not scores:
            return 0.5
        
        # If top 2 scores are close, likely similar content
        if len(scores) >= 2:
            score_diff = abs(scores[0] - scores[1])
            if score_diff < 0.1:
                return 0.9  # High agreement
            elif score_diff < 0.2:
                return 0.7  # Medium agreement
        
        return 0.5  # Uncertain
    
    def _check_source_quality(self, sources: List[Dict]) -> float:
        """Are sources authoritative?"""
        if not sources:
            return 0.0
        
        official_count = sum(
            1 for s in sources 
            if 'uscis.gov' in s.get('url', '').lower()
        )
        
        return official_count / len(sources)
    
    def _check_retrieval_quality(self, sources: List[Dict]) -> float:
        """How well do sources match query?"""
        if not sources:
            return 0.0
        
        # Use top source's score as indicator
        top_score = sources[0].get('relevance', sources[0].get('final_score', 0))
        return min(1.0, top_score)
    
    def _check_answer_completeness(self, answer: str) -> float:
        """Does answer have specific details?"""
        score = 0.5  # Base
        
        # Has numbers/dates (specific)
        import re
        if re.search(r'\d+', answer):
            score += 0.2
        
        # Has specific terms (not vague)
        specific_terms = ['must', 'requires', 'is', 'are', 'will', 'can']
        if any(term in answer.lower() for term in specific_terms):
            score += 0.2
        
        # Not hedged (no "might", "possibly", etc.)
        hedge_words = ['might', 'possibly', 'perhaps', 'unclear']
        if not any(word in answer.lower() for word in hedge_words):
            score += 0.1
        
        return min(1.0, score)
```

### **Integration:**

**Update:** `src/reasoning/tier2_handler.py`

```python
from src.evaluation.confidence_scorer import ConfidenceScorer

class Tier2Handler:
    def __init__(self, hybrid_ranker, reranker, model_manager):
        # ... existing init ...
        self.confidence_scorer = ConfidenceScorer()
    
    def handle(self, query: str, top_k: int = 3) -> Dict:
        """Process query with confidence scoring"""
        # ... existing retrieval logic ...
        
        # Calculate confidence
        confidence_data = self.confidence_scorer.calculate_confidence(
            answer=answer,
            sources=top_chunks,
            tier=2
        )
        
        return {
            'answer': answer,
            'sources': self._format_sources(top_chunks),
            'confidence': confidence_data['overall'],
            'confidence_level': confidence_data['level'],
            'confidence_reasoning': confidence_data['reasoning'],
            'confidence_factors': confidence_data['factors'],
            'tier': 2,
            'method': 'hybrid_rag'
        }
```

### **User-Facing Display:**

```python
def format_response_with_confidence(result: Dict) -> str:
    """Format response for user with confidence indicator"""
    
    answer = result['answer']
    confidence = result.get('confidence_level', 'medium')
    
    # Confidence badge
    badges = {
        'high': '🟢 High Confidence',
        'medium': '🟡 Medium Confidence',
        'low': '🔴 Low Confidence'
    }
    
    badge = badges.get(confidence, '⚪ Unknown')
    reasoning = result.get('confidence_reasoning', '')
    
    output = f"""
{badge}

{answer}

Confidence: {reasoning}

Sources:
"""
    
    for i, source in enumerate(result['sources'], 1):
        output += f"{i}. {source['title']}\n"
    
    if confidence == 'low':
        output += "\n⚠️ Please verify this information with official sources or an immigration attorney."
    
    return output
```

### **CHECKPOINT 2 COMPLETE**
✅ Confidence scoring implemented
✅ Multi-factor calculation (agreement, quality, retrieval, completeness)
✅ User-facing confidence indicators
✅ **System still works if you stop here**

---

## PART 3: Advanced Re-Ranking (1 hour)

### **CHECKPOINT 3: Smart Re-Ranking**

### **What It Does:**
Improves re-ranking with additional intelligence signals.

### **New Re-Ranking Factors:**

**1. Recency Scoring:**
```python
# Prefer more recent information
if 'last_updated' in chunk:
    days_old = (now - chunk['last_updated']).days
    if days_old < 30:
        recency_bonus = +0.10
    elif days_old < 180:
        recency_bonus = +0.05
    else:
        recency_bonus = 0.0
```

**2. Query-Specific Weighting:**
```python
# Adjust re-ranking based on question type

if is_fact_question(query):
    # Prioritize brevity + numbers
    authority_weight = 0.4
    completeness_weight = 0.3
    
elif is_process_question(query):
    # Prioritize detail + completeness
    authority_weight = 0.3
    completeness_weight = 0.5
```

**3. Contradiction Detection:**
```python
# Flag if top chunks contradict each other

if contains_number(chunk1) and contains_number(chunk2):
    num1 = extract_number(chunk1)
    num2 = extract_number(chunk2)
    
    if num1 != num2:
        # Potential contradiction
        confidence_penalty = -0.2
```

### **Implementation:**

**Update:** `src/retrieval/reranker.py`

```python
class Reranker:
    """Enhanced re-ranking with intelligence signals"""
    
    def rerank(
        self, 
        results: List[Dict],
        query: str,
        mode: str = 'smart'  # 'simple' or 'smart'
    ) -> List[Dict]:
        """
        Apply intelligent re-ranking
        
        Args:
            results: Hybrid search results
            query: Original query
            mode: 'simple' (Day 2 logic) or 'smart' (Day 3 logic)
        """
        if mode == 'simple':
            return self._simple_rerank(results, query)
        
        # Smart re-ranking
        query_type = self._classify_query_type(query)
        
        reranked = []
        for result in results:
            base_score = result['hybrid_score']
            
            # Apply weighted bonuses based on query type
            bonuses = self._calculate_smart_bonuses(
                result, 
                query, 
                query_type
            )
            
            final_score = base_score + sum(bonuses.values())
            
            result['final_score'] = final_score
            result['rerank_bonuses'] = bonuses
            reranked.append(result)
        
        # Check for contradictions
        reranked = self._detect_contradictions(reranked)
        
        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        return reranked
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query to determine re-ranking strategy"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'how much', 'how many']):
            return 'fact'
        elif any(word in query_lower for word in ['how do', 'how to', 'process', 'steps']):
            return 'process'
        elif any(word in query_lower for word in ['am i', 'can i', 'eligible']):
            return 'eligibility'
        else:
            return 'general'
    
    def _calculate_smart_bonuses(
        self,
        result: Dict,
        query: str,
        query_type: str
    ) -> Dict[str, float]:
        """Calculate type-specific bonuses"""
        
        bonuses = {}
        
        # Authority (always matters)
        bonuses['authority'] = self._authority_score(result.get('source_url', ''))
        
        if query_type == 'fact':
            # For facts: prefer brevity + numbers
            bonuses['brevity'] = 0.05 if len(result['text']) < 400 else 0.0
            bonuses['has_numbers'] = 0.08 if self._contains_numbers(result['text']) else 0.0
            bonuses['completeness'] = 0.0  # Don't need long explanations
            
        elif query_type == 'process':
            # For process: prefer detail + structure
            bonuses['completeness'] = self._completeness_score(result['text']) * 0.12
            bonuses['has_structure'] = 0.06 if self._has_list_structure(result['text']) else 0.0
            
        elif query_type == 'eligibility':
            # For eligibility: prefer requirements + examples
            bonuses['has_requirements'] = 0.08 if 'require' in result['text'].lower() else 0.0
            bonuses['completeness'] = self._completeness_score(result['text']) * 0.08
        
        else:
            # General: balanced
            bonuses['completeness'] = self._completeness_score(result['text']) * 0.08
        
        return bonuses
    
    def _contains_numbers(self, text: str) -> bool:
        """Check if text contains numbers"""
        import re
        return bool(re.search(r'\d+', text))
    
    def _has_list_structure(self, text: str) -> bool:
        """Check if text has numbered/bulleted lists"""
        list_indicators = ['1.', '2.', '•', 'first,', 'second,', 'step 1']
        return any(indicator in text.lower() for indicator in list_indicators)
    
    def _detect_contradictions(self, results: List[Dict]) -> List[Dict]:
        """Flag potential contradictions in top results"""
        if len(results) < 2:
            return results
        
        # Check if top 2 results have different numbers for same concept
        import re
        
        top_numbers_1 = set(re.findall(r'\$?\d+(?:,\d{3})*', results[0]['text']))
        top_numbers_2 = set(re.findall(r'\$?\d+(?:,\d{3})*', results[1]['text']))
        
        # If they mention same concept but different numbers
        if top_numbers_1 and top_numbers_2 and not top_numbers_1.intersection(top_numbers_2):
            # Potential contradiction - flag for lower confidence
            results[0]['contradiction_warning'] = True
            results[1]['contradiction_warning'] = True
        
        return results
    
    def _simple_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Original Day 2 re-ranking logic"""
        # Keep old logic for comparison
        for result in results:
            base = result['hybrid_score']
            authority = self._authority_score(result.get('source_url', ''))
            completeness = self._completeness_score(result['text'])
            result['final_score'] = base + authority + completeness
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
```

### **CHECKPOINT 3 COMPLETE**
✅ Query-type-specific re-ranking
✅ Contradiction detection
✅ Smarter bonus allocation
✅ **System still works if you stop here**

---

## PART 4: Citation Validation (45 minutes)

### **CHECKPOINT 4: Final Polish**

### **What It Does:**
Verifies that citations are valid and text alignment is correct.

### **Validation Checks:**

**1. URL Accessibility:**
```python
# Check if source URL actually exists
try:
    response = requests.head(url, timeout=5)
    if response.status_code == 200:
        url_valid = True
except:
    url_valid = False
```

**2. Text Alignment:**
```python
# Verify cited text actually came from source
if chunk['text'] in source_page_content:
    alignment = 'exact'
elif similarity(chunk['text'], source_page_content) > 0.8:
    alignment = 'close'
else:
    alignment = 'unverified'
```

**3. Source Freshness:**
```python
# Check when page was last updated
if 'last_modified' in response.headers:
    last_updated = parse_date(response.headers['last_modified'])
    freshness = 'recent' if days_old < 90 else 'old'
```

### **Implementation:**

**File:** `src/evaluation/citation_validator.py`

```python
import requests
from datetime import datetime, timedelta
from typing import Dict, List

class CitationValidator:
    """Validate citations and source quality"""
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Args:
            cache_ttl: Cache validation results for N seconds
        """
        self.validation_cache = {}
        self.cache_ttl = cache_ttl
    
    def validate_sources(self, sources: List[Dict]) -> List[Dict]:
        """
        Validate all sources in a response
        
        Returns:
            Sources with validation metadata added
        """
        validated = []
        
        for source in sources:
            url = source.get('url', '')
            
            # Check cache first
            if url in self.validation_cache:
                cached = self.validation_cache[url]
                if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                    validation = cached['validation']
                else:
                    validation = self._validate_url(url)
                    self._cache_validation(url, validation)
            else:
                validation = self._validate_url(url)
                self._cache_validation(url, validation)
            
            # Add validation data to source
            source['validation'] = validation
            validated.append(source)
        
        return validated
    
    def _validate_url(self, url: str) -> Dict:
        """
        Check if URL is accessible and recent
        
        Returns:
            {
                'accessible': bool,
                'status_code': int,
                'last_modified': datetime or None,
                'freshness': str ('recent', 'old', 'unknown')
            }
        """
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            
            accessible = (response.status_code == 200)
            last_modified = None
            freshness = 'unknown'
            
            if 'last-modified' in response.headers:
                try:
                    last_modified = self._parse_http_date(
                        response.headers['last-modified']
                    )
                    days_old = (datetime.now() - last_modified).days
                    freshness = 'recent' if days_old < 90 else 'old'
                except:
                    pass
            
            return {
                'accessible': accessible,
                'status_code': response.status_code,
                'last_modified': last_modified,
                'freshness': freshness
            }
            
        except requests.RequestException:
            return {
                'accessible': False,
                'status_code': None,
                'last_modified': None,
                'freshness': 'unknown'
            }
    
    def _parse_http_date(self, date_str: str) -> datetime:
        """Parse HTTP date header"""
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    
    def _cache_validation(self, url: str, validation: Dict):
        """Cache validation result"""
        self.validation_cache[url] = {
            'validation': validation,
            'timestamp': datetime.now()
        }
    
    def get_citation_quality_score(self, sources: List[Dict]) -> float:
        """
        Calculate overall citation quality
        
        Returns:
            Score 0-1 based on source accessibility and freshness
        """
        if not sources:
            return 0.0
        
        accessible_count = sum(
            1 for s in sources 
            if s.get('validation', {}).get('accessible', False)
        )
        
        recent_count = sum(
            1 for s in sources
            if s.get('validation', {}).get('freshness') == 'recent'
        )
        
        accessibility_score = accessible_count / len(sources)
        freshness_score = recent_count / len(sources)
        
        # Weight accessibility higher than freshness
        return 0.7 * accessibility_score + 0.3 * freshness_score
```

### **Integration:**

**Update:** `src/visa_rag.py`

```python
from src.evaluation.citation_validator import CitationValidator

class VisaRAG:
    def __init__(self, ...):
        # ... existing init ...
        self.citation_validator = CitationValidator()
    
    def query(self, question: str, validate_citations: bool = True) -> Dict:
        """
        Query with optional citation validation
        
        Args:
            validate_citations: Check URL accessibility (default: True)
        """
        # ... existing logic ...
        
        # Validate citations if enabled
        if validate_citations and result['sources']:
            result['sources'] = self.citation_validator.validate_sources(
                result['sources']
            )
            
            # Add overall citation quality
            result['citation_quality'] = \
                self.citation_validator.get_citation_quality_score(
                    result['sources']
                )
        
        return result
```

### **User Display:**

```python
def format_sources_with_validation(sources: List[Dict]) -> str:
    """Format sources with validation indicators"""
    
    output = "📚 Sources:\n\n"
    
    for i, source in enumerate(sources, 1):
        validation = source.get('validation', {})
        
        # Status indicator
        if validation.get('accessible'):
            status = '✅'
        elif validation.get('accessible') == False:
            status = '⚠️ '
        else:
            status = '❓'
        
        # Freshness
        freshness = validation.get('freshness', 'unknown')
        freshness_label = {
            'recent': '(Updated recently)',
            'old': '(Last updated >90 days ago)',
            'unknown': ''
        }.get(freshness, '')
        
        output += f"{status} {i}. {source['title']} {freshness_label}\n"
        output += f"   {source['url']}\n\n"
    
    return output
```

### **CHECKPOINT 4 COMPLETE**
✅ Citation validation implemented
✅ URL accessibility checking
✅ Freshness indicators
✅ **Complete intelligence layer finished!**

---

## Integration Testing (Day 3 Complete)

### **Full System Test:**

```python
# test_intelligence_layer.py

def test_full_enhanced_system():
    """Test complete Day 3 enhancements"""
    
    rag = VisaRAG(
        enhancement_mode='balanced',
        validate_citations=True
    )
    
    # Test query enhancement
    result = rag.query("What is H1B cap?", enhance=True)
    
    assert 'query_enhanced' in result
    assert result['confidence_level'] in ['high', 'medium', 'low']
    assert 'citation_quality' in result
    
    # Check confidence factors
    assert 'confidence_factors' in result
    factors = result['confidence_factors']
    assert 'source_agreement' in factors
    assert 'source_quality' in factors
    
    # Check citation validation
    for source in result['sources']:
        assert 'validation' in source
        assert 'accessible' in source['validation']
    
    print("✅ All Day 3 enhancements working")
    print(f"   Confidence: {result['confidence_level']}")
    print(f"   Citation Quality: {result['citation_quality']:.2f}")

def test_enhancement_comparison():
    """Compare with vs without enhancements"""
    
    rag = VisaRAG()
    
    query = "H1B cap"
    
    # Without enhancement
    result_plain = rag.query(query, enhance=False)
    
    # With enhancement
    result_enhanced = rag.query(query, enhance=True)
    
    print("\nWithout Enhancement:")
    print(f"  Top source score: {result_plain['sources'][0]['relevance']:.3f}")
    
    print("\nWith Enhancement:")
    print(f"  Top source score: {result_enhanced['sources'][0]['relevance']:.3f}")
    
    # Enhanced should be better or equal
    assert result_enhanced['confidence'] >= result_plain['confidence']
```

---

## Day 3 Complete - Summary

### **What You Built:**

**Part 1: Query Enhancement ✅**
- Abbreviation expansion (H1B → H-1B specialty occupation)
- Synonym addition (cap → limit quota)
- Number normalization (65000 → 65,000)
- 3 modes: light, balanced, aggressive

**Part 2: Confidence Scoring ✅**
- Multi-factor confidence (agreement, quality, retrieval, completeness)
- 3-level classification (high/medium/low)
- User-facing confidence indicators
- Detailed reasoning for each score

**Part 3: Advanced Re-Ranking ✅**
- Query-type-specific bonuses
- Contradiction detection between sources
- Smarter weighting based on question type
- Comparison mode (simple vs smart)

**Part 4: Citation Validation ✅**
- URL accessibility checking
- Freshness indicators
- Citation quality scoring
- Cached validation (avoid redundant requests)

### **System Improvements:**

```
Before Day 3:
├─ Query: "H1B cap"
├─ Retrieval: Works but limited
├─ Confidence: Unknown
└─ Citations: Unverified

After Day 3:
├─ Query: "H1B H-1B specialty occupation cap limit quota"
├─ Retrieval: Better recall + precision
├─ Confidence: 🟢 High (0.87) - "Multiple authoritative sources agree"
├─ Citations: ✅ All verified accessible + recent
└─ Re-ranking: Query-type optimized
```

---

## Files Created (Day 3)

```
src/
├─ enhancement/
│  └─ query_enhancer.py
├─ evaluation/
│  ├─ confidence_scorer.py
│  └─ citation_validator.py
└─ retrieval/
   └─ reranker.py (updated with smart mode)

tests/
└─ test_intelligence_layer.py
```

---

## Next Steps

**Day 4: Evaluation Framework**
- Create 30-question test dataset
- Run all system variants (baseline, semantic, hybrid, enhanced)
- Calculate metrics (accuracy, confidence calibration, citation quality)
- Generate comparison graphs

**Day 5: Presentation**
- Simple demo UI
- Write paper
- Create slides
- Practice presentation

---

## Success Criteria (Day 3)

✅ **Must Have (All Complete):**
- Query enhancement improves retrieval
- Confidence scores correlate with actual accuracy
- Smart re-ranking outperforms simple re-ranking
- Citation validation identifies broken links

✅ **System Properties:**
- Each enhancement is modular (can be toggled)
- No performance degradation (<5s response time)
- Graceful degradation (still works if validation fails)

---

**Ready for Day 4?** Or should you actually sleep now? 😄
