# -*- coding: utf-8 -*-
"""
MedLink AI — Hallucination Detection Module
Detects when the LLM generates information not supported by retrieved drug data.
"""
import re
import difflib


def extract_key_facts(text: str) -> list:
    """Extract key medical facts/numbers from text for comparison."""
    facts = []
    
    # Extract dosages (numbers + mg/mcg/g/ml)
    dosage_patterns = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|µg|units?|IU)', text, re.IGNORECASE)
    facts.extend([p.lower().strip() for p in dosage_patterns])
    
    # Extract drug names mentioned
    drug_pattern = re.findall(r'\b[A-Z][a-z]+(?:in|ol|am|ate|ide|ine|one|fen|pam|zam|xin|cin)\b', text)
    facts.extend([d.lower() for d in drug_pattern])
    
    # Extract percentages
    pct_patterns = re.findall(r'\d+(?:\.\d+)?%', text)
    facts.extend(pct_patterns)
    
    return list(set(facts))


def calculate_factual_overlap(llm_response: str, source_documents: list) -> float:
    """
    Calculate how much of the LLM response is backed by source documents.
    Returns a score between 0.0 (full hallucination) and 1.0 (fully grounded).
    """
    if not llm_response or not source_documents:
        return 0.5  # Unknown
    
    # Combine all source text — handle both strings and dicts
    parts = []
    for doc in source_documents:
        if isinstance(doc, str):
            parts.append(doc)
        elif isinstance(doc, dict):
            parts.append(doc.get('document', '') or doc.get('text', '') or str(doc))
    source_text = " ".join(parts).lower()
    llm_lower = llm_response.lower()
    
    # Method 1: Key fact extraction
    llm_facts = extract_key_facts(llm_response)
    
    if not llm_facts:
        # No specific facts to check — use text similarity
        similarity = difflib.SequenceMatcher(None, llm_lower[:500], source_text[:500]).ratio()
        return min(0.9, similarity * 1.5)  # Scale up since general text is harder to match
    
    # Count how many LLM facts appear in source
    matched_facts = sum(1 for fact in llm_facts if fact in source_text)
    fact_score = matched_facts / len(llm_facts) if llm_facts else 0.5
    
    # Method 2: Key sentence overlap
    llm_sentences = [s.strip() for s in re.split(r'[.!?\n]', llm_response) if len(s.strip()) > 20]
    sentence_scores = []
    
    for sentence in llm_sentences[:5]:  # Check first 5 sentences
        sentence_lower = sentence.lower()
        # Check if key words from sentence appear in source
        words = [w for w in sentence_lower.split() if len(w) > 5]
        if words:
            word_matches = sum(1 for w in words if w in source_text)
            sentence_scores.append(word_matches / len(words))
    
    sentence_score = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.5
    
    # Combined score
    final_score = (fact_score * 0.6) + (sentence_score * 0.4)
    return round(min(1.0, final_score), 2)


def detect_hallucination(llm_response: str, retrieved_docs: list, groq_key: str = '') -> dict:
    """
    Main hallucination detection function.
    Returns:
        {
          'confidence': 0.0-1.0,
          'is_reliable': bool,
          'warning_level': 'none' | 'low' | 'medium' | 'high',
          'warning_message': str,
          'grounded_score': float
        }
    """
    if not llm_response:
        return {
            'confidence': 0.0,
            'is_reliable': False,
            'warning_level': 'high',
            'warning_message': 'لم يتم توليد رد من النموذج.',
            'grounded_score': 0.0
        }
    
    # Calculate grounding score
    grounded_score = calculate_factual_overlap(llm_response, retrieved_docs)
    
    # Check for hallucination red flags
    red_flags = [
        # Fabricated confidence phrases
        r'guaranteed|مضمون ١٠٠٪|دائماً آمن|لا آثار جانبية',
        # Overly specific fake data
        r'\b(?:99|100)% effective\b',
        # Contradictory safety claims
        r'safe for everyone|آمن للجميع بدون استثناء',
    ]
    
    has_red_flag = any(re.search(pattern, llm_response, re.IGNORECASE) for pattern in red_flags)
    
    if has_red_flag:
        grounded_score = min(grounded_score, 0.3)
    
    # Determine warning level
    if grounded_score >= 0.75:
        warning_level = 'none'
        warning_message = ''
        is_reliable = True
    elif grounded_score >= 0.55:
        warning_level = 'low'
        warning_message = '💡 بعض التفاصيل قد تحتاج تحقق إضافي من صيدلاني.'
        is_reliable = True
    elif grounded_score >= 0.35:
        warning_level = 'medium'
        warning_message = '⚠️ يُنصح بمراجعة صيدلاني أو طبيب للتأكد من هذه المعلومات.'
        is_reliable = False
    else:
        warning_level = 'high'
        warning_message = '🚫 هذه المعلومات قد لا تكون دقيقة. يرجى استشارة طبيب أو صيدلاني متخصص.'
        is_reliable = False
    
    return {
        'confidence': grounded_score,
        'is_reliable': is_reliable,
        'warning_level': warning_level,
        'warning_message': warning_message,
        'grounded_score': grounded_score
    }


def format_confidence_badge(score: float) -> dict:
    """Return badge color and text for confidence score."""
    pct = int(score * 100)
    if score >= 0.75:
        return {'color': '#22c55e', 'text': f'{pct}%', 'label': 'موثوق'}
    elif score >= 0.55:
        return {'color': '#f59e0b', 'text': f'{pct}%', 'label': 'مقبول'}
    elif score >= 0.35:
        return {'color': '#f97316', 'text': f'{pct}%', 'label': 'تحقق مطلوب'}
    else:
        return {'color': '#ef4444', 'text': f'{pct}%', 'label': 'تحقق ضروري'}


if __name__ == '__main__':
    # Test
    test_response = "جرعة الأموكسيسيلين للبالغين هي 500mg كل 8 ساعات. تجنب في حالة حساسية البنسلين."
    test_docs = [{'document': 'Adults: 250-500mg every 8 hours. Contraindicated in penicillin allergy.'}]
    result = detect_hallucination(test_response, test_docs)
    print(f"Hallucination Detection Result: {result}")
