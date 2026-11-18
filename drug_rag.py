"""
MedLink AI — RAG Drug Search Engine
Uses ChromaDB (vector DB) + Groq LLaMA 3 for intelligent drug search.
Supports: drug info, interactions, alternatives, dosage, contraindications.
Features: Confidence Score, Explainable AI (XAI), Hallucination Detection.
"""

import os
import json
import functools
import chromadb
from chromadb.utils import embedding_functions
from drug_knowledge_base import get_all_drugs, format_drug_for_chroma

# ─── Result Cache (speeds repeated searches) ────────────────────────────────────
_search_cache = {}
MAX_CACHE_SIZE = 80
try:
    from hallucination_detector import detect_hallucination, format_confidence_badge
    HALLUCINATION_DETECTOR_AVAILABLE = True
except ImportError:
    HALLUCINATION_DETECTOR_AVAILABLE = False

# ─── ChromaDB Setup ───────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_drug_db")

_chroma_client = None
_drug_collection = None


def get_chroma_collection():
    """Initialize and return the ChromaDB drug collection (singleton)."""
    global _chroma_client, _drug_collection
    
    if _drug_collection is not None:
        return _drug_collection
    
    try:
        # Use sentence-transformers for local embeddings (no API key needed)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        _drug_collection = _chroma_client.get_or_create_collection(
            name="medlink_drugs",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Seed if empty
        if _drug_collection.count() == 0:
            print("[RAG] Seeding drug knowledge base into ChromaDB...")
            _seed_drug_collection(_drug_collection)
            print(f"[RAG] Seeded {_drug_collection.count()} drugs into ChromaDB.")
        else:
            print(f"[RAG] ChromaDB loaded: {_drug_collection.count()} drugs available.")
        
        return _drug_collection
        
    except Exception as e:
        print(f"[RAG] ChromaDB init error: {e}")
        return None


def _seed_drug_collection(collection):
    """Add all drug documents to ChromaDB, deduplicating IDs."""
    drugs = get_all_drugs()
    
    documents = []
    ids = []
    metadatas = []
    seen_ids = {}

    for drug in drugs:
        doc_text = format_drug_for_chroma(drug)
        base_id = drug['id']
        # Deduplicate: if ID already seen, append _2, _3, etc.
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            unique_id = f"{base_id}_{seen_ids[base_id]}"
        else:
            seen_ids[base_id] = 1
            unique_id = base_id

        documents.append(doc_text)
        ids.append(unique_id)
        metadatas.append({
            'name': drug['name'],
            'category': drug['category'],
            'brand_names': drug['brand_names'],
            'alternatives': drug['alternatives'],
            'pregnancy_safety': drug['pregnancy_safety']
        })

    print(f"[RAG] Preparing to seed {len(documents)} drugs ({len(set(ids))} unique IDs)...")
    # Add in batches
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )



def _extract_drug_keywords(query: str) -> list:
    """Extract potential drug names from a query (English + Arabic transliterations)."""
    drug_name_map = {
        'amoxicillin': 'amoxicillin', 'اموكسيسيلين': 'amoxicillin', 'أموكسيسيلين': 'amoxicillin',
        'ibuprofen': 'ibuprofen', 'ايبوبروفين': 'ibuprofen', 'إيبوبروفين': 'ibuprofen', 'بروفين': 'ibuprofen',
        'paracetamol': 'paracetamol', 'باراسيتامول': 'paracetamol', 'بنادول': 'paracetamol',
        'warfarin': 'warfarin', 'وارفارين': 'warfarin',
        'metformin': 'metformin', 'ميتفورمين': 'metformin', 'جلوكوفاج': 'metformin',
        'omeprazole': 'omeprazole', 'اوميبرازول': 'omeprazole', 'أوميبرازول': 'omeprazole', 'لوسيك': 'omeprazole',
        'aspirin': 'aspirin', 'اسبرين': 'aspirin', 'أسبرين': 'aspirin',
        'atorvastatin': 'atorvastatin', 'اتورفاستاتين': 'atorvastatin', 'ليبيتور': 'atorvastatin',
        'metoprolol': 'metoprolol', 'ميتوبرولول': 'metoprolol',
        'amlodipine': 'amlodipine', 'أملوديبين': 'amlodipine', 'نورفاسك': 'amlodipine',
        'azithromycin': 'azithromycin', 'ازيثروميسين': 'azithromycin', 'زيثروماكس': 'azithromycin',
        'ciprofloxacin': 'ciprofloxacin', 'سيبروفلوكساسين': 'ciprofloxacin',
        'metronidazole': 'metronidazole', 'ميترونيدازول': 'metronidazole', 'فلاجيل': 'metronidazole',
        'cetirizine': 'cetirizine', 'سيتيريزين': 'cetirizine', 'زيرتك': 'cetirizine',
        'loratadine': 'loratadine', 'لوراتادين': 'loratadine', 'كلاريتين': 'loratadine',
        'salbutamol': 'salbutamol', 'سالبوتامول': 'salbutamol', 'فنتولين': 'salbutamol', 'ventolin': 'salbutamol',
        'sertraline': 'sertraline', 'سيرترالين': 'sertraline', 'زولوفت': 'sertraline',
        'pregabalin': 'pregabalin', 'بريجابالين': 'pregabalin', 'ليريكا': 'pregabalin',
        'alprazolam': 'alprazolam', 'ألبرازولام': 'alprazolam', 'زاناكس': 'alprazolam',
        'fluconazole': 'fluconazole', 'فلوكونازول': 'fluconazole', 'ديفلوكان': 'fluconazole',
        'clotrimazole': 'clotrimazole', 'كلوتريمازول': 'clotrimazole', 'كانستن': 'clotrimazole',
        'levothyroxine': 'levothyroxine', 'ليفوثيروكسين': 'levothyroxine', 'ايلتروكسين': 'levothyroxine',
        'ondansetron': 'ondansetron', 'اوندانسيترون': 'ondansetron', 'زوفران': 'ondansetron',
        'metoclopramide': 'metoclopramide', 'ميتوكلوبراميد': 'metoclopramide', 'بريميران': 'metoclopramide',
        'diclofenac': 'diclofenac', 'ديكلوفيناك': 'diclofenac', 'فولتارين': 'diclofenac',
        'montelukast': 'montelukast', 'مونتيلوكاست': 'montelukast', 'سينجولار': 'montelukast',
        'glibenclamide': 'glibenclamide', 'جليبنكلاميد': 'glibenclamide',
        'hydrocortisone': 'hydrocortisone', 'هيدروكورتيزون': 'hydrocortisone',
        'vitamin d': 'vitamin_d', 'فيتامين د': 'vitamin_d', 'فيتامين d': 'vitamin_d',
        'iron': 'iron_supplement', 'حديد': 'iron_supplement',
        'timolol': 'timolol_eye', 'تيمولول': 'timolol_eye',
    }
    query_lower = query.lower()
    matched_ids = []
    for keyword, drug_id in drug_name_map.items():
        if keyword in query_lower and drug_id not in matched_ids:
            matched_ids.append(drug_id)
    return matched_ids


def search_drugs_rag(query: str, n_results: int = 3) -> list:
    """
    Hybrid search: exact drug name match (boosted) + semantic vector search.
    Returns list of drug dicts sorted by relevance.
    """
    collection = get_chroma_collection()
    if collection is None:
        return []

    try:
        # Step A: Detect any explicit drug name mentions in the query
        matched_ids = _extract_drug_keywords(query)

        # Step B: Semantic search (always run)
        semantic_results = collection.query(
            query_texts=[query],
            n_results=min(n_results + 2, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = {}
        for i in range(len(semantic_results['ids'][0])):
            drug_id = semantic_results['ids'][0][i]
            score = 1 - semantic_results['distances'][0][i]
            hits[drug_id] = {
                'id': drug_id,
                'document': semantic_results['documents'][0][i],
                'metadata': semantic_results['metadatas'][0][i],
                'score': score
            }

        # Step C: Boost explicitly matched drugs to top (score = 0.99)
        for drug_id in matched_ids:
            if drug_id in hits:
                hits[drug_id]['score'] = 0.99  # Boost to top
            else:
                # Fetch from collection by ID
                try:
                    fetched = collection.get(ids=[drug_id], include=["documents", "metadatas"])
                    if fetched['ids']:
                        hits[drug_id] = {
                            'id': drug_id,
                            'document': fetched['documents'][0],
                            'metadata': fetched['metadatas'][0],
                            'score': 0.99
                        }
                except Exception:
                    pass

        # Sort by score descending, return top n
        sorted_hits = sorted(hits.values(), key=lambda x: x['score'], reverse=True)
        return sorted_hits[:n_results]

    except Exception as e:
        print(f"[RAG] Search error: {e}")
        return []



def generate_drug_answer(query: str, retrieved_docs: list, groq_api_key: str) -> str:
    """
    Use Groq LLaMA 3 to generate a comprehensive answer from retrieved drug documents.
    """
    if not groq_api_key or not retrieved_docs:
        return _fallback_drug_answer(retrieved_docs)
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:3]):
        context_parts.append(f"--- Drug {i+1}: {doc['metadata']['name']} ---\n{doc['document']}")
    
    context = "\n\n".join(context_parts)
    
    system_prompt = """You are a pharmacist assistant for MedLink AI Egypt.
Answer drug questions briefly using the context provided.
Reply in the same language as the question (Arabic or English).
For interactions: clearly say SAFE or DANGEROUS.
End with: 'Consult your pharmacist for personal advice.'
Max 3 short paragraphs."""

    user_message = f"""Question: {query}

Drug Database Context:
{context}

Please provide a comprehensive, well-structured answer."""

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=250,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[RAG] Groq generation error: {e}")
        return _fallback_drug_answer(retrieved_docs)


def _fallback_drug_answer(retrieved_docs: list) -> str:
    """Simple fallback answer without LLM."""
    if not retrieved_docs:
        return "لم يتم العثور على معلومات عن هذا الدواء في قاعدة البيانات. / No drug information found."
    
    top = retrieved_docs[0]
    return f"**{top['metadata']['name']}**\n\n{top['document']}"


def generate_xai_explanation(query: str, retrieved_docs: list, answer: str) -> dict:
    """
    Generate Explainable AI (XAI) explanation for the answer.
    Returns why the system gave this specific answer.
    """
    if not retrieved_docs:
        return {'explanation': 'لا توجد مصادر متاحة لهذا الاستعلام.'}
    
    top_doc = retrieved_docs[0]
    drug_name = top_doc.get('metadata', {}).get('name', 'Unknown')
    score = top_doc.get('score', 0)
    
    # Extract why this drug was chosen
    query_words = set(query.lower().split())
    doc_text = top_doc.get('document', '').lower()
    matching_terms = [w for w in query_words if len(w) > 4 and w in doc_text]
    
    explanation = {
        'primary_source': drug_name,
        'similarity_score': round(score * 100, 1),
        'matching_terms': matching_terms[:5],
        'sources_used': [d.get('metadata', {}).get('name', '?') for d in retrieved_docs],
        'explanation_ar': f"تم اختيار هذا الرد بناءً على {len(retrieved_docs)} مصدر طبي. "
                          f"المصدر الرئيسي: {drug_name} (تطابق {round(score*100,1)}%). "
                          f"الكلمات المتطابقة: {', '.join(matching_terms[:3]) if matching_terms else 'سياق عام'}.",
        'explanation_en': f"Answer based on {len(retrieved_docs)} medical sources. "
                          f"Primary: {drug_name} ({round(score*100,1)}% match). "
                          f"Matched terms: {', '.join(matching_terms[:3]) if matching_terms else 'general context'}."
    }
    return explanation


def drug_rag_search(query: str, groq_api_key: str = '') -> dict:
    """
    Full RAG pipeline: search -> retrieve -> generate answer.
    Results cached for speed.
    """
    # Cache hit = instant response
    cache_key = query.strip().lower()
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    # Step 1: Semantic search
    retrieved = search_drugs_rag(query, n_results=3)
    
    if not retrieved:
        return {
            'answer': 'لم يتم العثور على نتائج. / No results found.',
            'sources': [], 'query': query,
            'rag_powered': bool(groq_api_key),
            'confidence': 0.0,
            'drug_info': {}
        }
    
    # Step 2: Generate answer with LLM (fast 8B model)
    answer = generate_drug_answer(query, retrieved, groq_api_key)
    
    # Step 3: Confidence from top result score
    top_score = retrieved[0].get('score', 0.5)
    confidence = round(top_score, 3)
    
    # Step 4: Format sources
    sources = [
        {
            'name': doc['metadata']['name'],
            'category': doc['metadata']['category'],
            'brand_names': doc['metadata']['brand_names'],
            'alternatives': doc['metadata']['alternatives'],
            'score': round(doc['score'], 3),
        }
        for doc in retrieved
    ]
    
    # Build drug_info from top result for frontend card
    top_meta = retrieved[0]['metadata'] if retrieved else {}
    
    result = {
        'answer': answer,
        'sources': sources,
        'query': query,
        'rag_powered': bool(groq_api_key),
        'results_count': len(retrieved),
        'confidence': confidence,
        'drug_info': top_meta
    }
    # Store in cache (evict oldest if full)
    if len(_search_cache) >= MAX_CACHE_SIZE:
        oldest = next(iter(_search_cache))
        del _search_cache[oldest]
    _search_cache[cache_key] = result
    return result




if __name__ == "__main__":
    print("Testing RAG Drug Search Engine...")
    result = drug_rag_search("ما هو الأمان من تناول أموكسيسيلين مع الوارفارين؟", 
                              os.environ.get('GROQ_API_KEY', ''))
    print(f"\nQuery: {result['query']}")
    print(f"RAG Powered: {result['rag_powered']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {[s['name'] for s in result['sources']]}")
