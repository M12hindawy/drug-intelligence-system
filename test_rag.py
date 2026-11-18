import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from drug_rag import get_chroma_collection, search_drugs_rag

print("Testing ChromaDB + RAG pipeline...")
col = get_chroma_collection()
print(f"Collection ready: {col.count()} drugs")

results = search_drugs_rag("ibuprofen warfarin interaction", 2)
print(f"Search results: {len(results)} hits")
for r in results:
    print(f"  - {r['id']} (score: {r['score']:.3f})")
    
results2 = search_drugs_rag("amoxicillin dosage children", 2)
print(f"\nSearch 2 results: {len(results2)} hits")
for r in results2:
    print(f"  - {r['id']} (score: {r['score']:.3f})")

print("\nRAG pipeline: OK")
