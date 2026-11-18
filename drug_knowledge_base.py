# -*- coding: utf-8 -*-
"""
MedLink AI - Drug Knowledge Base for RAG
Loads drugs dynamically from a CSV file (scalable to 5000+ drugs).
Used by ChromaDB for semantic drug search.
"""
import os
import csv

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, 'dataset', 'medlink_drugs_database.csv')

def load_drugs_from_csv():
    """Reads drugs from the CSV file and returns a list of dictionaries."""
    drugs = []
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[WARNING] Drug database CSV not found at {CSV_FILE_PATH}")
        return drugs
        
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                drugs.append(row)
        print(f"[SUCCESS] Loaded {len(drugs)} drugs from CSV.")
    except Exception as e:
        print(f"[ERROR] Failed to load drug CSV: {e}")
        
    return drugs

def get_all_drugs():
    """Returns the combined list of all drugs."""
    return load_drugs_from_csv()

def format_drug_for_chroma(drug):
    """
    Formats a drug dictionary into a text string for ChromaDB embedding.
    This function is required by drug_rag.py for vector indexing.
    """
    parts = []
    for key in ['name', 'category', 'uses', 'dosage', 'side_effects',
                'contraindications', 'interactions', 'alternatives',
                'pregnancy_safety', 'brand_names']:
        value = drug.get(key, '')
        if value:
            parts.append(f"{key}: {value}")
    return ' | '.join(parts)

if __name__ == "__main__":
    # Test loading
    drugs = get_all_drugs()
    print(f"Total drugs in DB: {len(drugs)}")
    if drugs:
        print(f"Sample drug: {drugs[0]['name']}")
