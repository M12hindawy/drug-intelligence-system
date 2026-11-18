"""
MedLink AI — Drug Database Expander
Fetches 5000+ drugs from ChEMBL API and adds them to ChromaDB RAG.
Run once to populate the database.

Usage:
    python expand_drug_db.py
"""

import os
import sys
import json
import time
import requests
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_drug_db")
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_API = "https://api.fda.gov/drug"

# ─── ChEMBL Drug Fetcher ──────────────────────────────────────────────────────

def fetch_chembl_drugs(max_drugs: int = 3000) -> list[dict]:
    """Fetch approved drugs from ChEMBL API."""
    drugs = []
    offset = 0
    limit = 100

    print(f"[ChEMBL] Fetching up to {max_drugs} approved drugs...")

    while len(drugs) < max_drugs:
        try:
            url = f"{CHEMBL_API}/molecule.json"
            params = {
                "molecule_type": "Small molecule",
                "max_phase": 4,  # Approved drugs only
                "limit": limit,
                "offset": offset,
                "format": "json"
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            molecules = data.get("molecules", [])
            if not molecules:
                break

            for mol in molecules:
                drug = _parse_chembl_molecule(mol)
                if drug:
                    drugs.append(drug)

            total = data.get("page_meta", {}).get("total_count", 0)
            offset += limit
            print(f"[ChEMBL] Fetched {len(drugs)}/{min(max_drugs, total)} drugs...")

            if offset >= total or offset >= max_drugs:
                break

            time.sleep(0.5)  # Rate limit respect

        except Exception as e:
            print(f"[ChEMBL] Error at offset {offset}: {e}")
            time.sleep(2)
            break

    print(f"[ChEMBL] Total fetched: {len(drugs)} drugs")
    return drugs


def _parse_chembl_molecule(mol: dict) -> dict | None:
    """Parse a ChEMBL molecule into our drug format."""
    try:
        name = mol.get("pref_name", "")
        if not name:
            return None

        chembl_id = mol.get("molecule_chembl_id", "")
        mol_props = mol.get("molecule_properties") or {}
        synonyms = mol.get("molecule_synonyms", []) or []

        # Get brand names from synonyms
        brand_names = []
        for syn in synonyms[:5]:
            syn_name = syn.get("molecule_synonym", "")
            if syn_name and syn_name != name and len(syn_name) < 30:
                brand_names.append(syn_name)

        # Get molecular info
        mw = mol_props.get("full_mw", "")
        formula = mol_props.get("full_molformula", "")

        # Get indications from molecule_type
        indication = mol.get("indication_class", "") or ""

        # Build drug document
        doc_id = f"chembl_{chembl_id.lower()}"
        drug_name = name.lower().replace(" ", "_")

        document_text = f"""Drug: {name}
ChEMBL ID: {chembl_id}
Brand Names: {', '.join(brand_names) if brand_names else 'N/A'}
Molecular Formula: {formula}
Molecular Weight: {mw}
Indication Class: {indication}
Type: Small Molecule - Phase 4 (Approved)
Source: ChEMBL Database
"""

        return {
            "id": doc_id,
            "name": name,
            "document": document_text,
            "metadata": {
                "name": name,
                "category": indication or "Pharmaceutical",
                "brand_names": ", ".join(brand_names[:3]),
                "alternatives": "",
                "pregnancy_safety": "Consult doctor",
                "source": "ChEMBL",
                "chembl_id": chembl_id,
                "formula": formula
            }
        }
    except Exception:
        return None


# ─── OpenFDA Drug Fetcher ─────────────────────────────────────────────────────

def fetch_openfda_drugs(max_drugs: int = 2000) -> list[dict]:
    """Fetch drug label data from OpenFDA."""
    drugs = []
    skip = 0
    limit = 100

    print(f"[OpenFDA] Fetching up to {max_drugs} drug labels...")

    while len(drugs) < max_drugs:
        try:
            url = f"{OPENFDA_API}/label.json"
            params = {
                "limit": limit,
                "skip": skip,
                "search": "_exists_:openfda.generic_name"
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for result in results:
                drug = _parse_openfda_label(result)
                if drug:
                    drugs.append(drug)

            skip += limit
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            print(f"[OpenFDA] Fetched {len(drugs)}/{min(max_drugs, total)} drugs...")

            if skip >= total or skip >= max_drugs:
                break

            time.sleep(0.3)

        except Exception as e:
            print(f"[OpenFDA] Error at skip {skip}: {e}")
            break

    print(f"[OpenFDA] Total fetched: {len(drugs)} drugs")
    return drugs


def _parse_openfda_label(label: dict) -> dict | None:
    """Parse OpenFDA drug label into our format."""
    try:
        openfda = label.get("openfda", {})
        generic_names = openfda.get("generic_name", [])
        brand_names = openfda.get("brand_name", [])

        if not generic_names:
            return None

        generic_name = generic_names[0].strip()
        if not generic_name or len(generic_name) > 100:
            return None

        # Get key sections
        indications = " ".join(label.get("indications_and_usage", [""])[:1])[:500]
        warnings = " ".join(label.get("warnings", [""])[:1])[:300]
        dosage = " ".join(label.get("dosage_and_administration", [""])[:1])[:300]
        contraindications = " ".join(label.get("contraindications", [""])[:1])[:300]

        doc_id = f"fda_{generic_name.lower().replace(' ', '_').replace('/', '_')[:50]}"

        document_text = f"""Drug: {generic_name}
Brand Names: {', '.join(brand_names[:5]) if brand_names else 'N/A'}
Indications: {indications[:400] if indications else 'See prescribing information'}
Warnings: {warnings[:300] if warnings else 'See prescribing information'}
Dosage: {dosage[:300] if dosage else 'As directed by physician'}
Contraindications: {contraindications[:300] if contraindications else 'See prescribing information'}
Source: FDA Drug Label Database
"""

        return {
            "id": doc_id,
            "name": generic_name,
            "document": document_text,
            "metadata": {
                "name": generic_name,
                "category": "Pharmaceutical",
                "brand_names": ", ".join(brand_names[:3]),
                "alternatives": "",
                "pregnancy_safety": "Consult doctor",
                "source": "OpenFDA"
            }
        }
    except Exception:
        return None


# ─── ChromaDB Seeder ──────────────────────────────────────────────────────────

def seed_chromadb(drugs: list[dict]):
    """Add drugs to ChromaDB in batches."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name="medlink_drugs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    print(f"[ChromaDB] Current drug count: {existing_count}")

    # Remove duplicates by ID
    existing_ids = set()
    if existing_count > 0:
        try:
            existing = collection.get(limit=existing_count)
            existing_ids = set(existing["ids"])
        except Exception:
            pass

    new_drugs = [d for d in drugs if d["id"] not in existing_ids]
    print(f"[ChromaDB] Adding {len(new_drugs)} new drugs...")

    batch_size = 50
    added = 0

    for i in range(0, len(new_drugs), batch_size):
        batch = new_drugs[i:i + batch_size]
        try:
            collection.add(
                ids=[d["id"] for d in batch],
                documents=[d["document"] for d in batch],
                metadatas=[d["metadata"] for d in batch]
            )
            added += len(batch)
            print(f"[ChromaDB] Added batch {i//batch_size + 1} — Total: {existing_count + added} drugs")
        except Exception as e:
            print(f"[ChromaDB] Batch error: {e}")
            time.sleep(1)

    final_count = collection.count()
    print(f"\n✅ Done! ChromaDB now has {final_count} drugs")
    return final_count


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MedLink AI — Drug Database Expander")
    print("Target: 5000+ drugs from ChEMBL + OpenFDA")
    print("=" * 60)

    # First, add existing knowledge base
    from drug_rag import get_chroma_collection
    existing = get_chroma_collection()
    if existing:
        count = existing.count()
        print(f"\n[Base] Existing drugs in ChromaDB: {count}")

    all_drugs = []

    # Fetch from ChEMBL (3000 drugs)
    print("\n[1/2] Fetching from ChEMBL...")
    try:
        chembl_drugs = fetch_chembl_drugs(max_drugs=3000)
        all_drugs.extend(chembl_drugs)
        print(f"✅ ChEMBL: {len(chembl_drugs)} drugs")
    except Exception as e:
        print(f"❌ ChEMBL failed: {e}")

    # Fetch from OpenFDA (2000 drugs)
    print("\n[2/2] Fetching from OpenFDA...")
    try:
        fda_drugs = fetch_openfda_drugs(max_drugs=2000)
        all_drugs.extend(fda_drugs)
        print(f"✅ OpenFDA: {len(fda_drugs)} drugs")
    except Exception as e:
        print(f"❌ OpenFDA failed: {e}")

    print(f"\n📊 Total new drugs to add: {len(all_drugs)}")

    if all_drugs:
        final_count = seed_chromadb(all_drugs)
        print(f"\n🎉 Database now has {final_count} drugs!")
    else:
        print("❌ No drugs fetched. Check internet connection.")
