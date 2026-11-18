"""Expand drug database from OpenFDA API"""
import urllib.request, json, csv, os, time

out_file = r'H:\Drug_Intelligence_System_COMPLETE_FINAL\final_project\dataset\medlink_drugs_database.csv'

# Read existing
with open(out_file, 'r', encoding='utf-8') as f:
    existing = list(csv.DictReader(f))
existing_names = {d.get('name','').lower() for d in existing}
print(f'Existing: {len(existing)} drugs')

categories = [
    ('antibiotic', 'Antibiotic'),
    ('antihistamine', 'Antihistamine'),
    ('antihypertensive', 'Antihypertensive'),
    ('antidiabetic', 'Antidiabetic'),
    ('analgesic', 'Analgesic / Pain Relief'),
    ('antidepressant', 'Antidepressant'),
    ('statin', 'Lipid Lowering / Statin'),
    ('corticosteroid', 'Corticosteroid'),
    ('antifungal', 'Antifungal'),
    ('antiviral', 'Antiviral'),
    ('bronchodilator', 'Bronchodilator / Respiratory'),
    ('omeprazole', 'GI / Proton Pump Inhibitor'),
    ('anticoagulant', 'Anticoagulant'),
    ('diuretic', 'Diuretic'),
    ('antiepileptic', 'Antiepileptic'),
    ('antipsychotic', 'Antipsychotic'),
    ('thyroid', 'Thyroid'),
    ('vitamin', 'Vitamin / Supplement'),
    ('calcium', 'Calcium / Supplement'),
    ('iron', 'Iron / Supplement'),
]

fieldnames = ['id','name','category','uses','dosage','side_effects',
              'contraindications','interactions','alternatives','pregnancy_safety','brand_names']

new_drugs = []

for term, cat in categories:
    try:
        safe_term = term.replace(' ', '+')
        url = f'https://api.fda.gov/drug/label.json?search=indications_and_usage:{safe_term}&limit=10'
        req = urllib.request.Request(url, headers={'User-Agent': 'MedLinkAI/1.0'})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        results = data.get('results', [])
        for r in results:
            name_list = r.get('openfda', {}).get('brand_name', [])
            generic_list = r.get('openfda', {}).get('generic_name', [])
            name = (name_list[0] if name_list else '') or (generic_list[0] if generic_list else '')
            if not name or name.lower() in existing_names:
                continue
            drug = {
                'id': name.lower().replace(' ','_').replace('/','_')[:40],
                'name': name.title(),
                'category': cat,
                'uses': ' '.join(r.get('indications_and_usage', [''])[:1])[:300],
                'dosage': ' '.join(r.get('dosage_and_administration', [''])[:1])[:300],
                'side_effects': ' '.join(r.get('adverse_reactions', [''])[:1])[:300],
                'contraindications': ' '.join(r.get('contraindications', [''])[:1])[:200],
                'interactions': ' '.join(r.get('drug_interactions', [''])[:1])[:200],
                'alternatives': '',
                'pregnancy_safety': ' '.join(r.get('pregnancy', [''])[:1])[:100],
                'brand_names': ', '.join(name_list[:3]),
            }
            new_drugs.append(drug)
            existing_names.add(name.lower())
        time.sleep(0.3)
        print(f'  {cat}: done')
    except Exception as e:
        print(f'  Skip {term}: {e}')

print(f'New drugs from OpenFDA: {len(new_drugs)}')

all_drugs = existing + new_drugs
with open(out_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_drugs)

print(f'Total in CSV: {len(all_drugs)} drugs')
