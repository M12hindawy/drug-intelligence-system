"""Patch: Add Arabic drug name mapping to dosage calculator in app.py"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the old fallback block and replace with new version
old = """    # Fallback: built-in common drug database
    drug_key = drug_name.lower().strip()
    drug_data = COMMON_DRUG_DOSAGES.get(drug_key)
    if not drug_data:
        # fuzzy match
        for k in COMMON_DRUG_DOSAGES:
            if k in drug_key or drug_key in k:
                drug_data = COMMON_DRUG_DOSAGES[k]
                break"""

new = """    # Fallback: built-in common drug database
    # Arabic phonetic to English mapping
    AR_TO_EN = {
        '\u0628\u0627\u0631\u0627\u0633\u064a\u062a\u0627\u0645\u0648\u0644': 'paracetamol', '\u0628\u0627\u0631\u0633\u064a\u062a\u0627\u0645\u0648\u0644': 'paracetamol',
        '\u0628\u0627\u0646\u0627\u062f\u0648\u0644': 'panadol', '\u0628\u0627\u0646\u0627\u0643\u0648\u0644': 'panadol', '\u0623\u062f\u0648\u0644': 'paracetamol',
        '\u0625\u064a\u0628\u0648\u0628\u0631\u0648\u0641\u064a\u0646': 'ibuprofen', '\u0627\u064a\u0628\u0648\u0628\u0631\u0648\u0641\u064a\u0646': 'ibuprofen', '\u0628\u0631\u0648\u0641\u064a\u0646': 'brufen',
        '\u0623\u0645\u0648\u0643\u0633\u064a\u0633\u064a\u0644\u064a\u0646': 'amoxicillin', '\u0627\u0645\u0648\u0643\u0633\u064a\u0633\u064a\u0644\u064a\u0646': 'amoxicillin',
        '\u0623\u0648\u062c\u0645\u0646\u062a\u064a\u0646': 'augmentin', '\u0627\u0648\u062c\u0645\u0646\u062a\u064a\u0646': 'augmentin',
        '\u0645\u064a\u062a\u0641\u0648\u0631\u0645\u064a\u0646': 'metformin', '\u062c\u0644\u0648\u0643\u0648\u0641\u0627\u062c': 'metformin', '\u0633\u064a\u062f\u0648\u0641\u0627\u062c': 'metformin',
        '\u0623\u0648\u0645\u064a\u0628\u0631\u0627\u0632\u0648\u0644': 'omeprazole', '\u0627\u0648\u0645\u064a\u0628\u0631\u0627\u0632\u0648\u0644': 'omeprazole', '\u0644\u0648\u0632\u064a\u0643': 'omeprazole',
        '\u0644\u0648\u0633\u0627\u0631\u062a\u0627\u0646': 'losartan', '\u0643\u0648\u0632\u0627\u0631': 'losartan',
        '\u0623\u062a\u0648\u0631\u0641\u0627\u0633\u062a\u0627\u062a\u064a\u0646': 'atorvastatin', '\u0644\u064a\u0628\u064a\u062a\u0648\u0631': 'atorvastatin',
        '\u0623\u0633\u0628\u0631\u064a\u0646': 'aspirin', '\u0627\u0633\u0628\u0631\u064a\u0646': 'aspirin', '\u0623\u0633\u0628\u0648\u0633\u064a\u062f': 'aspirin',
        '\u0623\u0632\u064a\u062b\u0631\u0648\u0645\u0627\u064a\u0633\u064a\u0646': 'azithromycin', '\u0632\u064a\u062b\u0631\u0648\u0645\u0627\u0643\u0633': 'azithromycin',
        '\u0633\u064a\u0628\u0631\u0648\u0641\u0644\u0648\u0643\u0633\u0627\u0633\u064a\u0646': 'ciprofloxacin', '\u0633\u064a\u0628\u0631\u0648': 'ciprofloxacin',
        '\u0633\u064a\u062a\u064a\u0631\u064a\u0632\u064a\u0646': 'cetirizine', '\u0632\u064a\u0631\u062a\u064a\u0643': 'cetirizine',
        '\u0628\u0631\u064a\u062f\u0646\u064a\u0632\u0648\u0644\u0648\u0646': 'prednisolone',
        '\u0643\u0644\u0648\u0628\u064a\u062f\u0648\u062c\u0631\u064a\u0644': 'clopidogrel', '\u0628\u0644\u0627\u0641\u064a\u0643\u0633': 'clopidogrel',
        '\u0644\u064a\u0632\u064a\u0646\u0648\u0628\u0631\u064a\u0644': 'lisinopril',
        '\u0641\u064a\u0648\u0631\u0648\u0633\u064a\u0645\u064a\u062f': 'furosemide', '\u0644\u0627\u0633\u064a\u0643\u0633': 'furosemide',
        '\u0623\u0646\u0633\u0648\u0644\u064a\u0646': 'insulin', '\u0627\u0646\u0633\u0648\u0644\u064a\u0646': 'insulin',
    }
    drug_input = drug_name.strip()
    drug_key = drug_input.lower()
    # Check Arabic map
    mapped = AR_TO_EN.get(drug_input) or AR_TO_EN.get(drug_key)
    if mapped:
        drug_key = mapped
    drug_data = COMMON_DRUG_DOSAGES.get(drug_key)
    if not drug_data:
        for k in COMMON_DRUG_DOSAGES:
            if k in drug_key or drug_key in k:
                drug_data = COMMON_DRUG_DOSAGES[k]
                break
    if not drug_data:
        for ar, en in AR_TO_EN.items():
            if ar in drug_input or drug_input in ar:
                drug_data = COMMON_DRUG_DOSAGES.get(en)
                if drug_data:
                    break"""

if old in content:
    content = content.replace(old, new)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Arabic mapping added!")
else:
    print("ERROR: Old text not found!")
    # Show what's around line 351
    lines = content.split('\n')
    for i, line in enumerate(lines[349:365], start=350):
        print(f"{i}: {repr(line)}")
