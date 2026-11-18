"""
Drug Interaction Checker — MedLink AI
=====================================
Checks for:
1. Drug-Drug Interactions (DDI): dangerous combinations
2. Drug-Allergy Conflicts: prescribed drug vs patient allergies
3. Drug-Condition Conflicts: prescribed drug vs chronic conditions
4. Duplicate Therapy: same drug class prescribed twice

Severity Levels:
- CRITICAL: Life-threatening, BLOCK prescription
- HIGH: Dangerous, WARN strongly  
- MODERATE: Monitor closely
- LOW: Minor, informational
"""

import json

# ============================================================
# 1. DRUG-DRUG INTERACTION DATABASE
#    Format: (drug_a, drug_b) -> {severity, description_ar, description_en}
#    Drug names are stored as lowercase generic names
# ============================================================

DRUG_DRUG_INTERACTIONS = {
    # --- Cardiovascular ---
    ("enalapril", "losartan"): {
        "severity": "HIGH",
        "en": "ACE inhibitor + ARB: Dual RAAS blockade increases risk of hyperkalemia, hypotension, and renal failure.",
        "ar": "⚠️ مثبط ACE + حاصر مستقبلات أنجيوتنسين: خطر ارتفاع البوتاسيوم وهبوط الضغط والفشل الكلوي."
    },
    ("amlodipine", "metoprolol"): {
        "severity": "MODERATE",
        "en": "CCB + Beta-blocker: Risk of excessive bradycardia and hypotension.",
        "ar": "⚠️ حاصر قنوات الكالسيوم + بيتا بلوكر: خطر بطء القلب المفرط وانخفاض الضغط."
    },
    ("enalapril", "spironolactone"): {
        "severity": "HIGH",
        "en": "ACE inhibitor + K-sparing diuretic: Severe hyperkalemia risk.",
        "ar": "⚠️ مثبط ACE + مدر بوتاسيوم: خطر ارتفاع البوتاسيوم الشديد."
    },

    # --- NSAIDs interactions ---
    ("ibuprofen", "enalapril"): {
        "severity": "HIGH",
        "en": "NSAID reduces antihypertensive effect of ACE inhibitor and increases renal failure risk.",
        "ar": "⚠️ مضاد التهاب + مثبط ACE: يقلل فاعلية خفض الضغط ويزيد خطر الفشل الكلوي."
    },
    ("ibuprofen", "losartan"): {
        "severity": "HIGH",
        "en": "NSAID reduces antihypertensive effect of ARB and increases renal failure risk.",
        "ar": "⚠️ مضاد التهاب + حاصر أنجيوتنسين: يقلل فاعلية خفض الضغط ويزيد خطر الفشل الكلوي."
    },
    ("diclofenac", "enalapril"): {
        "severity": "HIGH",
        "en": "NSAID + ACE inhibitor: Reduced BP control, increased renal risk.",
        "ar": "⚠️ ديكلوفيناك + مثبط ACE: تقليل فاعلية الضغط وزيادة خطر الكلى."
    },
    ("diclofenac", "losartan"): {
        "severity": "HIGH",
        "en": "NSAID + ARB: Reduced BP control, increased renal risk.",
        "ar": "⚠️ ديكلوفيناك + حاصر أنجيوتنسين: تقليل فاعلية الضغط وزيادة خطر الكلى."
    },
    ("ibuprofen", "diclofenac"): {
        "severity": "CRITICAL",
        "en": "Two NSAIDs together: Extremely high risk of GI bleeding and kidney damage.",
        "ar": "🚫 مضادين التهاب معاً: خطر شديد جداً للنزيف المعوي وتلف الكلى."
    },
    ("celecoxib", "ibuprofen"): {
        "severity": "CRITICAL",
        "en": "Two NSAIDs (COX-2 + non-selective): High GI bleeding and cardiovascular risk.",
        "ar": "🚫 مضادين التهاب معاً: خطر شديد للنزيف والقلب."
    },
    ("celecoxib", "diclofenac"): {
        "severity": "CRITICAL",
        "en": "Two NSAIDs: Extremely high risk of GI bleeding.",
        "ar": "🚫 مضادين التهاب معاً: خطر شديد جداً للنزيف المعوي."
    },
    ("ibuprofen", "methotrexate"): {
        "severity": "CRITICAL",
        "en": "NSAID increases Methotrexate toxicity — risk of bone marrow suppression.",
        "ar": "🚫 مضاد التهاب + ميثوتريكسات: خطر تسمم نخاع العظام."
    },
    ("diclofenac", "methotrexate"): {
        "severity": "CRITICAL",
        "en": "NSAID increases Methotrexate toxicity — risk of bone marrow suppression.",
        "ar": "🚫 مضاد التهاب + ميثوتريكسات: خطر تسمم نخاع العظام."
    },

    # --- Diabetes ---
    ("metformin", "enalapril"): {
        "severity": "LOW",
        "en": "ACE inhibitors may slightly enhance hypoglycemic effect of Metformin. Monitor blood sugar.",
        "ar": "ℹ️ مثبط ACE قد يزيد طفيفاً من تأثير خفض السكر. راقب مستوى السكر."
    },
    ("metformin", "ciprofloxacin"): {
        "severity": "MODERATE",
        "en": "Ciprofloxacin may cause unpredictable blood sugar changes with Metformin.",
        "ar": "⚠️ سيبروفلوكساسين قد يسبب تغيرات غير متوقعة في السكر مع الميتفورمين."
    },
    ("sitagliptin", "metformin"): {
        "severity": "LOW",
        "en": "Common safe combination. Monitor for hypoglycemia.",
        "ar": "ℹ️ توليفة شائعة وآمنة. راقب انخفاض السكر."
    },

    # --- Antibiotics ---
    ("amoxicillin", "methotrexate"): {
        "severity": "HIGH",
        "en": "Amoxicillin reduces Methotrexate excretion — increased toxicity risk.",
        "ar": "⚠️ أموكسيسيللين يقلل إخراج الميثوتريكسات — خطر السمية."
    },
    ("clarithromycin", "amlodipine"): {
        "severity": "MODERATE",
        "en": "Clarithromycin inhibits CYP3A4, increasing Amlodipine levels — risk of hypotension.",
        "ar": "⚠️ كلاريثرومايسين يزيد مستوى الأملوديبين — خطر هبوط الضغط."
    },
    ("clarithromycin", "sertraline"): {
        "severity": "HIGH",
        "en": "Both prolong QT interval — risk of fatal cardiac arrhythmia.",
        "ar": "⚠️ كلاهما يطيل QT — خطر اضطراب نظم القلب المميت."
    },
    ("clarithromycin", "escitalopram"): {
        "severity": "HIGH",
        "en": "Both prolong QT interval — risk of fatal cardiac arrhythmia.",
        "ar": "⚠️ كلاهما يطيل QT — خطر اضطراب نظم القلب المميت."
    },
    ("ciprofloxacin", "escitalopram"): {
        "severity": "HIGH",
        "en": "Both prolong QT interval — risk of cardiac arrhythmia.",
        "ar": "⚠️ كلاهما يطيل QT — خطر اضطراب نظم القلب."
    },

    # --- Psychiatric ---
    ("sertraline", "escitalopram"): {
        "severity": "CRITICAL",
        "en": "Two SSRIs together: Serotonin syndrome risk — potentially fatal.",
        "ar": "🚫 مضادي اكتئاب SSRI معاً: خطر متلازمة السيروتونين — قد يكون مميتاً."
    },
    ("sertraline", "fluoxetine"): {
        "severity": "CRITICAL",
        "en": "Two SSRIs together: Serotonin syndrome risk — potentially fatal.",
        "ar": "🚫 مضادي اكتئاب SSRI معاً: خطر متلازمة السيروتونين — قد يكون مميتاً."
    },
    ("escitalopram", "fluoxetine"): {
        "severity": "CRITICAL",
        "en": "Two SSRIs together: Serotonin syndrome risk — potentially fatal.",
        "ar": "🚫 مضادي اكتئاب SSRI معاً: خطر متلازمة السيروتونين — قد يكون مميتاً."
    },
    ("sertraline", "ibuprofen"): {
        "severity": "MODERATE",
        "en": "SSRI + NSAID: Increased risk of GI bleeding.",
        "ar": "⚠️ مضاد اكتئاب + مضاد التهاب: خطر متزايد للنزيف المعوي."
    },
    ("escitalopram", "ibuprofen"): {
        "severity": "MODERATE",
        "en": "SSRI + NSAID: Increased risk of GI bleeding.",
        "ar": "⚠️ مضاد اكتئاب + مضاد التهاب: خطر متزايد للنزيف المعوي."
    },

    # --- GI / PPI ---
    ("omeprazole", "methotrexate"): {
        "severity": "HIGH",
        "en": "PPI delays Methotrexate excretion — increased toxicity.",
        "ar": "⚠️ أوميبرازول يؤخر إخراج الميثوتريكسات — خطر السمية."
    },
    ("omeprazole", "amlodipine"): {
        "severity": "LOW",
        "en": "Omeprazole may slightly increase Amlodipine levels via CYP interaction.",
        "ar": "ℹ️ أوميبرازول قد يزيد مستوى الأملوديبين بشكل طفيف."
    },

    # --- Respiratory ---
    ("salbutamol", "metoprolol"): {
        "severity": "HIGH",
        "en": "Beta-blocker opposes Salbutamol bronchodilation — may worsen asthma.",
        "ar": "⚠️ بيتا بلوكر يعاكس موسع الشعب الهوائية — قد يزيد نوبات الربو."
    },
    ("montelukast", "fluoxetine"): {
        "severity": "MODERATE",
        "en": "Fluoxetine may increase Montelukast levels. Monitor for neuropsychiatric effects.",
        "ar": "⚠️ فلوكستين قد يزيد مستوى المونتيلوكاست. راقب التأثيرات النفسية."
    },
}

# ============================================================
# 2. DRUG-ALLERGY DATABASE
#    Maps allergy names to drugs that should be avoided
#    Includes cross-reactivity (e.g., Penicillin → Amoxicillin)
# ============================================================

ALLERGY_DRUG_MAP = {
    "penicillin": {
        "drugs": ["amoxicillin", "ampicillin", "penicillin", "piperacillin"],
        "cross_reactive": ["cephalosporins"],
        "severity": "CRITICAL",
        "en": "Patient is allergic to Penicillin. This drug belongs to the same family — risk of anaphylaxis.",
        "ar": "🚫 المريض لديه حساسية من البنسلين. هذا الدواء من نفس العائلة — خطر صدمة حساسية."
    },
    "nuts": {
        "drugs": [],
        "cross_reactive": [],
        "severity": "LOW",
        "en": "Patient has nut allergy. No direct drug conflict, but check excipients in oral medications.",
        "ar": "ℹ️ المريض لديه حساسية من المكسرات. تحقق من مكونات الأدوية الفموية."
    },
    "sulfa": {
        "drugs": ["sulfamethoxazole", "sulfasalazine", "celecoxib"],
        "cross_reactive": ["thiazides"],
        "severity": "HIGH",
        "en": "Patient is allergic to Sulfa drugs. Risk of severe allergic reaction.",
        "ar": "⚠️ المريض لديه حساسية من السلفا. خطر رد فعل تحسسي شديد."
    },
    "aspirin": {
        "drugs": ["aspirin", "ibuprofen", "diclofenac", "celecoxib"],
        "cross_reactive": ["nsaids"],
        "severity": "HIGH",
        "en": "Patient is allergic to Aspirin. Cross-reactivity with other NSAIDs is common.",
        "ar": "⚠️ المريض لديه حساسية من الأسبرين. تفاعل متبادل مع مضادات الالتهاب شائع."
    },
    "ibuprofen": {
        "drugs": ["ibuprofen", "diclofenac", "celecoxib"],
        "cross_reactive": ["nsaids"],
        "severity": "HIGH",
        "en": "Patient is allergic to Ibuprofen. Avoid all NSAIDs.",
        "ar": "⚠️ المريض لديه حساسية من الإيبوبروفين. تجنب كل مضادات الالتهاب."
    },
}

# ============================================================
# 3. DRUG-CONDITION CONFLICT DATABASE
#    Maps chronic conditions to drugs that are contraindicated
# ============================================================

CONDITION_DRUG_MAP = {
    "asthma": {
        "drugs": ["metoprolol", "propranolol", "atenolol", "carvedilol"],
        "drug_class": "Beta-blockers",
        "severity": "CRITICAL",
        "en": "Beta-blockers are contraindicated in asthma — risk of severe bronchospasm.",
        "ar": "🚫 حاصرات بيتا ممنوعة مع الربو — خطر تشنج شعبي حاد."
    },
    "kidney disease": {
        "drugs": ["metformin", "ibuprofen", "diclofenac", "celecoxib", "enalapril"],
        "drug_class": "Nephrotoxic drugs",
        "severity": "HIGH",
        "en": "This drug may worsen kidney function. Dose adjustment or alternative needed.",
        "ar": "⚠️ هذا الدواء قد يزيد سوء وظائف الكلى. يحتاج تعديل جرعة أو بديل."
    },
    "liver disease": {
        "drugs": ["paracetamol", "methotrexate", "fluconazole", "statins"],
        "drug_class": "Hepatotoxic drugs",
        "severity": "HIGH",
        "en": "This drug is hepatotoxic. Use with extreme caution or avoid.",
        "ar": "⚠️ هذا الدواء سام للكبد. استخدم بحذر شديد أو تجنبه."
    },
    "peptic ulcer": {
        "drugs": ["ibuprofen", "diclofenac", "celecoxib", "aspirin"],
        "drug_class": "NSAIDs",
        "severity": "CRITICAL",
        "en": "NSAIDs contraindicated with peptic ulcer — risk of GI bleeding.",
        "ar": "🚫 مضادات الالتهاب ممنوعة مع قرحة المعدة — خطر نزيف معوي."
    },
    "heart failure": {
        "drugs": ["amlodipine", "ibuprofen", "diclofenac", "celecoxib", "metformin"],
        "drug_class": "Various",
        "severity": "HIGH",
        "en": "This drug may worsen heart failure. Specialist consultation recommended.",
        "ar": "⚠️ هذا الدواء قد يزيد سوء قصور القلب. يُوصى باستشارة متخصص."
    },
    "pregnancy": {
        "drugs": ["methotrexate", "enalapril", "losartan", "ibuprofen", "diclofenac",
                  "celecoxib", "fluoxetine", "sertraline", "escitalopram", "ciprofloxacin",
                  "empagliflozin", "sitagliptin"],
        "drug_class": "Teratogenic / Contraindicated",
        "severity": "CRITICAL",
        "en": "This drug is contraindicated in pregnancy — risk of birth defects.",
        "ar": "🚫 هذا الدواء ممنوع أثناء الحمل — خطر تشوهات جنينية."
    },
    "diabetes": {
        "drugs": ["prednisone", "dexamethasone", "hydrochlorothiazide"],
        "drug_class": "Hyperglycemia-inducing",
        "severity": "MODERATE",
        "en": "This drug may increase blood sugar. Monitor glucose closely.",
        "ar": "⚠️ هذا الدواء قد يرفع السكر. راقب مستوى الجلوكوز بعناية."
    },
}

# ============================================================
# 4. DRUG CLASS MAP — for duplicate therapy detection
# ============================================================

DRUG_CLASS_MAP = {
    "paracetamol": "Analgesic",
    "ibuprofen": "NSAID",
    "diclofenac": "NSAID",
    "celecoxib": "NSAID",
    "cetirizine": "Antihistamine",
    "loratadine": "Antihistamine",
    "fexofenadine": "Antihistamine",
    "amlodipine": "CCB",
    "enalapril": "ACE_Inhibitor",
    "losartan": "ARB",
    "metformin": "Biguanide",
    "sitagliptin": "DPP4_Inhibitor",
    "empagliflozin": "SGLT2_Inhibitor",
    "amoxicillin": "Penicillin_Antibiotic",
    "clarithromycin": "Macrolide_Antibiotic",
    "ciprofloxacin": "Fluoroquinolone",
    "omeprazole": "PPI",
    "pantoprazole": "PPI",
    "salbutamol": "SABA",
    "fluticasone": "ICS",
    "montelukast": "LTRA",
    "sertraline": "SSRI",
    "escitalopram": "SSRI",
    "fluoxetine": "SSRI",
    "methotrexate": "DMARD",
}


# ============================================================
# 5. MAIN CHECK FUNCTION
# ============================================================

def normalize_drug_name(name):
    """Extract generic name from drug string like 'Paracetamol 500mg' -> 'paracetamol'"""
    if not name:
        return ""
    # Remove dosage info (numbers + mg/ml/mcg etc)
    parts = name.lower().strip().split()
    clean_parts = []
    for p in parts:
        # Skip parts that are dosage-like
        if any(unit in p for unit in ['mg', 'ml', 'mcg', 'iu', 'tab', 'cap', 'inhaler']):
            continue
        if p.replace('.', '').replace(',', '').isdigit():
            continue
        if p in ['tablet', 'capsule', 'syrup', 'injection', 'cream', 'gel', 'drops']:
            continue
        clean_parts.append(p)
    
    result = ' '.join(clean_parts).strip()
    
    # Handle combination drugs
    if '+' in name.lower():
        # Return both components
        return result
    
    return result if result else name.lower().strip()


def check_drug_interactions(medications, patient_allergies=None, patient_conditions=None):
    """
    Main function to check all drug safety concerns.
    
    Args:
        medications: list of dicts [{name: "Paracetamol 500mg", dosage: "...", ...}]
        patient_allergies: list of strings ["Penicillin", "Nuts"]
        patient_conditions: list of strings ["Asthma", "Diabetes"]
    
    Returns:
        dict with:
        - interactions: list of drug-drug interactions found
        - allergy_conflicts: list of allergy conflicts
        - condition_conflicts: list of condition conflicts  
        - duplicate_therapy: list of duplicate class warnings
        - summary: {critical: N, high: N, moderate: N, low: N}
        - safe: bool (True if no CRITICAL or HIGH issues)
    """
    if not medications:
        return {"interactions": [], "allergy_conflicts": [], "condition_conflicts": [],
                "duplicate_therapy": [], "summary": {"critical": 0, "high": 0, "moderate": 0, "low": 0}, "safe": True}

    allergies = [a.lower().strip() for a in (patient_allergies or [])]
    conditions = [c.lower().strip() for c in (patient_conditions or [])]
    
    drug_names = []
    for med in medications:
        name = med.get('name', '') if isinstance(med, dict) else str(med)
        normalized = normalize_drug_name(name)
        drug_names.append(normalized)
    
    results = {
        "interactions": [],
        "allergy_conflicts": [],
        "condition_conflicts": [],
        "duplicate_therapy": [],
        "summary": {"critical": 0, "high": 0, "moderate": 0, "low": 0}
    }

    # --- 1. Check Drug-Drug Interactions ---
    for i in range(len(drug_names)):
        for j in range(i + 1, len(drug_names)):
            pair = (drug_names[i], drug_names[j])
            pair_rev = (drug_names[j], drug_names[i])
            
            interaction = DRUG_DRUG_INTERACTIONS.get(pair) or DRUG_DRUG_INTERACTIONS.get(pair_rev)
            if interaction:
                results["interactions"].append({
                    "drug_a": medications[i].get('name', drug_names[i]) if isinstance(medications[i], dict) else drug_names[i],
                    "drug_b": medications[j].get('name', drug_names[j]) if isinstance(medications[j], dict) else drug_names[j],
                    "severity": interaction["severity"],
                    "message_en": interaction["en"],
                    "message_ar": interaction["ar"]
                })
                results["summary"][interaction["severity"].lower()] += 1

    # --- 2. Check Drug-Allergy Conflicts ---
    for allergy in allergies:
        allergy_info = ALLERGY_DRUG_MAP.get(allergy)
        if not allergy_info:
            continue
        
        for idx, drug in enumerate(drug_names):
            if drug in allergy_info["drugs"]:
                med_name = medications[idx].get('name', drug) if isinstance(medications[idx], dict) else drug
                results["allergy_conflicts"].append({
                    "drug": med_name,
                    "allergy": allergy.title(),
                    "severity": allergy_info["severity"],
                    "message_en": allergy_info["en"],
                    "message_ar": allergy_info["ar"]
                })
                results["summary"][allergy_info["severity"].lower()] += 1

    # --- 3. Check Drug-Condition Conflicts ---
    for condition in conditions:
        condition_info = CONDITION_DRUG_MAP.get(condition)
        if not condition_info:
            continue
        
        for idx, drug in enumerate(drug_names):
            if drug in condition_info["drugs"]:
                med_name = medications[idx].get('name', drug) if isinstance(medications[idx], dict) else drug
                results["condition_conflicts"].append({
                    "drug": med_name,
                    "condition": condition.title(),
                    "severity": condition_info["severity"],
                    "message_en": condition_info["en"],
                    "message_ar": condition_info["ar"]
                })
                results["summary"][condition_info["severity"].lower()] += 1

    # --- 4. Check Duplicate Therapy ---
    class_usage = {}
    for idx, drug in enumerate(drug_names):
        drug_class = DRUG_CLASS_MAP.get(drug)
        if drug_class:
            if drug_class in class_usage:
                med_name_a = medications[class_usage[drug_class]].get('name', '') if isinstance(medications[class_usage[drug_class]], dict) else drug_names[class_usage[drug_class]]
                med_name_b = medications[idx].get('name', '') if isinstance(medications[idx], dict) else drug
                results["duplicate_therapy"].append({
                    "drug_a": med_name_a,
                    "drug_b": med_name_b,
                    "drug_class": drug_class,
                    "severity": "MODERATE",
                    "message_en": f"Duplicate therapy: Both {med_name_a} and {med_name_b} belong to '{drug_class}' class.",
                    "message_ar": f"⚠️ علاج مكرر: كلا الدواءين {med_name_a} و {med_name_b} من نفس فئة '{drug_class}'."
                })
                results["summary"]["moderate"] += 1
            else:
                class_usage[drug_class] = idx

    # --- Final Safety Assessment ---
    results["safe"] = results["summary"]["critical"] == 0 and results["summary"]["high"] == 0

    return results


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # Test 1: Penicillin allergy + Amoxicillin prescription
    print("=== Test 1: Penicillin allergy + Amoxicillin ===")
    r = check_drug_interactions(
        medications=[{"name": "Amoxicillin 500mg"}, {"name": "Paracetamol 500mg"}],
        patient_allergies=["Penicillin"],
        patient_conditions=[]
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
    
    # Test 2: Two NSAIDs together
    print("\n=== Test 2: Two NSAIDs ===")
    r = check_drug_interactions(
        medications=[{"name": "Ibuprofen 400mg"}, {"name": "Diclofenac 50mg"}],
        patient_allergies=[],
        patient_conditions=[]
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
    
    # Test 3: Asthma patient + Beta-blocker
    print("\n=== Test 3: Asthma + Beta-blocker ===")
    r = check_drug_interactions(
        medications=[{"name": "Salbutamol Inhaler"}, {"name": "Metoprolol 50mg"}],
        patient_allergies=[],
        patient_conditions=["Asthma"]
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
    
    # Test 4: Two SSRIs
    print("\n=== Test 4: Two SSRIs ===")
    r = check_drug_interactions(
        medications=[{"name": "Sertraline 50mg"}, {"name": "Escitalopram 10mg"}],
        patient_allergies=[],
        patient_conditions=[]
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
