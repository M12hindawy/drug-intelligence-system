# -*- coding: utf-8 -*-
"""
MedLink AI — Clinical Decision Support System (CDSS)
Evidence-based clinical decision support for physicians.
"""
import os
import json


SYMPTOM_DISEASE_MAP = {
    # Cardiovascular
    "chest pain": {"diseases": ["Acute MI", "Angina Pectoris", "GERD", "Pulmonary Embolism", "Aortic Dissection"], "urgency": "emergency"},
    "ألم صدر": {"diseases": ["احتشاء قلبي حاد", "ذبحة صدرية", "انسداد رئوي"], "urgency": "emergency"},
    "palpitations": {"diseases": ["Atrial Fibrillation", "SVT", "Anxiety", "Hyperthyroidism"], "urgency": "urgent"},
    "خفقان": {"diseases": ["رجفان أذيني", "فرط الدرقية", "قلق"], "urgency": "urgent"},
    "shortness of breath": {"diseases": ["Heart Failure", "COPD Exacerbation", "Asthma", "Pneumonia", "PE"], "urgency": "urgent"},
    "ضيق تنفس": {"diseases": ["قصور قلبي", "تفاقم COPD", "نوبة ربو", "التهاب رئوي"], "urgency": "urgent"},
    
    # Neurological
    "severe headache": {"diseases": ["Subarachnoid Hemorrhage", "Migraine", "Meningitis", "Hypertensive Crisis"], "urgency": "emergency"},
    "صداع شديد مفاجئ": {"diseases": ["نزيف تحت العنكبوتية", "شقيقة", "التهاب السحايا", "أزمة ارتفاع ضغط"], "urgency": "emergency"},
    "dizziness": {"diseases": ["BPPV", "Vestibular Neuritis", "Orthostatic Hypotension", "Stroke"], "urgency": "urgent"},
    "دوخة": {"diseases": ["دوار الوضعة", "انخفاض ضغط انتصابي", "سكتة دماغية"], "urgency": "urgent"},
    "weakness one side": {"diseases": ["Stroke", "TIA", "Todd's Paralysis", "Hemiplegic Migraine"], "urgency": "emergency"},
    "شلل نصفي": {"diseases": ["سكتة دماغية", "نوبة نقص التروية العابرة TIA"], "urgency": "emergency"},
    
    # Gastrointestinal
    "abdominal pain": {"diseases": ["Appendicitis", "Peptic Ulcer", "IBS", "Cholecystitis", "Pancreatitis"], "urgency": "urgent"},
    "ألم بطن": {"diseases": ["التهاب زائدة", "قرحة هضمية", "التهاب المرارة", "التهاب بنكرياس"], "urgency": "urgent"},
    "vomiting blood": {"diseases": ["Upper GI Bleed", "Esophageal Varices", "Mallory-Weiss"], "urgency": "emergency"},
    "قيء دم": {"diseases": ["نزيف الجهاز الهضمي العلوي", "دوالي المريء"], "urgency": "emergency"},
    
    # Endocrine
    "polyuria polydipsia": {"diseases": ["Diabetes Mellitus", "Diabetes Insipidus", "Hypercalcemia"], "urgency": "non-urgent"},
    "كثرة البول والعطش": {"diseases": ["داء السكري", "السكري الكاذب", "ارتفاع الكالسيوم"], "urgency": "non-urgent"},
    "weight loss": {"diseases": ["Hyperthyroidism", "Malignancy", "Diabetes", "Malabsorption", "Depression"], "urgency": "urgent"},
    "نقص وزن": {"diseases": ["فرط الدرقية", "ورم خبيث", "سوء الامتصاص"], "urgency": "urgent"},
    
    # Respiratory
    "cough blood": {"diseases": ["TB", "Lung Cancer", "Bronchiectasis", "Pulmonary Embolism"], "urgency": "urgent"},
    "سعال دموي": {"diseases": ["سل رئوي", "سرطان رئة", "توسع قصبي"], "urgency": "urgent"},
    "fever cough": {"diseases": ["Pneumonia", "COVID-19", "Influenza", "TB"], "urgency": "urgent"},
    "حمى سعال": {"diseases": ["التهاب رئوي", "كوفيد-19", "إنفلونزا", "سل"], "urgency": "urgent"},
}

DRUG_PROTOCOLS = {
    "Acute MI": {
        "first_line": ["Aspirin 300mg stat", "Clopidogrel 600mg loading", "Heparin IV", "Nitrates (if no hypotension)", "Beta-blocker (if stable)"],
        "arabic": "أسبرين 300 مجم فورياً + كلوبيدوجريل 600 مجم + هيبارين + نترات",
        "contraindications": ["Aspirin allergy → use Clopidogrel alone", "Hypotension → avoid nitrates/beta-blockers"],
        "monitoring": ["ECG continuous", "Troponin q6h", "BP/HR monitoring", "Echo"],
        "referral": "Cardiology — urgent PCI within 90 minutes"
    },
    "Heart Failure": {
        "first_line": ["ACE inhibitor (Enalapril/Ramipril)", "Beta-blocker (Carvedilol/Bisoprolol)", "Loop diuretic (Furosemide)", "Spironolactone (if EF<35%)"],
        "arabic": "مثبط ACE + حاصر بيتا + مدر حلقي + سبيرونولاكتون",
        "contraindications": ["K+ >5.5 → avoid spironolactone", "HR<60 → caution with beta-blockers", "Cr>2.5 → caution with ACE-i"],
        "monitoring": ["Weight daily", "Electrolytes weekly initially", "Echo q6 months", "BNP"],
        "referral": "Cardiology for optimization"
    },
    "Diabetes Mellitus Type 2": {
        "first_line": ["Metformin 500mg BD (if eGFR>30)", "HbA1c target <7%", "Lifestyle modification", "If CVD: add Empagliflozin or Liraglutide"],
        "arabic": "ميتفورمين + تعديل نمط الحياة + إمباجليفلوزين لو عنده أمراض قلبية",
        "contraindications": ["eGFR<30 → stop Metformin", "DKA → Insulin only", "Renal failure → avoid SGLT2i"],
        "monitoring": ["HbA1c q3 months", "Renal function annually", "Lipids annually", "Eye exam annually", "Foot exam"],
        "referral": "Endocrinology if HbA1c >9% despite treatment"
    },
    "Hypertension": {
        "first_line": ["ACE inhibitor OR ARB (1st choice if DM/CKD)", "CCB (Amlodipine)", "Thiazide diuretic (HCTZ)", "Target BP <130/80 (DM/CKD) or <140/90"],
        "arabic": "مثبط ACE أو ARB + أملوديبين ± هيدروكلوروثيازيد",
        "contraindications": ["ACE-i in pregnancy → use Methyldopa/Nifedipine", "Beta-blockers in asthma/COPD", "ACE-i + ARB combination"],
        "monitoring": ["BP at every visit", "Electrolytes 1 month after starting ACEI", "Renal function", "ECG annually"],
        "referral": "Nephrology if CKD, Cardiology if target organ damage"
    },
    "Community Acquired Pneumonia": {
        "first_line": ["Outpatient mild: Amoxicillin 1g TID × 5-7d", "Atypical: Azithromycin 500mg OD × 5d", "Inpatient: Amoxicillin-clavulanate + Azithromycin", "Severe (ICU): Pip-taz + Azithromycin"],
        "arabic": "أموكسيسيلين (خفيف) أو أموكسيسيلين-كلافولانات + أزيثروميسين (دخول مستشفى)",
        "contraindications": ["Penicillin allergy → Levofloxacin", "Pregnancy → avoid Fluoroquinolones"],
        "monitoring": ["Temp/HR/RR/SpO2", "CXR at 6 weeks", "Blood cultures if admitted", "CURB-65 score"],
        "referral": "Admit if CURB-65 ≥2"
    },
    "Asthma Acute Exacerbation": {
        "first_line": ["Salbutamol (Ventolin) nebulization q20min × 3", "Ipratropium nebulization", "Systemic steroids (Prednisolone 40-60mg or Methylprednisolone IV)", "Oxygen to maintain SpO2 >92%"],
        "arabic": "سالبوتامول نيبيولايزر + إيبراتروبيوم + ستيرويد جهازي + أكسجين",
        "contraindications": ["Sedatives → contraindicated in acute asthma", "Non-selective beta-blockers"],
        "monitoring": ["Peak flow", "SpO2", "Respiratory rate", "ABG if severe"],
        "referral": "ICU if SpO2<90% despite treatment or silent chest"
    },
}


def analyze_symptoms(symptoms: str, patient_data: dict = None) -> dict:
    """
    Analyze symptoms and return possible diagnoses + treatment protocols.
    
    Args:
        symptoms: symptom description (Arabic or English)
        patient_data: dict with age, weight, allergies, current_meds, conditions
    
    Returns:
        dict with diagnoses, protocols, warnings, urgency level
    """
    symptoms_lower = symptoms.lower()
    patient_data = patient_data or {}
    
    # Find matching symptoms
    matched = []
    for symptom_key, data in SYMPTOM_DISEASE_MAP.items():
        if symptom_key in symptoms_lower:
            matched.append({
                'symptom': symptom_key,
                'diseases': data['diseases'],
                'urgency': data['urgency']
            })
    
    if not matched:
        return {
            'diagnoses': [],
            'urgency': 'non-urgent',
            'protocols': [],
            'warnings': ['يرجى وصف الأعراض بشكل أكثر تفصيلاً'],
            'disclaimer': '⚠️ هذا النظام مساعد للطبيب وليس بديلاً عن الفحص السريري.'
        }
    
    # Determine overall urgency (worst case)
    urgency_order = ['emergency', 'urgent', 'non-urgent']
    all_urgencies = [m['urgency'] for m in matched]
    overall_urgency = min(all_urgencies, key=lambda x: urgency_order.index(x))
    
    # Collect all possible diagnoses
    all_diagnoses = []
    for m in matched:
        all_diagnoses.extend(m['diseases'])
    all_diagnoses = list(dict.fromkeys(all_diagnoses))  # Remove duplicates preserving order
    
    # Get treatment protocols for top diagnoses
    protocols = []
    for disease in all_diagnoses[:3]:  # Top 3
        for protocol_key, protocol_data in DRUG_PROTOCOLS.items():
            if protocol_key.lower() in disease.lower() or disease.lower() in protocol_key.lower():
                protocols.append({
                    'disease': disease,
                    'protocol_name': protocol_key,
                    'first_line': protocol_data['first_line'],
                    'arabic_summary': protocol_data['arabic'],
                    'monitoring': protocol_data['monitoring'],
                    'referral': protocol_data['referral'],
                    'contraindications': protocol_data['contraindications']
                })
                break
    
    # Check for patient-specific warnings
    warnings = []
    allergies = patient_data.get('allergies', [])
    conditions = patient_data.get('conditions', [])
    
    if 'penicillin' in str(allergies).lower() or 'بنسلين' in str(allergies):
        warnings.append('⚠️ تحذير: حساسية من البنسلين — تجنب الأموكسيسيلين وجميع البيتا لاكتام')
    if 'sulfa' in str(allergies).lower():
        warnings.append('⚠️ تحذير: حساسية من السلفا — تجنب كوتريموكسازول والسلفاديازين')
    if 'aspirin' in str(allergies).lower():
        warnings.append('⚠️ تحذير: حساسية من الأسبرين — تجنب NSAIDs')
    if 'pregnancy' in str(conditions).lower() or 'حامل' in str(conditions):
        warnings.append('⚠️ المريضة حامل — راجع Category قبل وصف أي دواء')
    if 'renal' in str(conditions).lower() or 'كلى' in str(conditions):
        warnings.append('⚠️ قصور كلوي — تعديل الجرعات وتجنب NSAIDs/Metformin')
    if 'liver' in str(conditions).lower() or 'كبد' in str(conditions):
        warnings.append('⚠️ قصور كبدي — تجنب Paracetamol بجرعات عالية والستاتينات')
    
    return {
        'diagnoses': all_diagnoses[:6],
        'urgency': overall_urgency,
        'protocols': protocols,
        'warnings': warnings,
        'disclaimer': '⚠️ هذا النظام مساعد للطبيب فقط وليس بديلاً عن الفحص السريري والحكم الطبي المتخصص.',
        'urgency_label': {
            'emergency': '🔴 طارئ — التوجه الفوري للطوارئ',
            'urgent': '🟡 عاجل — يحتاج تقييم طبي خلال ساعات',
            'non-urgent': '🟢 غير طارئ — يمكن الزيارة المجدولة'
        }.get(overall_urgency, '⚪ غير محدد')
    }


def calculate_personalized_dosage(drug_name: str, weight_kg: float, age: int,
                                   renal_function: str = 'normal',
                                   hepatic_function: str = 'normal',
                                   is_pregnant: bool = False) -> dict:
    """
    Calculate personalized drug dosage based on patient parameters.
    """
    drug_lower = drug_name.lower().strip()
    
    # Drug dosage formulas
    dosage_db = {
        'amoxicillin': {
            'adult_dose': 500,  # mg
            'unit': 'mg',
            'frequency': 'كل 8 ساعات',
            'max_daily': 3000,
            'pediatric_formula': lambda w: f"{round(40 * w / 3)} mg كل 8 ساعات",
            'renal_adjustment': {'mild': '100%', 'moderate': '50%', 'severe': '25%'},
            'pregnancy': 'Category B — آمن نسبياً في الحمل'
        },
        'paracetamol': {
            'adult_dose': 1000,
            'unit': 'mg',
            'frequency': 'كل 6 ساعات',
            'max_daily': 4000,
            'pediatric_formula': lambda w: f"{round(15 * w)} mg كل 6 ساعات (max {round(15*w*4)}mg/day)",
            'renal_adjustment': {'mild': '100%', 'moderate': '100%', 'severe': 'كل 8 ساعات بدلاً من 6'},
            'hepatic_adjustment': 'تخفيض 50% في القصور الكبدي الشديد',
            'pregnancy': 'Category B — الخيار الأول في الحمل'
        },
        'ibuprofen': {
            'adult_dose': 400,
            'unit': 'mg',
            'frequency': 'كل 8 ساعات مع الطعام',
            'max_daily': 2400,
            'pediatric_formula': lambda w: f"{round(10 * w)} mg كل 6-8 ساعات",
            'renal_adjustment': {'mild': '100%', 'moderate': 'تجنب', 'severe': 'تجنب'},
            'pregnancy': 'Category C — تجنب في الثلث الثالث'
        },
        'metformin': {
            'adult_dose': 500,
            'unit': 'mg',
            'frequency': 'مرتين يومياً مع الطعام',
            'max_daily': 2550,
            'pediatric_formula': lambda w: f"{round(500)} mg مرتين يومياً (>10 سنوات)",
            'renal_adjustment': {'mild': '100%', 'moderate': '50%', 'severe': 'تجنب'},
            'pregnancy': 'Category B — يستخدم في سكري الحمل'
        },
        'furosemide': {
            'adult_dose': 40,
            'unit': 'mg',
            'frequency': 'مرة صباحاً',
            'max_daily': 600,
            'pediatric_formula': lambda w: f"{round(1 * w)} mg مرة أو مرتين يومياً",
            'renal_adjustment': {'mild': '100%', 'moderate': 'جرعات أعلى قد تلزم', 'severe': 'جرعات أعلى'},
            'pregnancy': 'Category C — استخدم بحذر'
        },
    }
    
    # Normalize drug name
    drug_key = None
    for key in dosage_db:
        if key in drug_lower or drug_lower in key:
            drug_key = key
            break
    
    if not drug_key:
        return {
            'drug': drug_name,
            'error': f'لا تتوفر معادلة جرعة مخصصة لـ {drug_name} في النظام',
            'recommendation': 'يرجى استشارة الصيدلاني لحساب الجرعة المناسبة'
        }
    
    drug = dosage_db[drug_key]
    
    # Base dose
    if age < 12:
        dose_str = drug['pediatric_formula'](weight_kg)
        dose_type = 'جرعة أطفال'
    else:
        dose_str = f"{drug['adult_dose']} {drug['unit']}"
        dose_type = 'جرعة البالغين'
    
    # Adjustments
    adjustments = []
    
    renal_adj = drug.get('renal_adjustment', {}).get(renal_function, '100%')
    if renal_adj != '100%':
        adjustments.append(f"تعديل كلوي: {renal_adj}")
    
    hepatic_adj = drug.get('hepatic_adjustment', '')
    if hepatic_function != 'normal' and hepatic_adj:
        adjustments.append(f"تعديل كبدي: {hepatic_adj}")
    
    pregnancy_note = drug.get('pregnancy', '')
    if is_pregnant and pregnancy_note:
        adjustments.append(f"في الحمل: {pregnancy_note}")
    
    return {
        'drug': drug_name,
        'dose_type': dose_type,
        'recommended_dose': dose_str,
        'frequency': drug['frequency'],
        'max_daily': f"{drug['max_daily']} {drug['unit']}/day",
        'adjustments': adjustments,
        'pregnancy_safety': drug.get('pregnancy', 'راجع طبيبك'),
        'disclaimer': '⚠️ هذه جرعة استرشادية. يجب مراجعة طبيب أو صيدلاني للتأكيد.'
    }


if __name__ == '__main__':
    # Test
    result = analyze_symptoms("chest pain and shortness of breath")
    print(f"Urgency: {result['urgency']}")
    print(f"Diagnoses: {result['diagnoses'][:3]}")
    
    dose = calculate_personalized_dosage('amoxicillin', weight_kg=70, age=35)
    print(f"Dose: {dose}")
