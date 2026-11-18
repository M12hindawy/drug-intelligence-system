"""
MedLink AI — Drug Name Detection (Arabic Dialect + Brand Names)
Maps 100+ common Arabic/Egyptian drug names to standard IDs
"""

# ─── Comprehensive Arabic/Brand → Standard Drug Name Map ─────────────────────
# Format: 'colloquial/brand/arabic' → 'standard_english_name'

DRUG_NAME_MAP = {
    # ─── Analgesics / Pain Relief ────────────────────────────────────────────
    'بنادول': 'paracetamol',
    'بانادول': 'paracetamol',
    'panadol': 'paracetamol',
    'paracetamol': 'paracetamol',
    'باراسيتامول': 'paracetamol',
    'أدول': 'paracetamol',
    'adol': 'paracetamol',
    'تايلينول': 'paracetamol',
    'tylenol': 'paracetamol',

    'بروفين': 'ibuprofen',
    'ibuprofen': 'ibuprofen',
    'ايبوبروفين': 'ibuprofen',
    'إيبوبروفين': 'ibuprofen',
    'نيوروفين': 'ibuprofen',
    'nurofen': 'ibuprofen',
    'brufen': 'ibuprofen',
    'برفين': 'ibuprofen',

    'فولتارين': 'diclofenac',
    'voltaren': 'diclofenac',
    'ديكلوفيناك': 'diclofenac',
    'diclofenac': 'diclofenac',
    'كتوفان': 'diclofenac',

    'أسبرين': 'aspirin',
    'اسبرين': 'aspirin',
    'aspirin': 'aspirin',
    'أكو': 'aspirin',

    'ليريكا': 'pregabalin',
    'lyrica': 'pregabalin',
    'بريجابالين': 'pregabalin',
    'pregabalin': 'pregabalin',

    # ─── Antibiotics ─────────────────────────────────────────────────────────
    'أموكسيل': 'amoxicillin',
    'اموكسيل': 'amoxicillin',
    'amoxil': 'amoxicillin',
    'أموكسيسيلين': 'amoxicillin',
    'اموكسيسيلين': 'amoxicillin',
    'amoxicillin': 'amoxicillin',
    'كلافوم': 'amoxicillin',

    'فلاجيل': 'metronidazole',
    'flagyl': 'metronidazole',
    'ميترونيدازول': 'metronidazole',
    'metronidazole': 'metronidazole',

    'زيثروماكس': 'azithromycin',
    'zithromax': 'azithromycin',
    'ازيثروميسين': 'azithromycin',
    'azithromycin': 'azithromycin',
    'أزيتروسين': 'azithromycin',

    'سيبروفلوكساسين': 'ciprofloxacin',
    'ciprofloxacin': 'ciprofloxacin',
    'سيبروكسين': 'ciprofloxacin',
    'ciproxin': 'ciprofloxacin',
    'حبة زرقا': 'ciprofloxacin',

    'دوكسيسيكلين': 'doxycycline',
    'doxycycline': 'doxycycline',
    'دوريكسات': 'doxycycline',

    'كلاريثروميسين': 'clarithromycin',
    'clarithromycin': 'clarithromycin',
    'كلاريد': 'clarithromycin',

    # ─── Antihistamines / Allergy ─────────────────────────────────────────────
    'زيرتك': 'cetirizine',
    'zyrtec': 'cetirizine',
    'سيتيريزين': 'cetirizine',
    'cetirizine': 'cetirizine',

    'كلاريتين': 'loratadine',
    'claritin': 'loratadine',
    'لوراتادين': 'loratadine',
    'loratadine': 'loratadine',

    'بيريتون': 'chlorpheniramine',
    'piriton': 'chlorpheniramine',
    'كلورفينيرامين': 'chlorpheniramine',

    # ─── GI / Stomach ────────────────────────────────────────────────────────
    'لوسيك': 'omeprazole',
    'losec': 'omeprazole',
    'اوميبرازول': 'omeprazole',
    'أوميبرازول': 'omeprazole',
    'omeprazole': 'omeprazole',
    'أوميز': 'omeprazole',

    'زوفران': 'ondansetron',
    'zofran': 'ondansetron',
    'اوندانسيترون': 'ondansetron',
    'ondansetron': 'ondansetron',

    'بريميران': 'metoclopramide',
    'primperan': 'metoclopramide',
    'ميتوكلوبراميد': 'metoclopramide',
    'metoclopramide': 'metoclopramide',

    'رانيتيدين': 'ranitidine',
    'ranitidine': 'ranitidine',
    'زانتاك': 'ranitidine',
    'zantac': 'ranitidine',

    # ─── Cardiovascular ──────────────────────────────────────────────────────
    'نورفاسك': 'amlodipine',
    'norvasc': 'amlodipine',
    'أملوديبين': 'amlodipine',
    'amlodipine': 'amlodipine',

    'وارفارين': 'warfarin',
    'warfarin': 'warfarin',
    'كومادين': 'warfarin',
    'coumadin': 'warfarin',

    'ليبيتور': 'atorvastatin',
    'lipitor': 'atorvastatin',
    'اتورفاستاتين': 'atorvastatin',
    'atorvastatin': 'atorvastatin',

    'ميتوبرولول': 'metoprolol',
    'metoprolol': 'metoprolol',
    'بيتالوك': 'metoprolol',
    'betaloc': 'metoprolol',

    'كابريل': 'captopril',
    'capril': 'captopril',
    'كابتوبريل': 'captopril',
    'captopril': 'captopril',

    # ─── Diabetes ────────────────────────────────────────────────────────────
    'جلوكوفاج': 'metformin',
    'glucophage': 'metformin',
    'ميتفورمين': 'metformin',
    'metformin': 'metformin',

    'جليبنكلاميد': 'glibenclamide',
    'glibenclamide': 'glibenclamide',
    'دياويل': 'glibenclamide',

    # ─── Respiratory / Asthma ────────────────────────────────────────────────
    'فنتولين': 'salbutamol',
    'ventolin': 'salbutamol',
    'سالبوتامول': 'salbutamol',
    'salbutamol': 'salbutamol',
    'بريكانيل': 'terbutaline',

    'سينجولار': 'montelukast',
    'singulair': 'montelukast',
    'مونتيلوكاست': 'montelukast',
    'montelukast': 'montelukast',

    # ─── Antifungals ─────────────────────────────────────────────────────────
    'ديفلوكان': 'fluconazole',
    'diflucan': 'fluconazole',
    'فلوكونازول': 'fluconazole',
    'fluconazole': 'fluconazole',

    'كانستن': 'clotrimazole',
    'canesten': 'clotrimazole',
    'كلوتريمازول': 'clotrimazole',
    'clotrimazole': 'clotrimazole',

    # ─── Psychiatric / Neurological ───────────────────────────────────────────
    'زولوفت': 'sertraline',
    'zoloft': 'sertraline',
    'سيرترالين': 'sertraline',
    'sertraline': 'sertraline',

    'زاناكس': 'alprazolam',
    'xanax': 'alprazolam',
    'ألبرازولام': 'alprazolam',
    'alprazolam': 'alprazolam',

    'فاليوم': 'diazepam',
    'valium': 'diazepam',
    'ديازيبام': 'diazepam',
    'diazepam': 'diazepam',

    # ─── Thyroid ─────────────────────────────────────────────────────────────
    'ايلتروكسين': 'levothyroxine',
    'eltroxin': 'levothyroxine',
    'ليفوثيروكسين': 'levothyroxine',
    'levothyroxine': 'levothyroxine',
    'ثيروكسين': 'levothyroxine',
    'thyroxine': 'levothyroxine',

    # ─── Steroids ────────────────────────────────────────────────────────────
    'هيدروكورتيزون': 'hydrocortisone',
    'hydrocortisone': 'hydrocortisone',
    'كورتيزون': 'hydrocortisone',
    'cortisone': 'hydrocortisone',
    'ديكساميثازون': 'dexamethasone',
    'dexamethasone': 'dexamethasone',
    'ديكادرون': 'dexamethasone',

    # ─── Supplements ─────────────────────────────────────────────────────────
    'فيتامين د': 'vitamin_d',
    'فيتامين d': 'vitamin_d',
    'vitamin d': 'vitamin_d',
    'حديد': 'iron_supplement',
    'iron': 'iron_supplement',
    'فيتامين سي': 'vitamin_c',
    'vitamin c': 'vitamin_c',
    'كالسيوم': 'calcium',
    'calcium': 'calcium',
}


def normalize_drug_name(query: str) -> list[str]:
    """
    Detect drug names in a query (Arabic/English/Brand names).
    Returns list of standard drug IDs found in the query.
    """
    query_lower = query.lower().strip()
    found_drugs = []

    for keyword, standard_name in DRUG_NAME_MAP.items():
        if keyword.lower() in query_lower:
            if standard_name not in found_drugs:
                found_drugs.append(standard_name)

    return found_drugs


def get_drug_display_name(drug_id: str) -> str:
    """Get a human-readable display name for a drug ID."""
    display_names = {
        'paracetamol': 'Paracetamol (بنادول)',
        'ibuprofen': 'Ibuprofen (بروفين)',
        'diclofenac': 'Diclofenac (فولتارين)',
        'aspirin': 'Aspirin (أسبرين)',
        'amoxicillin': 'Amoxicillin (أموكسيل)',
        'metronidazole': 'Metronidazole (فلاجيل)',
        'azithromycin': 'Azithromycin (زيثروماكس)',
        'ciprofloxacin': 'Ciprofloxacin (سيبروكسين)',
        'omeprazole': 'Omeprazole (لوسيك)',
        'metformin': 'Metformin (جلوكوفاج)',
        'warfarin': 'Warfarin (وارفارين)',
        'atorvastatin': 'Atorvastatin (ليبيتور)',
        'amlodipine': 'Amlodipine (نورفاسك)',
        'salbutamol': 'Salbutamol (فنتولين)',
        'sertraline': 'Sertraline (زولوفت)',
        'alprazolam': 'Alprazolam (زاناكس)',
        'fluconazole': 'Fluconazole (ديفلوكان)',
        'levothyroxine': 'Levothyroxine (ايلتروكسين)',
        'cetirizine': 'Cetirizine (زيرتك)',
        'loratadine': 'Loratadine (كلاريتين)',
        'pregabalin': 'Pregabalin (ليريكا)',
    }
    return display_names.get(drug_id, drug_id.replace('_', ' ').title())


# ─── Safety Classification ─────────────────────────────────────────────────────
CONTROLLED_DRUGS = {
    'alprazolam', 'diazepam', 'pregabalin', 'tramadol',
    'codeine', 'morphine', 'oxycodone'
}

DANGEROUS_OTC = {
    'warfarin',      # Blood thinner — needs monitoring
    'methotrexate',  # Chemotherapy
    'digoxin',       # Heart — narrow therapeutic window
    'lithium',       # Psychiatric — narrow window
    'phenytoin',     # Anti-seizure
}

HIGH_INTERACTION_RISK = {
    'warfarin', 'aspirin', 'metformin', 'digoxin',
    'sertraline', 'alprazolam', 'methotrexate'
}


def classify_drug_safety(drug_ids: list[str]) -> dict:
    """
    Classify query safety level based on detected drugs.
    Returns: { 'level': 'safe'|'caution'|'danger', 'reason': str, 'requires_doctor': bool }
    """
    if not drug_ids:
        return {'level': 'safe', 'reason': 'No specific drugs detected', 'requires_doctor': False}

    # Check for controlled substances
    controlled_found = [d for d in drug_ids if d in CONTROLLED_DRUGS]
    if controlled_found:
        return {
            'level': 'danger',
            'reason': f"الأدوية المنضبطة — تحتاج وصفة طبية: {', '.join(controlled_found)}",
            'requires_doctor': True
        }

    # Check dangerous OTC
    dangerous_found = [d for d in drug_ids if d in DANGEROUS_OTC]
    if dangerous_found:
        return {
            'level': 'caution',
            'reason': f"أدوية تحتاج متابعة طبية: {', '.join(dangerous_found)}",
            'requires_doctor': True
        }

    # Check high interaction risk
    high_risk = [d for d in drug_ids if d in HIGH_INTERACTION_RISK]
    if len(drug_ids) > 1 and high_risk:
        return {
            'level': 'caution',
            'reason': f"تفاعلات دوائية محتملة — راجع الصيدلاني",
            'requires_doctor': False
        }

    return {'level': 'safe', 'reason': 'General drug information query', 'requires_doctor': False}
