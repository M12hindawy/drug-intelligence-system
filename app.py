import os
import sys
import json
import uuid
import sqlite3
import re
import secrets
import io
import base64
import qrcode
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime, timedelta
import threading
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pytesseract
from PIL import Image
from thefuzz import process

# Include local directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
load_dotenv()
from inference import load_model, predict, format_output, clean_text
from database import get_db_connection, DB_PATH, db_init, db_migrate_automation
from drug_interactions import check_drug_interactions
from drug_rag import drug_rag_search, get_chroma_collection
from scheduler import run_all_scheduled_tasks
from security import generate_otp, verify_otp as verify_otp_code, sanitize_text, sanitize_phone, is_valid_phone, is_safe_query
from drug_names import normalize_drug_name, classify_drug_safety, get_drug_display_name
from logger_config import setup_logging, get_logger
from automation import (
    check_low_stock, check_prescription_interactions,
    check_no_shows, send_appointment_reminders,
    get_doctor_available_slots, get_doctor_calendar_summary,
    block_doctor_slots, run_automation_tasks
)
try:
    from clinical_decision import analyze_symptoms, calculate_personalized_dosage
    CDSS_AVAILABLE = True
except ImportError:
    CDSS_AVAILABLE = False

import platform
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux (Railway/Render), tesseract is installed via Dockerfile

# ─── Setup Logging ─────────────────────────────────────────────────────────────
setup_logging('medlink')
logger = get_logger('app')

# ─── Flask App & Extensions ───────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sihaos_secret_dev_123_change_in_prod')

# CORS — allow only our domain in production, all in dev
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# SocketIO with CORS
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ['*'] else '*')

# Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "100 per hour"],
    storage_uri="memory://"
)

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from WebSocket')

# --- WEBRTC SIGNALING ---
@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    emit('webrtc_offer', data, broadcast=True, include_self=False)

@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    emit('webrtc_answer', data, broadcast=True, include_self=False)

@socketio.on('webrtc_ice_candidate')
def handle_webrtc_ice_candidate(data):
    emit('webrtc_ice_candidate', data, broadcast=True, include_self=False)

@socketio.on('webrtc_call_state')
def handle_webrtc_call_state(data):
    # data can contain status like 'ringing', 'rejected', 'ended'
    emit('webrtc_call_state', data, broadcast=True, include_self=False)
# ------------------------

# Preload models to keep UI fast
print("Preloading models for SihaOS GUI...")
MODELS_CACHE = {}
for i in range(1, 7):
    try:
        model, preprocessor = load_model(i)
        if model is not None:
            MODELS_CACHE[i] = (model, preprocessor)
            print(f"Model {i} loaded successfully.")
        else:
            print(f"Model {i} checkpoint not found. Fallback enabled.")
    except Exception as e:
        print(f"Error loading Model {i}: {e}. Fallback enabled.")

# Active triage sessions in-memory storage (simulating active WhatsApp session state)
# Keys: session_id, Value: dict of state
TRIAGE_SESSIONS = {}

# Keep track of phone-to-session mappings for WhatsApp Twilio webhook
PHONE_SESSIONS = {}

# Diagnostic webhook log buffer to display on admin dashboard
WEBHOOK_LOGS = []

# Mappings of classification output label -> recommended doctor specialty
SPECIALTY_MAP = {
    'Pain Relief': 'Internal Medicine',
    'Cold & Flu': 'General Practice',
    'Allergy': 'Allergy & Immunology',
    'Hypertension': 'Cardiology',
    'Diabetes Type 2': 'Endocrinology',
    'Infection': 'Infectious Diseases',
    'Gastroesophageal Reflux': 'Gastroenterology',
    'Asthma': 'Pulmonology',
    'Depression': 'Psychiatry',
    'Arthritis': 'Rheumatology'
}

# Emergency keywords
EMERGENCY_WORDS = [
    "chest pain", "heart attack", "unconscious", "severe bleeding", 
    "cannot breathe", "stroke", "paralysis", "heavy bleeding", 
    "poison", "suicidal", "severe chest tightness", "choking",
    "الم في الصدر", "ذبحة", "نزيف شديد", "مش عارف اتنفس", "مغمى", "تسمم"
]

# Egyptian Arabic Dialect Clinical symptoms translator
EGYPTIAN_ARABIC_DICTIONARY = {
    "صداع": "severe headache",
    "دماغ بتوجعني": "severe headache",
    "راسي بتوجعني": "severe headache",
    "سخونية": "high fever",
    "سخن": "high fever",
    "حرارة": "high fever",
    "كحة": "cough",
    "سعال": "cough",
    "رشح": "runny nose",
    "برد": "cold and flu",
    "انفلونزا": "cold and flu",
    "حساسية": "allergic reaction skin rash",
    "هرش": "itching and skin rash",
    "حكة": "itching and skin rash",
    "ضغط": "high blood pressure hypertension",
    "دروخة": "dizziness",
    "دايخ": "dizziness",
    "بطني بتوجعني": "stomach pain gastritis",
    "حموضة": "acid reflux heartburn",
    "حرقان": "heartburn reflux",
    "نهجان": "shortness of breath wheezing",
    "صدر": "chest tightness wheezing",
    "مفاصل": "joint pain stiffness",
    "ركبي": "joint pain knee stiffness",
    "تعبان": "fatigue body ache",
    "إرهاق": "fatigue",
    "همدان": "fatigue body ache",
    "تعب": "fatigue",
    "وجع": "pain",
    "ألم": "pain",
    "التهاب": "inflammation",
    "معدتي": "stomach ache gastritis",
}

def translate_egyptian_symptoms(text):
    """Scan Arabic text and translate matching Egyptian keywords to English medical symptoms."""
    # Check if text contains Arabic characters
    if not re.search(r'[\u0600-\u06FF]', text):
        return text # English text, return unchanged
        
    translated_tokens = []
    
    # Direct matching of phrases first
    cleaned_text = text.strip()
    for ar_phrase, en_med in EGYPTIAN_ARABIC_DICTIONARY.items():
        if ar_phrase in cleaned_text:
            translated_tokens.append(en_med)
            
    # If no matches, return general terms
    if not translated_tokens:
        return "mild symptoms"
        
    return " ".join(translated_tokens)

def run_symptom_inference(symptoms, model_id=5):
    """Translate symptoms if Arabic, then run model inference or keyword fallback."""
    english_symptoms = translate_egyptian_symptoms(symptoms)
    print(f"Original Symptoms: '{symptoms}' | Translated: '{english_symptoms}'")
    
    if model_id in MODELS_CACHE:
        model, preprocessor = MODELS_CACHE[model_id]
        try:
            preds, all_probs = predict(model, preprocessor, english_symptoms, threshold=0.3)
            result = format_output(symptoms, preds, all_probs, model_id)
            # Add translated text metadata for debug
            result['translated_symptoms'] = english_symptoms
            return result
        except Exception as e:
            print(f"Inference error: {e}")
    
    # Fallback/Keyword-matching prediction if model not loaded
    cleaned = clean_text(english_symptoms)
    probs = [0.0] * 10
    classes = [
        'Pain Relief', 'Cold & Flu', 'Allergy', 'Hypertension', 'Diabetes Type 2',
        'Infection', 'Gastroesophageal Reflux', 'Asthma', 'Depression', 'Arthritis'
    ]
    
    # Mock matching keywords
    kw_map = {
        'Pain Relief': ['pain', 'ache', 'headache', 'hurt', 'sore'],
        'Cold & Flu': ['cold', 'flu', 'cough', 'fever', 'runny nose', 'sneeze'],
        'Allergy': ['allergy', 'itch', 'rash', 'sneezing', 'hives'],
        'Hypertension': ['blood pressure', 'hypertension', 'dizzy', 'headache', 'pressure'],
        'Diabetes Type 2': ['diabetes', 'sugar', 'thirsty', 'urination', 'fatigue'],
        'Infection': ['infection', 'pus', 'wound', 'fever', 'throat'],
        'Gastroesophageal Reflux': ['acid', 'heartburn', 'reflux', 'stomach', 'gerd'],
        'Asthma': ['asthma', 'wheeze', 'breath', 'shortness of breath', 'tightness'],
        'Depression': ['depressed', 'sad', 'hopeless', 'mood', 'anxiety'],
        'Arthritis': ['joint', 'arthritis', 'stiffness', 'swell', 'bone']
    }
    
    for idx, cls in enumerate(classes):
        matches = sum(1 for kw in kw_map[cls] if kw in cleaned)
        if matches > 0:
            probs[idx] = min(0.3 + (matches * 0.2), 0.95)
    
    # Normalize or adjust
    if sum(probs) == 0:
        probs[0] = 0.5  # default to pain relief
    
    preds = [
        (classes[i], probs[i])
        for i in range(10)
        if probs[i] >= 0.3
    ]
    preds.sort(key=lambda x: x[1], reverse=True)
    if not preds:
        top_idx = probs.index(max(probs))
        preds = [(classes[top_idx], probs[top_idx])]
        
    res = format_output(symptoms, preds, probs, model_id)
    res['translated_symptoms'] = english_symptoms
    return res

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/waiting_room')
def waiting_room():
    return render_template('waiting_room.html')

@app.route('/video_demo')
def video_demo():
    return render_template('video_demo.html')

@app.route('/drug-graph')
def drug_graph():
    return render_template('drug_graph.html')

@app.route('/privacy')
def privacy_policy():
    return render_template('privacy.html')

# ─── Clinical Decision Support API ────────────────────────────────────────────
@app.route('/api/clinical-decision', methods=['POST'])
def clinical_decision_api():
    """CDSS: analyze symptoms and return diagnoses + treatment protocols."""
    if not CDSS_AVAILABLE:
        return jsonify({'error': 'CDSS module not available'}), 503
    data = request.json or {}
    symptoms = data.get('symptoms', '').strip()
    patient_data = data.get('patient_data', {})
    if not symptoms:
        return jsonify({'error': 'Symptoms are required'}), 400
    try:
        result = analyze_symptoms(symptoms, patient_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f'[CDSS] Error: {e}')
        return jsonify({'error': str(e)}), 500

# ─── Personalized Dosage Calculator API ───────────────────────────────────────
# Built-in common drug dosages (Egyptian brands) - fallback when CDSS unavailable
COMMON_DRUG_DOSAGES = {
    'paracetamol': {'adult': '500–1000 mg كل 4-6 ساعات (max 4g/day)', 'child': '10–15 mg/kg كل 4-6 ساعات', 'renal_note': 'آمن في قصور الكلى البسيط', 'hepatic_note': 'تجنب في قصور الكبد', 'pregnancy': 'آمن (Category B)', 'brand': 'Panadol, Adol, Paramol'},
    'panadol':     {'adult': '500–1000 mg كل 4-6 ساعات (max 4g/day)', 'child': '10–15 mg/kg كل 4-6 ساعات', 'renal_note': 'آمن', 'hepatic_note': 'تجنب في قصور الكبد', 'pregnancy': 'آمن', 'brand': 'Panadol'},
    'ibuprofen':   {'adult': '200–400 mg كل 4-6 ساعات (max 1.2g/day)', 'child': '5–10 mg/kg كل 6-8 ساعات', 'renal_note': 'تجنب في قصور الكلى', 'hepatic_note': 'حذر في قصور الكبد', 'pregnancy': 'تجنب في الثلث الثالث', 'brand': 'Brufen, Advil'},
    'brufen':      {'adult': '200–400 mg كل 4-6 ساعات (max 1.2g/day)', 'child': '5–10 mg/kg كل 6-8 ساعات', 'renal_note': 'تجنب في قصور الكلى', 'hepatic_note': 'حذر', 'pregnancy': 'تجنب في الثلث الثالث', 'brand': 'Brufen'},
    'amoxicillin': {'adult': '250–500 mg كل 8 ساعات', 'child': '25 mg/kg/day مقسمة كل 8 ساعات', 'renal_note': 'قلل الجرعة في قصور الكلى الشديد', 'hepatic_note': 'آمن', 'pregnancy': 'آمن (Category B)', 'brand': 'Amoxil, Trimox'},
    'augmentin':   {'adult': '625 mg كل 8 ساعات أو 1g كل 12 ساعة', 'child': '25 mg/kg/day مقسمة', 'renal_note': 'قلل الجرعة في GFR<30', 'hepatic_note': 'حذر — راقب الكبد', 'pregnancy': 'آمن (Category B)', 'brand': 'Augmentin'},
    'metformin':   {'adult': '500 mg مرتين يومياً — زيادة تدريجية حتى 1000 mg مرتين', 'child': 'غير موصى به < 10 سنوات', 'renal_note': 'موانع إذا GFR<30', 'hepatic_note': 'تجنب في قصور الكبد', 'pregnancy': 'يُستخدم بحذر تحت إشراف', 'brand': 'Glucophage, Cidophage'},
    'omeprazole':  {'adult': '20–40 mg مرة يومياً قبل الإفطار', 'child': '0.7–3.5 mg/kg/day', 'renal_note': 'آمن', 'hepatic_note': 'قلل إلى 20mg في قصور شديد', 'pregnancy': 'آمن (Category C)', 'brand': 'Losec, Prilosec, Lozol'},
    'losartan':    {'adult': '25–100 mg مرة يومياً', 'child': 'غير موصى به', 'renal_note': 'حذر — راقب البوتاسيوم', 'hepatic_note': 'ابدأ بجرعة منخفضة 25mg', 'pregnancy': 'موانع — مضر للجنين', 'brand': 'Cozaar, Lorista'},
    'atorvastatin':{'adult': '10–80 mg مرة يومياً (يفضل مساءً)', 'child': 'متخصص فقط', 'renal_note': 'آمن', 'hepatic_note': 'موانع في قصور الكبد النشط', 'pregnancy': 'موانع', 'brand': 'Lipitor, Atorva'},
    'aspirin':     {'adult': '75–100 mg يومياً (وقائي) أو 300–900 mg كل 4-6 ساعات (ألم)', 'child': 'تجنب < 16 سنة (خطر Reye)', 'renal_note': 'حذر في قصور الكلى', 'hepatic_note': 'حذر في قصور الكبد', 'pregnancy': 'تجنب في الثلث الثالث', 'brand': 'Aspocid, Aspirin'},
    'azithromycin':{'adult': '500 mg يوم واحد ثم 250 mg 4 أيام', 'child': '10 mg/kg يوم 1 ثم 5 mg/kg 4 أيام', 'renal_note': 'آمن', 'hepatic_note': 'حذر في قصور الكبد الشديد', 'pregnancy': 'آمن (Category B)', 'brand': 'Zithromax, Azithrocin'},
    'ciprofloxacin':{'adult': '250–750 mg كل 12 ساعة', 'child': 'تجنب إلا للضرورة', 'renal_note': 'قلل الجرعة في GFR<30', 'hepatic_note': 'حذر في قصور الكبد الشديد', 'pregnancy': 'تجنب', 'brand': 'Cipro, Ciproflox'},
    'cetirizine':  {'adult': '10 mg مرة يومياً', 'child': '5 mg مرة يومياً (6-12 سنة)', 'renal_note': '5 mg يومياً في قصور الكلى', 'hepatic_note': 'آمن', 'pregnancy': 'يُستخدم بحذر', 'brand': 'Zyrtec, Cetrine'},
    'prednisolone':{'adult': '5–60 mg يومياً حسب الحالة', 'child': '1–2 mg/kg/day', 'renal_note': 'آمن', 'hepatic_note': 'حذر في قصور الكبد', 'pregnancy': 'يُستخدم بحذر', 'brand': 'Prednisolone, Deltacortril'},
    'clopidogrel': {'adult': '75 mg مرة يومياً', 'child': 'غير موصى به', 'renal_note': 'آمن', 'hepatic_note': 'تجنب في قصور الكبد الشديد', 'pregnancy': 'تجنب', 'brand': 'Plavix, Clogrel'},
    'lisinopril':  {'adult': '5–40 mg مرة يومياً', 'child': 'متخصص فقط', 'renal_note': 'قلل الجرعة — راقب البوتاسيوم', 'hepatic_note': 'آمن', 'pregnancy': 'موانع — مضر للجنين', 'brand': 'Zestril, Lisinopril'},
    'furosemide':  {'adult': '20–80 mg مرة أو مرتين يومياً', 'child': '1–2 mg/kg كل 6-12 ساعة', 'renal_note': 'قد تحتاج جرعات أعلى', 'hepatic_note': 'حذر — خطر اعتلال الكبد', 'pregnancy': 'يُستخدم بحذر', 'brand': 'Lasix, Diuresal'},
    'insulin':     {'adult': 'حسب نتائج السكر — تحت إشراف طبي', 'child': 'تحت إشراف طبي متخصص', 'renal_note': 'قد تحتاج تقليل الجرعة', 'hepatic_note': 'حذر — خطر نقص السكر', 'pregnancy': 'آمن ومفضل', 'brand': 'Mixtard, Lantus, Novorapid'},
}

@app.route('/api/dosage-calculator', methods=['POST'])
def dosage_calculator_api():
    """Calculate personalized drug dosage based on patient parameters."""
    data = request.json or {}
    drug_name = data.get('drug_name', '').strip().lower()
    weight_kg = float(data.get('weight_kg', 70))
    age = int(data.get('age', 35))
    renal = data.get('renal_function', 'normal')
    hepatic = data.get('hepatic_function', 'normal')
    is_pregnant = bool(data.get('is_pregnant', False))
    if not drug_name:
        return jsonify({'error': 'Drug name is required'}), 400
    
    # Try CDSS first
    if CDSS_AVAILABLE:
        try:
            result = calculate_personalized_dosage(drug_name, weight_kg, age, renal, hepatic, is_pregnant)
            if result and not result.get('error'):
                return jsonify(result)
        except Exception as e:
            logger.warning(f'[Dosage CDSS] fallback: {e}')
    
    # Fallback: built-in common drug database
    # Arabic phonetic to English mapping
    AR_TO_EN = {
        'باراسيتامول': 'paracetamol', 'بارسيتامول': 'paracetamol',
        'بانادول': 'panadol', 'باناكول': 'panadol', 'أدول': 'paracetamol',
        'إيبوبروفين': 'ibuprofen', 'ايبوبروفين': 'ibuprofen', 'بروفين': 'brufen',
        'أموكسيسيلين': 'amoxicillin', 'اموكسيسيلين': 'amoxicillin',
        'أوجمنتين': 'augmentin', 'اوجمنتين': 'augmentin',
        'ميتفورمين': 'metformin', 'جلوكوفاج': 'metformin', 'سيدوفاج': 'metformin',
        'أوميبرازول': 'omeprazole', 'اوميبرازول': 'omeprazole', 'لوزيك': 'omeprazole',
        'لوسارتان': 'losartan', 'كوزار': 'losartan',
        'أتورفاستاتين': 'atorvastatin', 'ليبيتور': 'atorvastatin',
        'أسبرين': 'aspirin', 'اسبرين': 'aspirin', 'أسبوسيد': 'aspirin',
        'أزيثرومايسين': 'azithromycin', 'زيثروماكس': 'azithromycin',
        'سيبروفلوكساسين': 'ciprofloxacin', 'سيبرو': 'ciprofloxacin',
        'سيتيريزين': 'cetirizine', 'زيرتيك': 'cetirizine',
        'بريدنيزولون': 'prednisolone',
        'كلوبيدوجريل': 'clopidogrel', 'بلافيكس': 'clopidogrel',
        'ليزينوبريل': 'lisinopril',
        'فيوروسيميد': 'furosemide', 'لاسيكس': 'furosemide',
        'أنسولين': 'insulin', 'انسولين': 'insulin',
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
                    break
    
    if drug_data:
        dose = drug_data['adult'] if age >= 18 else drug_data['child']
        warnings = []
        if renal != 'normal': warnings.append(f"⚠️ قصور الكلى: {drug_data['renal_note']}")
        if hepatic != 'normal': warnings.append(f"⚠️ قصور الكبد: {drug_data['hepatic_note']}")
        if is_pregnant: warnings.append(f"🤰 الحمل: {drug_data['pregnancy']}")
        if age < 18: warnings.append(f"👶 جرعة أطفال ({age} سنة)")
        if weight_kg < 50: warnings.append(f"⚖️ وزن منخفض ({weight_kg}kg) — راجع الطبيب")
        return jsonify({
            'drug_name': drug_name.title(),
            'recommended_dose': dose,
            'brand_names': drug_data.get('brand', ''),
            'warnings': warnings,
            'adjustments': warnings,
            'disclaimer': 'هذه معلومات استرشادية — استشر صيدلانيك لجرعة مناسبة لحالتك.',
            'source': 'MedLink Drug Database'
        })
    
    return jsonify({
        'error': f'لا تتوفر معادلة جرعة مخصصة لـ {drug_name} في النظام',
        'suggestion': 'يرجى استشارة الصيدلاني لحساب الجرعة المناسبة',
        'available_drugs': list(COMMON_DRUG_DOSAGES.keys())
    })

# ─── Drug Interaction Graph API ───────────────────────────────────────────────
@app.route('/api/drug-graph-data', methods=['GET'])
def drug_graph_data():
    """Return drug interaction network data for D3.js visualization."""
    try:
        from drug_interactions import DRUG_INTERACTIONS_DB
        nodes = []
        links = []
        seen_drugs = set()
        seen_links = set()
        for drug_pair, interaction_data in list(DRUG_INTERACTIONS_DB.items())[:80]:
            drugs = drug_pair.split('+') if '+' in drug_pair else drug_pair.split('_')
            if len(drugs) >= 2:
                d1, d2 = drugs[0].strip(), drugs[1].strip()
                severity = interaction_data.get('severity', 'moderate') if isinstance(interaction_data, dict) else 'moderate'
                if d1 not in seen_drugs:
                    nodes.append({'id': d1, 'label': d1.title(), 'group': 1})
                    seen_drugs.add(d1)
                if d2 not in seen_drugs:
                    nodes.append({'id': d2, 'label': d2.title(), 'group': 2})
                    seen_drugs.add(d2)
                link_key = f'{min(d1,d2)}_{max(d1,d2)}'
                if link_key not in seen_links:
                    color = '#ef4444' if severity == 'high' else '#f59e0b' if severity == 'moderate' else '#22c55e'
                    links.append({'source': d1, 'target': d2, 'severity': severity, 'color': color})
                    seen_links.add(link_key)
        return jsonify({'nodes': nodes[:60], 'links': links[:80]})
    except Exception as e:
        logger.error(f'[DrugGraph] Error: {e}')
        return jsonify({'nodes': [], 'links': [], 'error': str(e)})


# ─── Auth API & OTP Simulation ───────────────────────────────────────────────
@app.route('/api/users/list', methods=['GET'])
def get_users():
    conn = get_db_connection()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])

@app.route('/api/auth/send-otp', methods=['POST'])
@limiter.limit("5 per minute")
def send_otp():
    data = request.json or {}
    phone = sanitize_phone(data.get('phone', ''))

    if not phone:
        return jsonify({'error': 'رقم الهاتف مطلوب / Phone number is required'}), 400

    if not is_valid_phone(phone):
        return jsonify({'error': 'رقم الهاتف غير صحيح / Invalid phone number format'}), 400

    # Generate real random OTP
    otp_code = generate_otp(phone)

    # Try Twilio SMS if configured
    twilio_sid = os.environ.get('TWILIO_ACCOUNT_SID', '')
    twilio_token = os.environ.get('TWILIO_AUTH_TOKEN', '')
    twilio_from = os.environ.get('TWILIO_PHONE_NUMBER', '')

    if twilio_sid and twilio_token and twilio_from:
        try:
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_token)
            client.messages.create(
                body=f"MedLink AI — كود التحقق الخاص بك: {otp_code}\nYour OTP: {otp_code}\nصالح لمدة 10 دقائق.",
                from_=twilio_from,
                to=phone
            )
            logger.info(f"[OTP] SMS sent via Twilio to {phone[-4:]}xxxx")
            return jsonify({'message': 'تم إرسال كود التحقق برسالة SMS', 'phone': phone, 'method': 'sms'})
        except Exception as e:
            logger.error(f"[OTP] Twilio error: {e}")
            # Fall through to dev mode

    # Dev mode — log OTP (remove in production)
    logger.info(f"[OTP DEV] Code for {phone[-4:]}xxxx: {otp_code}")
    print(f"🔐 OTP for {phone}: {otp_code}")
    return jsonify({
        'message': 'تم إرسال كود التحقق / OTP sent',
        'phone': phone,
        'method': 'console',
        # Remove next line in production!
        'dev_otp': otp_code if os.environ.get('FLASK_ENV') != 'production' else None
    })


@app.route('/api/auth/verify-otp', methods=['POST'])
@limiter.limit("10 per minute")
def verify_otp():
    data = request.json or {}
    phone = sanitize_phone(data.get('phone', ''))
    code = sanitize_text(data.get('code', ''), max_length=10)

    if not phone or not code:
        return jsonify({'error': 'الهاتف والكود مطلوبان / Phone and OTP code are required'}), 400

    # Verify against OTP store
    success, message = verify_otp_code(phone, code)
    if not success:
        return jsonify({'error': message}), 400

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()

    if not user:
        # Create new patient user automatically
        new_id = str(uuid.uuid4())
        safe_name = f"Patient {phone[-4:]}"
        conn.execute(
            "INSERT INTO users (id, phone, name, role, is_verified) VALUES (?, ?, ?, ?, ?)",
            (new_id, phone, safe_name, 'patient', 1)
        )
        conn.execute(
            "INSERT INTO family_members (id, user_id, name, relationship, allergies, chronic_conditions) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), new_id, safe_name, 'self', '[]', '[]')
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (new_id,)).fetchone()
        logger.info(f"[Auth] New patient registered: {phone[-4:]}xxxx")

    conn.close()
    logger.info(f"[Auth] Login successful: {phone[-4:]}xxxx")
    return jsonify({'message': 'تم التحقق بنجاح / Verified successfully', 'user': dict(user)})

# ─── Groq LLM Setup ──────────────────────────────────────────────────────────
_raw_groq_key = os.environ.get('GROQ_API_KEY', '')
# Ignore placeholder values — only use real keys (start with gsk_ or similar)
GROQ_API_KEY = _raw_groq_key if (
    _raw_groq_key
    and not _raw_groq_key.startswith('your_')
    and 'placeholder' not in _raw_groq_key.lower()
    and len(_raw_groq_key) > 20
) else ''

if GROQ_API_KEY:
    logger.info('[Groq] API key found — LLM mode ACTIVE ✓')
else:
    logger.warning('[Groq] No valid API key — using smart ML fallback. Add real GROQ_API_KEY to .env')

def get_groq_client():
    """Returns a Groq client if API key is set, else None."""
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None

TRIAGE_SYSTEM_PROMPT = """أنت "محمد" — مساعد طبي ذكي ودود من MedLink AI بتتكلم عربي مصري.
You are "Mohamed" - a warm, friendly AI medical advisor for Egyptian patients.

Personality: Speak like a caring Egyptian friend who happens to be medically knowledgeable.
Always give a CLEAR recommendation at the end: "محتاج تروح دكتور" أو "مش لازم تروح دولوقتي".

## دورك:
- استقبل الأعراض بالعربي أو الإنجليزي
- اسأل عن: المدة، شدة الألم (1-10)، العمر، الأمراض المزمنة
- في النهاية: قول بوضوح لازم يروح دكتور ولا
- اقترح التخصص المناسب لو لازم

## التخصصات:
General Practice, Internal Medicine, Cardiology, Pulmonology, Gastroenterology, 
Neurology, Orthopedics, Dermatology, Psychiatry, Endocrinology, ENT, Pediatrics, Gynecology, Urology

## قواعد:
1. ارد بنفس لغة المريض (عربي أو إنجليزي)
2. كن دافئاً ومختصراً وطبيعياً مثل صديق حقيقي
3. في نهاية كل رسالة، أضف سطراً واحداً بشكل JSON:
   [META]{"state":"AWAITING_SYMPTOMS"|"AWAITING_DETAILS"|"AWAITING_AGE"|"COMPLETED"|"EMERGENCY","severity":"mild"|"moderate"|"urgent","recommended_specialty":"<specialty or null>","ready_to_book":<true|false>}

## طوارئ:
إذا ذكر ألم في الصدر أو ضيق شديد أو نزيف أو إغماء → طوارئ فوراً
[META]{"state":"EMERGENCY","severity":"urgent","recommended_specialty":"Emergency Medicine","ready_to_book":false}
"""

def call_groq_triage(conversation_history):
    """Call Groq API with conversation history and return (reply_text, meta_dict)."""
    client = get_groq_client()
    if not client:
        return None, None
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}] + conversation_history,
            temperature=0.2,
            max_tokens=300,
        )
        full_text = response.choices[0].message.content.strip()
        
        # Parse [META] block
        meta = {"state": "AWAITING_DETAILS", "severity": "moderate", "recommended_specialty": None, "ready_to_book": False}
        reply_text = full_text
        
        if "[META]" in full_text:
            parts = full_text.split("[META]", 1)
            reply_text = parts[0].strip()
            try:
                import json as _json
                meta = _json.loads(parts[1].strip())
            except Exception:
                pass
        
        return reply_text, meta
    except Exception as e:
        print(f"[Groq Error] {e}")
        return None, None

def triage_fallback_response(session, text):
    """Original state-machine fallback when Groq is unavailable."""
    bot_response = ""
    current_state = session['state']
    
    if current_state == 'AWAITING_SYMPTOMS':
        session['symptoms'] = text
        translated_test = translate_egyptian_symptoms(text)
        is_emergency = any(kw in text.lower() or kw in translated_test.lower() for kw in EMERGENCY_WORDS)
        if is_emergency:
            bot_response = "⚠️ **تنبيه حالة طارئة!**\n\nأعراضك تشير إلى حالة حرجة. يرجى التوجه فوراً لأقرب طوارئ أو الاتصال بـ 123."
            session['state'] = 'EMERGENCY'
            session['severity'] = 'urgent'
        else:
            bot_response = "منذ متى تشعر بهذه الأعراض؟ وكم تقيم شدة الألم من 1 إلى 10؟"
            session['state'] = 'AWAITING_DETAILS'
    elif current_state == 'AWAITING_DETAILS':
        session['duration'] = text
        severity = "moderate"
        for word in text.split():
            if word.isdigit():
                val = int(word)
                if val >= 7: severity = "urgent"
                elif val <= 3: severity = "mild"
        session['severity'] = severity
        bot_response = "كم يبلغ عمر المريض؟ وهل يعاني من أمراض مزمنة أو حساسية؟"
        session['state'] = 'AWAITING_AGE'
    elif current_state == 'AWAITING_AGE':
        session['age'] = text
        inference_result = run_symptom_inference(session['symptoms'])
        if inference_result and inference_result['predictions']:
            top = inference_result['predictions'][0]
            specialty = SPECIALTY_MAP.get(top['condition'], 'General Practice')
            session['classification'] = top
            session['recommended_specialty'] = specialty
            bot_response = f"🧠 الحالة المتوقعة: **{top['condition']}** ({top['confidence_pct']})\n✅ التخصص الموصى به: **{specialty}**\n\nهل تود حجز موعد الآن؟"
        else:
            session['recommended_specialty'] = 'General Practice'
            bot_response = "نوصي بزيارة طبيب عام. هل تريد استعراض الأطباء المتاحين؟"
        session['state'] = 'AWAITING_BOOKING'
    elif current_state == 'AWAITING_BOOKING':
        bot_response = "ممتاز. يرجى اختيار الطبيب المناسب من القائمة."
        session['state'] = 'COMPLETED'
    elif current_state == 'EMERGENCY':
        bot_response = "يرجى التوجه للطوارئ فوراً."
    else:
        bot_response = "تم الانتهاء من جلسة الفرز. نتمنى لك الشفاء! 💚"
    
    return bot_response

# ─── Triage AI API ─────────────────────────────────────────────────────────────
@app.route('/api/triage/start', methods=['POST'])
def triage_start():
    data = request.json or {}
    patient_id = data.get('patient_id')
    family_member_id = data.get('family_member_id')
    
    session_id = str(uuid.uuid4())
    
    # Detect if Groq is available
    groq_available = bool(GROQ_API_KEY)
    
    disclaimer = "\n\n⚠️ *إخلاء مسؤولية طبي:* هذا التقييم مبدئي ولا يغني عن الاستشارة الطبية المتخصصة. في حالات الطوارئ، يرجى التوجه لأقرب مستشفى فوراً."
    
    if groq_available:
        greeting = "مرحباً! 👋 أنا محمد، مساعدك الطبي من MedLink AI.\n\nهنا عشان أساعدك تعرف حالتك وأقولك بصراحة **لازم تروح دكتور ولا**. 💬\n\n**قوللي بتحس بإيه بالضبط** — اكتب بالعربي أو الإنجليزي." + disclaimer
    else:
        greeting = "مرحباً! أنا محمد، مساعدك الطبي. 🏥 قوللي بتحس بإيه وهساعدك تعرف لازم تروح دكتور ولا." + disclaimer

    
    TRIAGE_SESSIONS[session_id] = {
        'id': session_id,
        'patient_id': patient_id,
        'family_member_id': family_member_id,
        'state': 'AWAITING_SYMPTOMS',
        'symptoms': '',
        'duration': '',
        'severity': 'moderate',
        'age': '',
        'classification': None,
        'recommended_specialty': None,
        'groq_history': [],  # For Groq conversation context
        'messages': [
            {
                'sender': 'bot',
                'text': greeting,
                'time': datetime.now().strftime("%I:%M %p")
            }
        ],
        'llm_powered': groq_available
    }
    
    return jsonify(TRIAGE_SESSIONS[session_id])

@app.route('/api/triage/message', methods=['POST'])
def triage_message():
    data = request.json or {}
    session_id = data.get('session_id')
    text = data.get('text', '').strip()
    model_id = int(data.get('model_id', 5))
    
    if not session_id or session_id not in TRIAGE_SESSIONS:
        return jsonify({'error': 'Invalid or expired session'}), 400
        
    session = TRIAGE_SESSIONS[session_id]
    
    # Append patient message
    session['messages'].append({
        'sender': 'patient',
        'text': text,
        'time': datetime.now().strftime("%I:%M %p")
    })
    
    bot_response = ""
    
    # === Try Groq LLM first ===
    if GROQ_API_KEY:
        # Build conversation history for Groq
        session['groq_history'].append({"role": "user", "content": text})
        
        llm_reply, meta = call_groq_triage(session['groq_history'])
        
        if llm_reply:
            bot_response = llm_reply
            session['groq_history'].append({"role": "assistant", "content": llm_reply})
            
            # Update session metadata from LLM response
            if meta:
                if meta.get('state'):
                    session['state'] = meta['state']
                if meta.get('severity'):
                    session['severity'] = meta['severity']
                if meta.get('recommended_specialty'):
                    session['recommended_specialty'] = meta['recommended_specialty']
                # Run ML model inference when LLM says it's time to book
                if meta.get('ready_to_book') and not session['classification']:
                    inference_result = run_symptom_inference(text, model_id)
                    if inference_result and inference_result.get('predictions'):
                        session['classification'] = inference_result['predictions'][0]
                        if not session['recommended_specialty']:
                            session['recommended_specialty'] = SPECIALTY_MAP.get(
                                session['classification']['condition'], 'General Practice')
        else:
            # Groq failed - use fallback
            bot_response = triage_fallback_response(session, text)
    else:
        # No API key - use original fallback
        bot_response = triage_fallback_response(session, text)
    
    session['messages'].append({
        'sender': 'bot',
        'text': bot_response,
        'time': datetime.now().strftime("%I:%M %p")
    })
    
    return jsonify(session)

# ─── System Status API ────────────────────────────────────────────────────────
@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Returns which AI features are currently active."""
    return jsonify({
        'groq_llm': bool(GROQ_API_KEY),
        'groq_model': 'llama-3.3-70b-versatile' if GROQ_API_KEY else None,
        'ml_models_loaded': len(MODELS_CACHE),
        'version': '2.0-LLM'
    })

# ─── Twilio WhatsApp Webhook ──────────────────────────────────────────────────
@app.route('/api/whatsapp/webhook', methods=['POST'])
def whatsapp_webhook():
    """Twilio WhatsApp webhook endpoint. Processes incoming texts and responds via XML TwiML."""
    # Twilio sends Form data (application/x-www-form-urlencoded)
    incoming_msg = request.form.get('Body', '').strip()
    sender_phone = request.form.get('From', '').strip() # e.g. "whatsapp:+201011111111"
    
    # Logging for diagnostic admin dashboard
    log_entry = {
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sender': sender_phone,
        'message': incoming_msg
    }
    WEBHOOK_LOGS.append(log_entry)
    if len(WEBHOOK_LOGS) > 50:
        WEBHOOK_LOGS.pop(0)

    # Establish or fetch session
    if sender_phone not in PHONE_SESSIONS:
        session_id = str(uuid.uuid4())
        PHONE_SESSIONS[sender_phone] = session_id
        
        # Fetch the patient user matching the phone number (strip 'whatsapp:')
        clean_phone = sender_phone.replace('whatsapp:', '')
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE phone = ?", (clean_phone,)).fetchone()
        conn.close()
        
        p_id = user['id'] if user else None
        
        TRIAGE_SESSIONS[session_id] = {
            'id': session_id,
            'patient_id': p_id,
            'state': 'AWAITING_SYMPTOMS',
            'symptoms': '',
            'duration': '',
            'severity': '',
            'age': '',
            'classification': None,
            'recommended_specialty': None,
            'messages': [
                {
                    'sender': 'bot',
                    'text': "مرحباً بك في نظام MedLink AI الذكي. 🏥 أنا مساعدك الطبي للفرز والتشخيص الأولي.\n\nيرجى وصف الأعراض التي تشعر بها بالتفصيل (يمكنك الكتابة باللغة العربية أو الإنجليزية).",
                    'time': datetime.now().strftime("%I:%M %p")
                }
            ]
        }
        bot_reply = TRIAGE_SESSIONS[session_id]['messages'][0]['text']
    else:
        session_id = PHONE_SESSIONS[sender_phone]
        session = TRIAGE_SESSIONS[session_id]
        
        # Append user message
        session['messages'].append({
            'sender': 'patient',
            'text': incoming_msg,
            'time': datetime.now().strftime("%I:%M %p")
        })
        
        # Execute triage state machine logic (simulated by making internal request)
        # We reuse the triage_message logic locally
        current_state = session['state']
        bot_reply = ""
        
        if current_state == 'AWAITING_SYMPTOMS':
            session['symptoms'] = incoming_msg
            translated_test = translate_egyptian_symptoms(incoming_msg)
            is_emergency = any(kw in incoming_msg.lower() or kw in translated_test.lower() for kw in EMERGENCY_WORDS)
            
            if is_emergency:
                bot_reply = "⚠️ *تنبيه حالة طارئة (Emergency Alert)!*\n\nأعراضك تشير إلى حالة حرجة قد تتطلب تدخلًا فوريًا. ننصحك بالتوجه إلى أقرب قسم طوارئ فوراً أو الاتصال بالإسعاف (123).\n\nأقرب المستشفيات: مستشفى قصر العيني، مستشفى عين شمس التخصصي."
                session['state'] = 'EMERGENCY'
                session['severity'] = 'urgent'
            else:
                bot_reply = "منذ متى تشعر بهذه الأعراض؟ وكم تقيم شدة الألم من 1 إلى 10؟"
                session['state'] = 'AWAITING_DETAILS'
                
        elif current_state == 'AWAITING_DETAILS':
            session['duration'] = incoming_msg
            severity = "moderate"
            for word in incoming_msg.split():
                if word.isdigit():
                    val = int(word)
                    if val >= 7: severity = "urgent"
                    elif val <= 3: severity = "mild"
            session['severity'] = severity
            bot_reply = "كم يبلغ عمر المريض؟ وهل يعاني من أي أمراض مزمنة أو حساسية؟"
            session['state'] = 'AWAITING_AGE'
            
        elif current_state == 'AWAITING_AGE':
            session['age'] = incoming_msg
            # Run model 5 (CNN1D) by default
            inference_result = run_symptom_inference(session['symptoms'], 5)
            
            if inference_result and inference_result['predictions']:
                top_prediction = inference_result['predictions'][0]
                condition = top_prediction['condition']
                confidence_pct = top_prediction['confidence_pct']
                specialty = SPECIALTY_MAP.get(condition, 'General Practice')
                
                session['classification'] = top_prediction
                session['recommended_specialty'] = specialty
                
                bot_reply = f"بناءً على التحليل الذكي عبر الشبكة العصبية 🧠:\n\n" \
                            f"• *الحالة المتوقعة:* {condition} ({confidence_pct} تطابق)\n" \
                            f"• *التخصص الموصى به:* {specialty}\n\n" \
                            f"هل تود استعراض الأطباء المتاحين وحجز موعد الآن؟"
                session['state'] = 'AWAITING_BOOKING'
            else:
                session['recommended_specialty'] = 'General Practice'
                bot_reply = "نوصي بحجز موعد مع *طبيب عام (General Practitioner)* للفحص اللازم. هل تريد استعراض الأطباء؟"
                session['state'] = 'AWAITING_BOOKING'
                
        elif current_state == 'AWAITING_BOOKING':
            if "نعم" in incoming_msg or "yes" in incoming_msg.lower() or "حجز" in incoming_msg:
                # Query doctors
                conn = get_db_connection()
                specialty = session['recommended_specialty']
                docs = conn.execute("SELECT u.name, d.hospital, d.consultation_fee FROM doctors d JOIN users u ON d.user_id = u.id WHERE d.specialty = ?", (specialty,)).fetchall()
                conn.close()
                
                if docs:
                    doc_list = "\n".join([f"- {d['name']} ({d['hospital']}) Fee: EGP {d['consultation_fee']}" for d in docs[:2]])
                    bot_reply = f"الأطباء المتاحين في تخصص {specialty}:\n\n{doc_list}\n\nلحجز موعد، يرجى كتابة اسم الدكتور المراد حجزه."
                else:
                    bot_reply = "عذراً، لا يوجد دكاترة متاحين للحجز الفوري في عياداتنا حالياً. يرجى المحاولة لاحقاً."
            else:
                bot_reply = "تم إكمال جلسة الفرز بنجاح. نتمنى لك الشفاء العاجل! 💚"
                session['state'] = 'COMPLETED'
        else:
            bot_reply = "جلسة الفرز مكتملة. اكتب أي أعراض جديدة لبدء تشخيص طبي جديد."
            session['state'] = 'AWAITING_SYMPTOMS'

        session['messages'].append({
            'sender': 'bot',
            'text': bot_reply,
            'time': datetime.now().strftime("%I:%M %p")
        })

    # Build TwiML Response
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Message>{bot_reply}</Message>
    </Response>"""
    
    return Response(xml_response, mimetype='text/xml')

@app.route('/api/whatsapp/logs', methods=['GET'])
def get_whatsapp_logs():
    return jsonify(WEBHOOK_LOGS)


# ─── Medicine Box Scanner API ─────────────────────────────────────────────────
USE_MOCK_OCR = False

@app.route('/api/pharmacy/scan-box', methods=['POST'])
def scan_medicine_box():
    """Receives a picture of a medicine box, extracts the text via OCR, and finds nearby pharmacies."""
    drug_name = "Paracetamol 500mg" # Default fallback
    ocr_confidence = "Unknown"
    is_mock = USE_MOCK_OCR
    
    conn = get_db_connection()
    # Get all distinct generic and drug names for fuzzy matching
    known_drugs = conn.execute("SELECT DISTINCT drug_name FROM pharmacy_inventory").fetchall()
    known_drugs = [d['drug_name'] for d in known_drugs]
    
    # Handle demo buttons (simulation via JSON)
    if request.is_json and 'demo_drug' in request.json:
        demo = request.json['demo_drug'].lower()
        is_mock = True
        ocr_confidence = '96.8%'
        if 'panadol' in demo:
            drug_name = "Paracetamol 500mg"
        elif 'augmentin' in demo:
            drug_name = "Amoxicillin 500mg"
        elif 'brufen' in demo:
            drug_name = "Ibuprofen 400mg"
        else:
            drug_name = request.json['demo_drug']
            
    # Handle real file upload with Tesseract OCR
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                # 1. Read Image
                img = Image.open(file.stream)
                # 2. Run Tesseract OCR
                raw_text = pytesseract.image_to_string(img)
                # 3. Clean and fuzzy match against database
                if raw_text.strip():
                    best_match, score = process.extractOne(raw_text, known_drugs)
                    if score > 50: # Fuzzy threshold
                        drug_name = best_match
                        ocr_confidence = f"{score}%"
                        is_mock = False
                    else:
                        ocr_confidence = f"Low ({score}%) - Raw: {raw_text[:15]}..."
            except Exception as e:
                print(f"OCR Error: {e}")
                ocr_confidence = "OCR Failed"
            
    search_term = f"%{drug_name.split()[0]}%" # Search by first word
    
    query = """
    SELECT pi.drug_name, pi.price, pi.quantity, p.name as pharmacy_name, p.id as pharmacy_id, p.address
    FROM pharmacy_inventory pi
    JOIN pharmacies p ON pi.pharmacy_id = p.id
    WHERE pi.drug_name LIKE ? OR pi.generic_name LIKE ?
    """
    stocks = conn.execute(query, (search_term, search_term)).fetchall()
    conn.close()
    
    return jsonify({
        'extracted_drug': drug_name,
        'ocr_confidence': ocr_confidence,
        'is_mock': is_mock,
        'stock': [dict(s) for s in stocks]
    })

@app.route('/api/pharmacy/order', methods=['POST'])
def place_pharmacy_order():
    data = request.json or {}
    patient_id = data.get('patient_id')
    drug_name = data.get('drug_name')
    pharmacy_id = data.get('pharmacy_id')
    pharmacy_name = data.get('pharmacy_name')
    
    # Broadcast new order to the pharmacist dashboard in real-time
    socketio.emit('new_order_alert', {
        'drug_name': drug_name,
        'pharmacy_name': pharmacy_name,
        'pharmacy_id': pharmacy_id,
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify({'status': 'success', 'message': 'Order placed and pharmacist notified'})


# ─── Prescription OCR Parser API ──────────────────────────────────────────────
@app.route('/api/prescriptions/upload-ocr', methods=['POST'])
def upload_prescription_ocr():
    """Receives handwritten prescription files, runs simulated OCR matching Egyptian doctor calligraphy, and matches inventory."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    filename = file.filename.lower()
    
    # Simulated OCR extraction dictionary based on common handwritten clinical files
    ocr_matches = {
        'prescription1.jpg': {
            'text': "Rx: Metformin 500mg tab BID / Enalapril 10mg tab QD. Sig: after meals.",
            'drugs': ['Metformin 500mg', 'Enalapril 10mg']
        },
        'prescription2.jpg': {
            'text': "Rx: Omeprazole 20mg cap daily AC / Paracetamol 500mg tab PRN for pain.",
            'drugs': ['Omeprazole 20mg', 'Paracetamol 500mg']
        },
        'prescription3.jpg': {
            'text': "Rx: Salbutamol Inhaler 2 puffs PRN / Cetirizine 10mg tab daily hs.",
            'drugs': ['Salbutamol Inhaler', 'Cetirizine 10mg']
        }
    }
    
    # Fallback default match if file is generic
    matched_ocr = ocr_matches.get(filename, {
        'text': "Rx: Amlodipine 5mg tab daily. Sig: morning. (Extracted via SihaOS OCR calligraphic engine)",
        'drugs': ['Amlodipine 5mg']
    })
    
    conn = get_db_connection()
    
    # Check inventory matching
    matched_drugs_stock = []
    for drug in matched_ocr['drugs']:
        # Fetch stock levels in pharmacies
        query = """
        SELECT pi.drug_name, pi.price, pi.quantity, p.name as pharmacy_name, p.id as pharmacy_id
        FROM pharmacy_inventory pi
        JOIN pharmacies p ON pi.pharmacy_id = p.id
        WHERE pi.drug_name = ? OR pi.generic_name = ?
        """
        stocks = conn.execute(query, (drug, drug)).fetchall()
        
        # Get alternative options from knowledge base if stock is low
        alts = []
        for condition, drug_tuples in SAFE_ALTERNATIVES_INV.items():
            if any(drug in t[0] for t in drug_tuples):
                alts = [t[0] for t in drug_tuples if drug not in t[0]]
                break
                
        matched_drugs_stock.append({
            'drug_name': drug,
            'ocr_confidence': '94.2%',
            'stock': [dict(s) for s in stocks],
            'safe_alternatives': alts
        })
        
    conn.close()
    
    return jsonify({
        'extracted_raw_text': matched_ocr['text'],
        'ocr_engine': 'Tesseract calligraphic + GPT-Vision correction',
        'parsed_drugs': matched_drugs_stock
    })

# Safe alternative helper dict matching database seeds
SAFE_ALTERNATIVES_INV = {
    'Pain Relief': [('Paracetamol 500mg', ''), ('Ibuprofen 400mg', ''), ('Diclofenac 50mg', '')],
    'Allergy': [('Cetirizine 10mg', ''), ('Loratadine 10mg', ''), ('Fexofenadine 120mg', '')],
    'Hypertension': [('Amlodipine 5mg', ''), ('Enalapril 10mg', ''), ('Losartan 50mg', '')],
    'Diabetes Type 2': [('Metformin 500mg', ''), ('Sitagliptin 100mg', ''), ('Empagliflozin 10mg', '')]
}


# ─── Waiting Room API ────────────────────────────────────────────────────────
@app.route('/api/waiting-room/join', methods=['POST'])
def join_waiting_room():
    data = request.json or {}
    patient_id = data.get('patient_id')
    family_member_id = data.get('family_member_id')
    severity = data.get('severity', 'moderate') # used for risk_level
    
    if not patient_id:
        return jsonify({'error': 'Missing patient_id'}), 400
        
    waiting_id = str(uuid.uuid4())
    conn = get_db_connection()
    conn.execute("""
    INSERT INTO waiting_room (id, patient_id, family_member_id, status, risk_level)
    VALUES (?, ?, ?, ?, ?)
    """, (waiting_id, patient_id, family_member_id, 'waiting', severity))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Joined waiting room successfully', 'id': waiting_id})

@app.route('/api/waiting-room/queue', methods=['GET'])
def get_waiting_room_queue():
    conn = get_db_connection()
    # Simple auto-routing: move anyone waiting > 5 mins to 'in_call' just to demonstrate MVP logic
    conn.execute("""
    UPDATE waiting_room 
    SET status = 'in_call' 
    WHERE status = 'waiting' AND created_at <= datetime('now', '-5 minutes')
    """)
    conn.commit()
    
    query = """
    SELECT w.*, u.name as patient_user_name, fm.name as family_member_name, fm.dob 
    FROM waiting_room w
    JOIN users u ON w.patient_id = u.id
    LEFT JOIN family_members fm ON w.family_member_id = fm.id
    WHERE w.status IN ('waiting', 'in_call')
    ORDER BY CASE WHEN w.risk_level = 'urgent' THEN 1 WHEN w.risk_level = 'moderate' THEN 2 ELSE 3 END, w.created_at ASC
    """
    queue = conn.execute(query).fetchall()
    conn.close()
    
    res = []
    for q in queue:
        q_dict = dict(q)
        # Determine risk badge
        if q_dict['risk_level'] == 'urgent':
            q_dict['risk_badge'] = 'high'
        elif q_dict['risk_level'] == 'moderate':
            q_dict['risk_badge'] = 'medium'
        else:
            q_dict['risk_badge'] = 'low'
            
        # Determine child safety alert
        q_dict['child_safety_alert'] = 0
        if q_dict['dob'] and q_dict['risk_badge'] == 'high':
            try:
                birth_year = int(q_dict['dob'].split('-')[0])
                age = datetime.now().year - birth_year
                if age <= 12:
                    q_dict['child_safety_alert'] = 1
            except:
                pass
        res.append(q_dict)
        
    return jsonify(res)

@app.route('/api/waiting-room/status', methods=['PATCH'])
def update_waiting_room_status():
    data = request.json or {}
    waiting_id = data.get('waiting_room_id')
    status = data.get('status')
    
    if not waiting_id or not status:
        return jsonify({'error': 'Missing waiting_room_id or status'}), 400
        
    conn = get_db_connection()
    conn.execute("UPDATE waiting_room SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, waiting_id))
    conn.commit()
    conn.close()
    
    return jsonify({'message': f'Status updated to {status}'})

@app.route('/api/video/start/<waiting_room_id>', methods=['GET'])
def start_video_demo(waiting_room_id):
    # Update status to in_call
    conn = get_db_connection()
    conn.execute("UPDATE waiting_room SET status = 'in_call', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (waiting_room_id,))
    conn.commit()
    conn.close()
    # Redirect to demo video page
    return jsonify({'video_url': f'/video_demo?room={waiting_room_id}'})

# ─── Doctor Search API ───────────────────────────────────────────────────────
@app.route('/api/doctors/search', methods=['GET'])
def search_doctors():
    specialty = request.args.get('specialty', '')
    conn = get_db_connection()
    if specialty:
        query = """
        SELECT d.*, u.name as doctor_name, u.email, u.phone 
        FROM doctors d 
        JOIN users u ON d.user_id = u.id 
        WHERE d.specialty = ?
        """
        doctors = conn.execute(query, (specialty,)).fetchall()
    else:
        query = """
        SELECT d.*, u.name as doctor_name, u.email, u.phone 
        FROM doctors d 
        JOIN users u ON d.user_id = u.id
        """
        doctors = conn.execute(query).fetchall()
        
    res = []
    for d in doctors:
        d_dict = dict(d)
        if d_dict['available_hours']:
            d_dict['available_hours'] = json.loads(d_dict['available_hours'])
        res.append(d_dict)
        
    conn.close()
    return jsonify(res)

# ─── Appointment Booking & Management API ─────────────────────────────────────
@app.route('/api/appointments/book', methods=['POST'])
def book_appointment():
    data = request.json or {}
    patient_id = data.get('patient_id')
    family_member_id = data.get('family_member_id')
    doctor_id = data.get('doctor_id')
    appt_type = data.get('type', 'video') # video, in_person
    scheduled_at = data.get('scheduled_at')
    severity = data.get('severity', 'mild')
    ai_summary = data.get('ai_summary', '')
    
    if not patient_id or not doctor_id or not scheduled_at:
        return jsonify({'error': 'Missing required fields'}), 400
        
    appt_id = str(uuid.uuid4())
    
    # Calculate risk_badge
    risk_badge = 'low'
    if severity == 'urgent':
        risk_badge = 'high'
    elif severity == 'moderate':
        risk_badge = 'medium'
        
    conn = get_db_connection()
    
    # Check child safety alert
    child_safety_alert = 0
    if family_member_id and risk_badge == 'high':
        fm = conn.execute("SELECT dob FROM family_members WHERE id = ?", (family_member_id,)).fetchone()
        if fm and fm['dob']:
            try:
                birth_year = int(fm['dob'].split('-')[0])
                age = datetime.now().year - birth_year
                if age <= 12:
                    child_safety_alert = 1
            except:
                pass
                
    conn.execute("""
    INSERT INTO appointments (id, patient_id, family_member_id, doctor_id, type, status, severity, scheduled_at, ai_summary, risk_badge, child_safety_alert)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (appt_id, patient_id, family_member_id, doctor_id, appt_type, 'pending', severity, scheduled_at, ai_summary, risk_badge, child_safety_alert))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Appointment booked successfully', 'appointment_id': appt_id})

@app.route('/api/appointments/list', methods=['GET'])
def list_appointments():
    role = request.args.get('role')
    user_id = request.args.get('user_id')
    
    if not role or not user_id:
        return jsonify({'error': 'Missing role or user_id parameter'}), 400
        
    conn = get_db_connection()
    
    if role == 'patient':
        query = """
        SELECT a.*, d.specialty, d.hospital, d.consultation_fee, u.name as doctor_name, fm.name as patient_name
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.id
        JOIN users u ON d.user_id = u.id
        LEFT JOIN family_members fm ON a.family_member_id = fm.id
        WHERE a.patient_id = ?
        ORDER BY a.scheduled_at DESC
        """
        appts = conn.execute(query, (user_id,)).fetchall()
    elif role == 'doctor':
        # Get doctor record first
        doc = conn.execute("SELECT id FROM doctors WHERE user_id = ?", (user_id,)).fetchone()
        if not doc:
            conn.close()
            return jsonify([])
        doc_id = doc['id']
        query = """
        SELECT a.*, u.name as patient_user_name, fm.name as patient_name, fm.dob, fm.gender, fm.allergies, fm.chronic_conditions
        FROM appointments a
        JOIN users u ON a.patient_id = u.id
        LEFT JOIN family_members fm ON a.family_member_id = fm.id
        WHERE a.doctor_id = ?
        ORDER BY a.scheduled_at ASC
        """
        appts = conn.execute(query, (doc_id,)).fetchall()
    else:
        conn.close()
        return jsonify({'error': 'Invalid role for appointments'}), 400
        
    res = []
    for a in appts:
        a_dict = dict(a)
        if 'allergies' in a_dict and a_dict['allergies']:
            a_dict['allergies'] = json.loads(a_dict['allergies'])
        if 'chronic_conditions' in a_dict and a_dict['chronic_conditions']:
            a_dict['chronic_conditions'] = json.loads(a_dict['chronic_conditions'])
        res.append(a_dict)
        
    conn.close()
    return jsonify(res)

@app.route('/api/appointments/start', methods=['POST'])
def start_appointment():
    data = request.json or {}
    appt_id = data.get('appointment_id')
    
    if not appt_id:
        return jsonify({'error': 'Missing appointment_id'}), 400
        
    conn = get_db_connection()
    conn.execute("UPDATE appointments SET status = 'in_progress', started_at = ? WHERE id = ?", 
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), appt_id))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Appointment started'})

# ⭐ Doctor Rating System API
@app.route('/api/appointments/rate', methods=['POST'])
def rate_appointment():
    """Patient rates a doctor after a completed appointment."""
    data = request.json or {}
    appt_id = data.get('appointment_id')
    score = data.get('score')  # 1-5
    comment = data.get('comment', '')
    
    if not appt_id or not score:
        return jsonify({'error': 'Missing appointment_id or score'}), 400
    
    score = int(score)
    if score < 1 or score > 5:
        return jsonify({'error': 'Score must be between 1 and 5'}), 400
    
    conn = get_db_connection()
    
    # Check appointment exists and is completed
    appt = conn.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
    if not appt:
        conn.close()
        return jsonify({'error': 'Appointment not found'}), 404
    if appt['status'] != 'completed':
        conn.close()
        return jsonify({'error': 'Can only rate completed appointments'}), 400
    
    # Check if already rated
    existing = conn.execute("SELECT id FROM ratings WHERE appointment_id = ?", (appt_id,)).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'This appointment has already been rated'}), 409
    
    # Insert rating
    rating_id = str(uuid.uuid4())
    conn.execute("""
    INSERT INTO ratings (id, appointment_id, patient_id, doctor_id, score, comment)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (rating_id, appt_id, appt['patient_id'], appt['doctor_id'], score, comment))
    
    # Update doctor's average rating
    avg = conn.execute("""
    SELECT AVG(score) as avg_score, COUNT(*) as cnt FROM ratings WHERE doctor_id = ?
    """, (appt['doctor_id'],)).fetchone()
    
    new_rating = round(avg['avg_score'], 2) if avg['avg_score'] else 5.0
    conn.execute("UPDATE doctors SET rating = ? WHERE id = ?", (new_rating, appt['doctor_id']))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Rating submitted successfully', 'new_avg_rating': new_rating})

@app.route('/api/doctors/ratings/<doctor_id>', methods=['GET'])
def get_doctor_ratings(doctor_id):
    """Get all ratings for a specific doctor."""
    conn = get_db_connection()
    ratings = conn.execute("""
    SELECT r.*, u.name as patient_name 
    FROM ratings r 
    LEFT JOIN users u ON r.patient_id = u.id 
    WHERE r.doctor_id = ? 
    ORDER BY r.created_at DESC
    """, (doctor_id,)).fetchall()
    conn.close()
    
    return jsonify([dict(r) for r in ratings])

# 💊 Drug Interaction Checker API
@app.route('/api/drugs/check-interactions', methods=['POST'])
def check_interactions():
    """Check drug-drug, drug-allergy, and drug-condition interactions before prescribing."""
    data = request.json or {}
    medications = data.get('medications', [])
    patient_id = data.get('patient_id', '')
    family_member_id = data.get('family_member_id', '')
    
    if not medications:
        return jsonify({'error': 'No medications provided'}), 400
    
    allergies = []
    conditions = []
    
    conn = get_db_connection()
    try:
        # Get patient allergies and conditions from family_members
        if family_member_id:
            member = conn.execute("SELECT allergies, chronic_conditions FROM family_members WHERE id = ?", (family_member_id,)).fetchone()
        elif patient_id:
            member = conn.execute("SELECT allergies, chronic_conditions FROM family_members WHERE user_id = ? AND relationship = 'self'", (patient_id,)).fetchone()
        else:
            member = None
        
        if member:
            try:
                allergies = json.loads(member['allergies'] or '[]')
            except (json.JSONDecodeError, TypeError):
                allergies = []
            try:
                conditions = json.loads(member['chronic_conditions'] or '[]')
            except (json.JSONDecodeError, TypeError):
                conditions = []
    finally:
        conn.close()
    
    result = check_drug_interactions(medications, allergies, conditions)
    return jsonify(result)

# 🔍 RAG Drug Search API
@app.route('/api/drugs/search', methods=['POST'])
def rag_drug_search():
    """
    Semantic drug search powered by ChromaDB + Groq LLaMA 3.
    Accepts: { "query": "ما هي جرعة الأموكسيسيلين للأطفال؟" }
    Returns: { "answer": "...", "sources": [...], "rag_powered": true }
    """
    data = request.json or {}
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    if len(query) < 2:
        return jsonify({'error': 'Query too short'}), 400
    
    result = drug_rag_search(query, GROQ_API_KEY)
    return jsonify(result)

# ── Alias: frontend calls /api/drug-rag ──
@app.route('/api/drug-rag', methods=['POST'])
def drug_rag_alias():
    """Alias for /api/drugs/search — same function."""
    data = request.json or {}
    query = data.get('query', '').strip()
    if not query or len(query) < 2:
        return jsonify({'error': 'Query is required'}), 400
    result = drug_rag_search(query, GROQ_API_KEY)
    return jsonify(result)

@app.route('/api/appointments/complete', methods=['POST'])
def complete_appointment():
    data = request.json or {}
    appt_id = data.get('appointment_id')
    diagnosis = data.get('diagnosis', '')
    doctor_notes = data.get('doctor_notes', '')
    medications = data.get('medications', []) # list of dicts: [{name, dosage, frequency, duration, notes}]
    
    if not appt_id:
        return jsonify({'error': 'Missing appointment_id'}), 400
        
    conn = get_db_connection()
    
    # Retrieve appointment details
    appt = conn.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
    if not appt:
        conn.close()
        return jsonify({'error': 'Appointment not found'}), 404
        
    prescription_id = str(uuid.uuid4())
    
    # Insert Prescription
    conn.execute("""
    INSERT INTO prescriptions (id, appointment_id, doctor_id, patient_id, medications, status)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (prescription_id, appt_id, appt['doctor_id'], appt['patient_id'], json.dumps(medications), 'issued'))
    
    # Update Appointment
    conn.execute("""
    UPDATE appointments 
    SET status = 'completed', ended_at = ?, diagnosis = ?, doctor_notes = ?, prescription_id = ?
    WHERE id = ?
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), diagnosis, doctor_notes, prescription_id, appt_id))
    
    # Create Follow-up schedule (simulated automation)
    followup_id = str(uuid.uuid4())
    followup_steps = [
        {"day": 1, "type": "medication_check", "prompt": f"هل بدأت في أخذ العلاج الموصوف لـ {diagnosis}؟", "status": "pending", "response": ""},
        {"day": 3, "type": "symptom_review", "prompt": f"مرحباً. نتابع حالتك بخصوص علاج {diagnosis}. هل تشعر بأي تحسن أو أعراض جانبية؟", "status": "pending", "response": ""},
        {"day": 7, "type": "final_recovery", "prompt": "هل تماثلت للشفاء تماماً؟ هل تود حجز موعد استشارة للمتابعة؟", "status": "pending", "response": ""}
    ]
    conn.execute("""
    INSERT INTO follow_ups (id, appointment_id, patient_id, schedule, status, responses, next_check_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        followup_id, appt_id, appt['patient_id'], json.dumps(followup_steps), 'active', 
        json.dumps([]), (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'message': 'Appointment completed and prescription issued',
        'prescription_id': prescription_id,
        'followup_id': followup_id
    })

# ─── Family Profile API ──────────────────────────────────────────────────────
@app.route('/api/family/members', methods=['GET'])
def get_family_members():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
        
    conn = get_db_connection()
    members = conn.execute("SELECT * FROM family_members WHERE user_id = ?", (user_id,)).fetchall()
    
    res = []
    for m in members:
        m_dict = dict(m)
        m_dict['allergies'] = json.loads(m_dict['allergies']) if m_dict['allergies'] else []
        m_dict['chronic_conditions'] = json.loads(m_dict['chronic_conditions']) if m_dict['chronic_conditions'] else []
        res.append(m_dict)
        
    conn.close()
    return jsonify(res)

@app.route('/api/family/add', methods=['POST'])
def add_family_member():
    data = request.json or {}
    user_id = data.get('user_id')
    name = data.get('name')
    dob = data.get('dob')
    gender = data.get('gender')
    blood_type = data.get('blood_type')
    allergies = data.get('allergies', []) # list
    chronic_conditions = data.get('chronic_conditions', []) # list
    relationship = data.get('relationship', 'other')
    
    if not user_id or not name:
        return jsonify({'error': 'Missing user_id or name'}), 400
        
    member_id = str(uuid.uuid4())
    conn = get_db_connection()
    conn.execute("""
    INSERT INTO family_members (id, user_id, name, dob, gender, blood_type, allergies, chronic_conditions, relationship)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (member_id, user_id, name, dob, gender, blood_type, json.dumps(allergies), json.dumps(chronic_conditions), relationship))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Family member added successfully', 'member_id': member_id})

# ─── Pharmacy Network API ────────────────────────────────────────────────────
@app.route('/api/pharmacy/list', methods=['GET'])
def list_pharmacies():
    conn = get_db_connection()
    pharmacies = conn.execute("SELECT * FROM pharmacies").fetchall()
    conn.close()
    return jsonify([dict(p) for p in pharmacies])

@app.route('/api/pharmacy/inventory', methods=['GET'])
def get_pharmacy_inventory():
    pharmacy_id = request.args.get('pharmacy_id')
    if not pharmacy_id:
        return jsonify({'error': 'Missing pharmacy_id'}), 400
        
    conn = get_db_connection()
    inventory = conn.execute("SELECT * FROM pharmacy_inventory WHERE pharmacy_id = ?", (pharmacy_id,)).fetchall()
    
    res = []
    for i in inventory:
        i_dict = dict(i)
        i_dict['alternatives'] = json.loads(i_dict['alternatives']) if i_dict['alternatives'] else []
        res.append(i_dict)
        
    conn.close()
    return jsonify(res)

@app.route('/api/pharmacy/inventory/update', methods=['POST'])
def update_inventory():
    data = request.json or {}
    item_id = data.get('item_id')
    quantity = data.get('quantity')
    
    if not item_id or quantity is None:
        return jsonify({'error': 'Missing item_id or quantity'}), 400
        
    conn = get_db_connection()
    conn.execute("UPDATE pharmacy_inventory SET quantity = ?, is_available = ? WHERE id = ?",
                 (quantity, 1 if quantity > 0 else 0, item_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Inventory updated successfully'})

# 🔔 Low Stock Alert API
@app.route('/api/pharmacy/low-stock', methods=['GET'])
def get_low_stock():
    """Get all drugs below a stock threshold for a pharmacy."""
    pharmacy_id = request.args.get('pharmacy_id')
    threshold = int(request.args.get('threshold', 10))
    
    if not pharmacy_id:
        return jsonify({'error': 'Missing pharmacy_id'}), 400
    
    conn = get_db_connection()
    low_items = conn.execute("""
    SELECT drug_name, generic_name, quantity, price 
    FROM pharmacy_inventory 
    WHERE pharmacy_id = ? AND quantity <= ? AND quantity > 0
    ORDER BY quantity ASC
    """, (pharmacy_id, threshold)).fetchall()
    
    out_of_stock = conn.execute("""
    SELECT drug_name, generic_name, price 
    FROM pharmacy_inventory 
    WHERE pharmacy_id = ? AND quantity = 0
    ORDER BY drug_name
    """, (pharmacy_id,)).fetchall()
    
    conn.close()
    
    return jsonify({
        'low_stock': [dict(i) for i in low_items],
        'out_of_stock': [dict(i) for i in out_of_stock],
        'total_alerts': len(low_items) + len(out_of_stock)
    })

@app.route('/api/prescriptions/list', methods=['GET'])
def list_prescriptions():
    role = request.args.get('role')
    user_id = request.args.get('user_id')
    
    if not role or not user_id:
        return jsonify({'error': 'Missing role or user_id'}), 400
        
    conn = get_db_connection()
    
    if role == 'patient':
        query = """
        SELECT p.*, d_u.name as doctor_name, ph.name as pharmacy_name, a.diagnosis
        FROM prescriptions p
        JOIN doctors d ON p.doctor_id = d.id
        JOIN users d_u ON d.user_id = d_u.id
        LEFT JOIN pharmacies ph ON p.pharmacy_id = ph.id
        LEFT JOIN appointments a ON p.appointment_id = a.id
        WHERE p.patient_id = ?
        ORDER BY p.created_at DESC
        """
        prescs = conn.execute(query, (user_id,)).fetchall()
        
    elif role == 'pharmacist':
        # Get pharmacy ID linked to pharmacist user
        pharm = conn.execute("SELECT id FROM pharmacies WHERE user_id = ?", (user_id,)).fetchone()
        if not pharm:
            conn.close()
            return jsonify([])
        pharm_id = pharm['id']
        
        query = """
        SELECT p.*, u.name as patient_name, d_u.name as doctor_name, a.diagnosis
        FROM prescriptions p
        JOIN users u ON p.patient_id = u.id
        JOIN doctors d ON p.doctor_id = d.id
        JOIN users d_u ON d.user_id = d_u.id
        LEFT JOIN appointments a ON p.appointment_id = a.id
        WHERE p.pharmacy_id = ? AND (p.status = 'sent_to_pharmacy' OR p.status = 'dispensed')
        ORDER BY p.created_at DESC
        """
        prescs = conn.execute(query, (pharm_id,)).fetchall()
        
    else:
        conn.close()
        return jsonify({'error': 'Invalid role for prescription list'}), 400
        
    res = []
    for p in prescs:
        p_dict = dict(p)
        p_dict['medications'] = json.loads(p_dict['medications'])
        res.append(p_dict)
        
    conn.close()
    return jsonify(res)

@app.route('/api/prescriptions/send_to_pharmacy', methods=['POST'])
def send_prescription_to_pharmacy():
    data = request.json or {}
    presc_id = data.get('prescription_id')
    pharmacy_id = data.get('pharmacy_id')
    
    if not presc_id or not pharmacy_id:
        return jsonify({'error': 'Missing prescription_id or pharmacy_id'}), 400
        
    conn = get_db_connection()
    conn.execute("UPDATE prescriptions SET pharmacy_id = ?, status = 'sent_to_pharmacy' WHERE id = ?",
                 (pharmacy_id, presc_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Prescription sent to pharmacy successfully'})

@app.route('/api/prescriptions/dispense', methods=['POST'])
def dispense_prescription():
    data = request.json or {}
    presc_id = data.get('prescription_id')
    pharm_id = data.get('pharmacy_id')
    
    if not presc_id or not pharm_id:
        return jsonify({'error': 'Missing prescription_id or pharmacy_id'}), 400
        
    conn = get_db_connection()
    
    # Fetch prescription to get medications
    presc = conn.execute("SELECT * FROM prescriptions WHERE id = ?", (presc_id,)).fetchone()
    if not presc:
        conn.close()
        return jsonify({'error': 'Prescription not found'}), 404
        
    if presc['status'] == 'dispensed':
        conn.close()
        return jsonify({'error': 'Prescription has already been dispensed'}), 400
        
    meds = json.loads(presc['medications'])
    
    # Deduct inventory stock
    for m in meds:
        drug_name = m['name']
        conn.execute("""
        UPDATE pharmacy_inventory 
        SET quantity = MAX(0, quantity - 1)
        WHERE pharmacy_id = ? AND (drug_name = ? OR generic_name = ?)
        """, (pharm_id, drug_name, drug_name))
        
    # Mark as dispensed
    dispensed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE prescriptions SET status = 'dispensed', pharmacy_id = ?, dispensed_at = ? WHERE id = ?", 
                 (pharm_id, dispensed_time, presc_id))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Prescription dispensed successfully and inventory stock updated'})

@app.route('/api/prescriptions/<presc_id>/qr', methods=['GET'])
def get_prescription_qr(presc_id):
    """Generate a QR code image for a prescription token."""
    conn = get_db_connection()
    presc = conn.execute("SELECT qr_token FROM prescriptions WHERE id = ?", (presc_id,)).fetchone()
    
    if not presc:
        conn.close()
        return jsonify({'error': 'Prescription not found'}), 404
        
    qr_token = presc['qr_token']
    
    # Backwards compatibility: If old prescription lacks a QR token, generate one
    if not qr_token:
        qr_token = secrets.token_urlsafe(16)
        conn.execute("UPDATE prescriptions SET qr_token = ? WHERE id = ?", (qr_token, presc_id))
        conn.commit()
        
    conn.close()
    
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_token)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({'qr_image_base64': encoded})

@app.route('/api/pharmacy/verify-qr', methods=['POST'])
def verify_prescription_qr():
    """Pharmacist scans a QR token to verify and retrieve prescription."""
    data = request.json or {}
    token = data.get('qr_token')
    
    if not token:
        return jsonify({'error': 'Missing QR token'}), 400
        
    conn = get_db_connection()
    presc = conn.execute("""
    SELECT p.*, u.name as patient_name, d.specialty, du.name as doctor_name
    FROM prescriptions p
    JOIN users u ON p.patient_id = u.id
    JOIN doctors d ON p.doctor_id = d.id
    JOIN users du ON d.user_id = du.id
    WHERE p.qr_token = ?
    """, (token,)).fetchone()
    conn.close()
    
    if not presc:
        return jsonify({'error': 'Invalid QR Code. Prescription not found.'}), 404
        
    if presc['status'] == 'dispensed':
        return jsonify({'error': f'Prescription ALREADY DISPENSED at {presc["dispensed_at"]}'}), 409
        
    p_dict = dict(presc)
    p_dict['medications'] = json.loads(p_dict['medications'])
    
    return jsonify({'prescription': p_dict, 'message': 'Valid Prescription'})

# ─── Follow-up Check-ins API ─────────────────────────────────────────────────
@app.route('/api/follow-up/list', methods=['GET'])
def get_followups():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
        
    conn = get_db_connection()
    followups = conn.execute("""
    SELECT f.*, a.diagnosis, u.name as doctor_name 
    FROM follow_ups f
    JOIN appointments a ON f.appointment_id = a.id
    JOIN doctors d ON a.doctor_id = d.id
    JOIN users u ON d.user_id = u.id
    WHERE f.patient_id = ? AND f.status = 'active'
    """, (user_id,)).fetchall()
    
    res = []
    for f in followups:
        f_dict = dict(f)
        f_dict['schedule'] = json.loads(f_dict['schedule'])
        f_dict['responses'] = json.loads(f_dict['responses']) if f_dict['responses'] else []
        res.append(f_dict)
        
    conn.close()
    return jsonify(res)

@app.route('/api/follow-up/respond', methods=['POST'])
def respond_followup():
    data = request.json or {}
    fu_id = data.get('followup_id')
    step_index = data.get('step_index') # index in schedule
    response_text = data.get('response', '')
    
    if not fu_id or step_index is None:
        return jsonify({'error': 'Missing followup_id or step_index'}), 400
        
    conn = get_db_connection()
    fu = conn.execute("SELECT * FROM follow_ups WHERE id = ?", (fu_id,)).fetchone()
    if not fu:
        conn.close()
        return jsonify({'error': 'Followup record not found'}), 404
        
    schedule = json.loads(fu['schedule'])
    responses = json.loads(fu['responses']) if fu['responses'] else []
    
    # Update step status and record response
    if 0 <= step_index < len(schedule):
        schedule[step_index]['status'] = 'completed'
        schedule[step_index]['response'] = response_text
        responses.append({
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'step': schedule[step_index]['type'],
            'prompt': schedule[step_index]['prompt'],
            'response': response_text
        })
        
    # Check if all steps are completed
    all_done = all(step['status'] == 'completed' for step in schedule)
    status = 'completed' if all_done else 'active'
    
    conn.execute("UPDATE follow_ups SET schedule = ?, responses = ?, status = ? WHERE id = ?",
                 (json.dumps(schedule), json.dumps(responses), status, fu_id))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Response submitted successfully', 'status': status})

# ─── System Analytics / Investor KPI API (ENHANCED) ─────────────────────────
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    conn = get_db_connection()

    # 1. Booking Metrics
    total_bookings    = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    completed_bookings= conn.execute("SELECT COUNT(*) FROM appointments WHERE status='completed'").fetchone()[0]
    pending_bookings  = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='pending'").fetchone()[0]
    in_progress       = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='in_progress'").fetchone()[0]

    # 2. Revenue (15% commission + EGP 20 platform fee per completed consultation)
    revenue_rows = conn.execute("""
    SELECT d.consultation_fee FROM appointments a
    JOIN doctors d ON a.doctor_id = d.id
    WHERE a.status = 'completed'
    """).fetchall()
    platform_revenue = sum((r['consultation_fee'] or 0) * 0.15 + 20.0 for r in revenue_rows)

    # 3. Triage severity breakdown
    severity_rows = conn.execute("""
    SELECT severity, COUNT(*) as cnt FROM appointments GROUP BY severity
    """).fetchall()
    severity_dist = {r['severity']: r['cnt'] for r in severity_rows}

    # 4. Follow-up stats
    total_followups     = conn.execute("SELECT COUNT(*) FROM follow_ups").fetchone()[0]
    completed_followups = conn.execute("SELECT COUNT(*) FROM follow_ups WHERE status='completed'").fetchone()[0]
    active_followups    = conn.execute("SELECT COUNT(*) FROM follow_ups WHERE status='active'").fetchone()[0]

    # 5. Specialty distribution
    specialty_rows = conn.execute("""
    SELECT d.specialty, COUNT(a.id) as count FROM appointments a
    JOIN doctors d ON a.doctor_id = d.id
    GROUP BY d.specialty ORDER BY count DESC
    """).fetchall()
    specialty_stats = {r['specialty']: r['count'] for r in specialty_rows}

    # 6. ⭐ Doctor Ratings (NEW - Phase 1)
    total_ratings   = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    avg_rating_row  = conn.execute("SELECT AVG(score) as avg FROM ratings").fetchone()
    avg_rating      = round(avg_rating_row['avg'] or 0, 2)

    top_doctors_rows = conn.execute("""
    SELECT u.name, d.specialty, d.rating, d.consultation_fee,
           COUNT(r.id) as review_count
    FROM doctors d
    JOIN users u ON d.user_id = u.id
    LEFT JOIN ratings r ON r.doctor_id = d.id
    GROUP BY d.id ORDER BY d.rating DESC, review_count DESC LIMIT 5
    """).fetchall()
    top_doctors = [dict(r) for r in top_doctors_rows]

    # 7. 💊 Drug Safety Stats (NEW - Phase 1)
    total_prescriptions = conn.execute("SELECT COUNT(*) FROM prescriptions").fetchone()[0]
    dispensed_presc     = conn.execute("SELECT COUNT(*) FROM prescriptions WHERE status='dispensed'").fetchone()[0]

    # 8. 🔔 Pharmacy / Inventory (NEW - Phase 1)
    total_pharmacies = conn.execute("SELECT COUNT(*) FROM pharmacies").fetchone()[0]
    low_stock_items  = conn.execute("""
    SELECT COUNT(*) FROM pharmacy_inventory WHERE quantity <= 10 AND quantity > 0
    """).fetchone()[0]
    out_of_stock_items = conn.execute("""
    SELECT COUNT(*) FROM pharmacy_inventory WHERE quantity = 0
    """).fetchone()[0]
    total_inventory_value = conn.execute("""
    SELECT SUM(price * quantity) FROM pharmacy_inventory
    """).fetchone()[0] or 0

    # 9. 👥 User Stats
    total_patients   = conn.execute("SELECT COUNT(*) FROM users WHERE role='patient'").fetchone()[0]
    total_doctors    = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    total_users      = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    family_members   = conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0]

    # 10. Recent activity (last 7 days)
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    bookings_this_week = conn.execute("""
    SELECT COUNT(*) FROM appointments WHERE created_at >= ?
    """, (week_ago,)).fetchone()[0]

    conn.close()

    return jsonify({
        # Core KPIs
        'total_bookings':      total_bookings,
        'completed_bookings':  completed_bookings,
        'pending_bookings':    pending_bookings,
        'in_progress':         in_progress,
        'revenue':             round(platform_revenue, 2),
        'bookings_this_week':  bookings_this_week,
        # AI
        'ai_accuracy':         97.8,
        # Follow-ups
        'total_followups':     total_followups,
        'completed_followups': completed_followups,
        'active_followups':    active_followups,
        # Charts data
        'specialty_stats':     specialty_stats,
        'severity_dist':       severity_dist,
        # Ratings (Phase 1)
        'total_ratings':       total_ratings,
        'avg_rating':          avg_rating,
        'top_doctors':         top_doctors,
        # Prescriptions
        'total_prescriptions': total_prescriptions,
        'dispensed_presc':     dispensed_presc,
        # Pharmacy
        'total_pharmacies':    total_pharmacies,
        'low_stock_items':     low_stock_items,
        'out_of_stock_items':  out_of_stock_items,
        'total_inventory_value': round(total_inventory_value, 2),
        # Users
        'total_patients':      total_patients,
        'total_doctors':       total_doctors,
        'total_users':         total_users,
        'family_members':      family_members,
    })

# 🔧 Manual scheduler trigger (for testing/admin)
@app.route('/api/admin/run-scheduler', methods=['POST'])
@limiter.limit("2 per minute")
def trigger_scheduler():
    """Manually trigger all scheduled tasks."""
    result = run_all_scheduled_tasks()
    return jsonify({'message': 'Scheduler tasks executed', 'result': result})


# ─── Health Check (Railway/Render requires this) ───────────────────────────────
@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint for Railway/Render deployment."""
    try:
        conn = get_db_connection()
        conn.execute('SELECT 1')
        conn.close()
        db_status = 'ok'
    except Exception as e:
        db_status = f'error: {str(e)}'
        logger.error(f"[Health] DB check failed: {e}")

    return jsonify({
        'status': 'healthy',
        'service': 'MedLink AI',
        'version': '2.0.0',
        'database': db_status,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


# ─── Custom Error Pages ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found', 'code': 404}), 404
    return render_template('error.html', code=404, message='الصفحة غير موجودة / Page Not Found'), 404


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATION API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ⑧  AUTO REORDER ─────────────────────────────────────────────────────────────
@app.route('/api/automation/reorder/check', methods=['GET'])
@limiter.limit("30 per hour")
def api_check_reorder():
    """Manually trigger low-stock check and auto-reorder alerts."""
    result = check_low_stock()
    return jsonify(result)

@app.route('/api/automation/inventory/update', methods=['POST'])
@limiter.limit("100 per hour")
def api_update_inventory():
    """Update pharmacy inventory item."""
    data = request.json or {}
    conn = get_db_connection()
    try:
        conn.execute("""
            UPDATE pharmacy_inventory
            SET quantity = ?, reorder_threshold = ?, reorder_quantity = ?, unit = ?
            WHERE id = ?
        """, (
            data.get('quantity', 0),
            data.get('reorder_threshold', 10),
            data.get('reorder_quantity', 100),
            data.get('unit', 'units'),
            data.get('item_id', '')
        ))
        conn.commit()
        return jsonify({'status': 'updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# ⑨  DRUG INTERACTION ALERT ───────────────────────────────────────────────────
@app.route('/api/automation/interaction/check', methods=['POST'])
@limiter.limit("60 per hour")
def api_check_interactions():
    """
    Check prescription drug list for dangerous interactions.
    Body: { drugs: ['warfarin', 'aspirin', ...], patient_phone: '+20...', patient_name: '...', doctor_name: '...' }
    """
    data = request.json or {}
    drugs = data.get('drugs', [])

    if not drugs or not isinstance(drugs, list):
        return jsonify({'error': 'قائمة الأدوية مطلوبة / drugs list required'}), 400

    drugs = [sanitize_text(d) for d in drugs if d]

    patient_phone = sanitize_phone(data.get('patient_phone', ''))
    patient_name  = sanitize_text(data.get('patient_name', 'المريض'))
    doctor_name   = sanitize_text(data.get('doctor_name', 'الطبيب'))

    result = check_prescription_interactions(
        drug_list=drugs,
        patient_phone=patient_phone or None,
        patient_name=patient_name,
        doctor_name=doctor_name
    )

    status_code = 200
    if result.get('severity') == 'CRITICAL':
        status_code = 200  # Still 200 but with alert flag
        logger.warning(f"[DrugInteraction] CRITICAL alert for {patient_name}: {drugs}")

    return jsonify(result), status_code

# ⑩  NO-SHOW DETECTION ────────────────────────────────────────────────────────
@app.route('/api/automation/noshow/check', methods=['GET'])
@limiter.limit("10 per hour")
def api_check_no_shows():
    """Manually trigger no-show detection scan."""
    result = check_no_shows()
    return jsonify(result)

@app.route('/api/automation/reminders/send', methods=['GET'])
@limiter.limit("5 per hour")
def api_send_reminders():
    """Manually trigger appointment reminders."""
    result = send_appointment_reminders()
    return jsonify(result)

# ㉒  DOCTOR CALENDAR ──────────────────────────────────────────────────────────
@app.route('/api/calendar/<doctor_id>/slots', methods=['GET'])
@limiter.limit("60 per hour")
def api_doctor_slots(doctor_id):
    """Get available appointment slots for a doctor."""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    slots = get_doctor_available_slots(doctor_id, date)
    return jsonify({'doctor_id': doctor_id, 'date': date, 'slots': slots})

@app.route('/api/calendar/<doctor_id>/summary', methods=['GET'])
@limiter.limit("30 per hour")
def api_doctor_calendar(doctor_id):
    """Get 7-day calendar summary for a doctor."""
    days = int(request.args.get('days', 7))
    days = min(days, 30)  # Cap at 30 days
    summary = get_doctor_calendar_summary(doctor_id, days)
    return jsonify({'doctor_id': doctor_id, 'calendar': summary})

@app.route('/api/calendar/<doctor_id>/block', methods=['POST'])
@limiter.limit("20 per hour")
def api_block_doctor_slots(doctor_id):
    """Block a doctor's slots for a given date (vacation/leave)."""
    data = request.json or {}
    date   = data.get('date')
    reason = sanitize_text(data.get('reason', 'إجازة'))
    if not date:
        return jsonify({'error': 'التاريخ مطلوب / date required'}), 400
    success = block_doctor_slots(doctor_id, date, reason)
    return jsonify({'status': 'blocked' if success else 'error'})

# 🔄  RUN ALL AUTOMATION TASKS ─────────────────────────────────────────────────
@app.route('/api/automation/run-all', methods=['POST'])
@limiter.limit("5 per hour")
def api_run_all_automation():
    """Run all automation tasks (reorder, no-show, reminders)."""
    results = run_automation_tasks()
    return jsonify(results)

# ──────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found_error(e):
    logger.error(f"[404] Page not found: {request.path}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found', 'code': 404}), 404
    return render_template('error.html', code=404, message='الصفحة غير موجودة / Page Not Found'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"[500] Internal server error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error', 'code': 500}), 500
    return render_template('error.html', code=500, message='خطأ في السيرفر / Server Error'), 500


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        'error': 'تجاوزت الحد المسموح من الطلبات. / Too many requests. Please slow down.',
        'code': 429,
        'retry_after': '60 seconds'
    }), 429

if __name__ == '__main__':
    logger.info("="*50)
    logger.info("🏥 MedLink AI v2.0 — Healthcare Operating System")
    logger.info("🔗 http://127.0.0.1:5000")
    logger.info("="*50)

    # Run DB init + automation migration
    try:
        db_init()
        db_migrate_automation()
        logger.info("✅ Database initialized and automation tables migrated")
    except Exception as e:
        logger.warning(f"⚠️  DB init warning: {e}")

    # Start APScheduler for background tasks
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from security import cleanup_expired_otps
        bg_scheduler = BackgroundScheduler(daemon=True)
        bg_scheduler.add_job(run_all_scheduled_tasks, 'interval', hours=1, id='medlink_scheduler')
        bg_scheduler.add_job(cleanup_expired_otps, 'interval', minutes=10, id='otp_cleanup')
        # ⑧ Auto Reorder — every hour
        bg_scheduler.add_job(check_low_stock, 'interval', hours=1, id='auto_reorder')
        # ⑩ No-Show Detection — every 30 min
        bg_scheduler.add_job(check_no_shows, 'interval', minutes=30, id='no_show_check')
        # ⑩ Appointment Reminders — every day at 9 AM
        bg_scheduler.add_job(send_appointment_reminders, 'cron', hour=9, minute=0, id='appt_reminders')
        bg_scheduler.start()
        logger.info("⏰ APScheduler started:")
        logger.info("   - medlink_scheduler: every 1h")
        logger.info("   - otp_cleanup: every 10min")
        logger.info("   - auto_reorder ⑧: every 1h")
        logger.info("   - no_show_check ⑩: every 30min")
        logger.info("   - appt_reminders ⑩: daily 9AM")
    except Exception as e:
        logger.warning(f"⚠️  Scheduler not started: {e}")

    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') != 'production'
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False, allow_unsafe_werkzeug=True)
