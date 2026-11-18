import sqlite3
import os
import json
import uuid
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medlink_ai.db')
DATABASE_URL = os.getenv('DATABASE_URL')  # Use for PostgreSQL in production

def get_db_connection():
    if DATABASE_URL:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        conn.row_factory = psycopg2.extras.DictCursor
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def db_init():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable foreign keys (SQLite only)
    if not DATABASE_URL:
        cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        phone TEXT UNIQUE NOT NULL,
        email TEXT,
        name TEXT NOT NULL,
        role TEXT CHECK(role IN ('patient', 'doctor', 'pharmacist', 'admin')) NOT NULL,
        is_verified INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Doctors Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id TEXT PRIMARY KEY,
        user_id TEXT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
        specialty TEXT NOT NULL,
        sub_specialty TEXT,
        license_number TEXT UNIQUE NOT NULL,
        hospital TEXT,
        years_experience INTEGER,
        consultation_fee REAL,
        rating REAL DEFAULT 5.0,
        session_status TEXT CHECK(session_status IN ('Available', 'Busy', 'In Consultation', 'Offline')) DEFAULT 'Offline',
        is_on_call INTEGER DEFAULT 0,
        bio TEXT,
        available_hours TEXT -- JSON format
    );
    """)

    # 3. Family Members Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS family_members (
        id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        dob TEXT,
        gender TEXT CHECK(gender IN ('male', 'female', 'other')),
        blood_type TEXT,
        allergies TEXT, -- JSON format list of strings
        chronic_conditions TEXT, -- JSON format list of strings
        relationship TEXT CHECK(relationship IN ('self', 'child', 'spouse', 'parent', 'sibling', 'other'))
    );
    """)

    # 4. Pharmacies Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pharmacies (
        id TEXT PRIMARY KEY,
        user_id TEXT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        address TEXT,
        phone TEXT,
        is_24h INTEGER DEFAULT 0,
        delivery_available INTEGER DEFAULT 0
    );
    """)

    # 5. Pharmacy Inventory Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pharmacy_inventory (
        id TEXT PRIMARY KEY,
        pharmacy_id TEXT REFERENCES pharmacies(id) ON DELETE CASCADE,
        drug_name TEXT NOT NULL,
        generic_name TEXT,
        price REAL,
        quantity INTEGER DEFAULT 0,
        is_available INTEGER DEFAULT 1,
        alternatives TEXT -- JSON format list of strings
    );
    """)

    # 6. Appointments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id TEXT PRIMARY KEY,
        patient_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        family_member_id TEXT REFERENCES family_members(id) ON DELETE SET NULL,
        doctor_id TEXT REFERENCES doctors(id) ON DELETE SET NULL,
        type TEXT CHECK(type IN ('in_person', 'video', 'on_call')) NOT NULL,
        status TEXT CHECK(status IN ('pending', 'confirmed', 'in_progress', 'completed', 'cancelled')) NOT NULL,
        severity TEXT CHECK(severity IN ('mild', 'moderate', 'urgent')) NOT NULL,
        scheduled_at TEXT,
        started_at TEXT,
        ended_at TEXT,
        ai_summary TEXT,
        ai_confidence TEXT CHECK(ai_confidence IN ('High', 'Medium', 'Low')) DEFAULT 'Medium',
        risk_badge TEXT,
        child_safety_alert INTEGER DEFAULT 0,
        timeline JSON,
        doctor_notes TEXT,
        diagnosis TEXT,
        prescription_id TEXT, -- REFERENCES prescriptions(id) but can be set later
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 7. Prescriptions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prescriptions (
        id TEXT PRIMARY KEY,
        appointment_id TEXT REFERENCES appointments(id) ON DELETE SET NULL,
        doctor_id TEXT REFERENCES doctors(id) ON DELETE SET NULL,
        patient_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        medications TEXT NOT NULL, -- JSON list: [{"name": "...", "dosage": "...", "frequency": "...", "duration": "...", "notes": "..."}]
        status TEXT CHECK(status IN ('issued', 'sent_to_pharmacy', 'dispensed')) DEFAULT 'issued',
        pharmacy_id TEXT REFERENCES pharmacies(id) ON DELETE SET NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 8. Follow-ups Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS follow_ups (
        id TEXT PRIMARY KEY,
        appointment_id TEXT REFERENCES appointments(id) ON DELETE CASCADE,
        patient_id TEXT REFERENCES users(id) ON DELETE CASCADE,
        schedule TEXT, -- JSON structure of follow-up steps
        status TEXT CHECK(status IN ('active', 'completed', 'missed')) DEFAULT 'active',
        responses TEXT, -- JSON structure of patient check-in responses
        next_check_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 9. Waiting Room Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS waiting_room (
        id TEXT PRIMARY KEY,
        patient_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        family_member_id TEXT REFERENCES family_members(id) ON DELETE SET NULL,
        status TEXT CHECK(status IN ('waiting','in_call','completed')) DEFAULT 'waiting',
        risk_level TEXT CHECK(risk_level IN ('urgent','moderate','mild')) NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 9. Audit Logs Table (For Security & Compliance)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        action TEXT NOT NULL,
        details TEXT, -- JSON string describing the action
        ip_address TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 10. Doctor Ratings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        id TEXT PRIMARY KEY,
        appointment_id TEXT UNIQUE REFERENCES appointments(id) ON DELETE CASCADE,
        patient_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        doctor_id TEXT REFERENCES doctors(id) ON DELETE SET NULL,
        score INTEGER CHECK(score BETWEEN 1 AND 5) NOT NULL,
        comment TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


def db_migrate_automation():
    """Add automation-related columns and tables to existing DB."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── pharmacy_inventory: add reorder columns ───────────────────────────────
    for col, definition in [
        ("reorder_threshold", "INTEGER DEFAULT 10"),
        ("reorder_quantity",  "INTEGER DEFAULT 100"),
        ("unit",              "TEXT DEFAULT 'units'"),
        ("last_reorder_at",   "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE pharmacy_inventory ADD COLUMN {col} {definition}")
        except Exception:
            pass  # Column already exists

    # ── users: doctor scheduling columns ─────────────────────────────────────
    for col, definition in [
        ("working_hours_start",    "TEXT DEFAULT '09:00'"),
        ("working_hours_end",      "TEXT DEFAULT '17:00'"),
        ("appointment_duration_min","INTEGER DEFAULT 30"),
        ("max_daily_appointments", "INTEGER DEFAULT 16"),
        ("phone",                  "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
        except Exception:
            pass

    # ── appointments: add date/time/reminder columns ──────────────────────────
    for col, definition in [
        ("appointment_date",  "TEXT"),
        ("appointment_time",  "TEXT"),
        ("reminder_sent",     "INTEGER DEFAULT 0"),
        ("reason",            "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE appointments ADD COLUMN {col} {definition}")
        except Exception:
            pass

    # ── automation_log table ──────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS automation_log (
        id          TEXT PRIMARY KEY,
        event_type  TEXT NOT NULL,
        entity_id   TEXT,
        message     TEXT,
        status      TEXT DEFAULT 'pending',
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # ── doctor_unavailable table ──────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctor_unavailable (
        id          TEXT PRIMARY KEY,
        doctor_id   TEXT NOT NULL,
        blocked_date TEXT NOT NULL,
        reason      TEXT DEFAULT 'Unavailable',
        created_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("Automation migration complete.")



def db_seed():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if seeded
    cursor.execute("SELECT COUNT(*) FROM users;")
    if cursor.fetchone()[0] > 0:
        conn.close()
        print("Database already contains data. Seeding skipped.")
        return

    print("Seeding database with demo startup data...")

    # Define User IDs
    p_id = str(uuid.uuid4())  # Patient Mohamed Ali
    a_id = str(uuid.uuid4())  # Admin User
    pharm_ids = [str(uuid.uuid4()) for _ in range(3)] # Pharmacists
    doc_ids = [str(uuid.uuid4()) for _ in range(10)] # Doctors

    # 1. Insert Users
    users_data = [
        (p_id, "+201011111111", "mohamed.ali@email.com", "Mohamed Ali", "patient", 1),
        (a_id, "+201022222222", "admin@sihaos.com", "SihaOS Admin", "admin", 1),
        (pharm_ids[0], "+201033333333", "ezaby.zam@sihaos.com", "Pharmacist Ezaby Zamalek", "pharmacist", 1),
        (pharm_ids[1], "+201044444444", "seif.helio@sihaos.com", "Pharmacist Seif Heliopolis", "pharmacist", 1),
        (pharm_ids[2], "+201055555555", "care.maadi@sihaos.com", "Pharmacist Care Maadi", "pharmacist", 1),
    ]

    # Doctors Users mapping
    doc_specialties = [
        ("Pain Relief", "Internal Medicine", "Dr. Ahmed Mansour", 8, "Nasr City Hospital", 15, 200.0, 4.9),
        ("Cold & Flu", "General Practice", "Dr. Sarah Hegazi", 12, "Heliopolis Medical Clinic", 6, 100.0, 4.7),
        ("Allergy", "Allergy & Immunology", "Dr. Khaled Selim", 10, "Zamalek Allergy Center", 11, 250.0, 4.85),
        ("Hypertension", "Cardiology", "Dr. Laila Kamel", 14, "Cairo Heart Institute", 18, 350.0, 4.95),
        ("Diabetes Type 2", "Endocrinology", "Dr. Youssef Nagi", 7, "Maadi Diabetes Clinic", 9, 220.0, 4.8),
        ("Infection", "Infectious Diseases", "Dr. Mona Farid", 9, "Giza International Hospital", 10, 180.0, 4.75),
        ("Gastroesophageal Reflux", "Gastroenterology", "Dr. Tarek Amin", 11, "Cairo Gastro Clinic", 13, 260.0, 4.9),
        ("Asthma", "Pulmonology", "Dr. Sherif Gad", 6, "Delta Chest Center", 8, 150.0, 4.6),
        ("Depression", "Psychiatry", "Dr. Amr Fawzy", 13, "Heliopolis Psychiatry Clinic", 14, 300.0, 4.92),
        ("Arthritis", "Rheumatology", "Dr. Nadia Osman", 15, "Cairo Joint Center", 16, 280.0, 4.88)
    ]

    for i, item in enumerate(doc_specialties):
        users_data.append((doc_ids[i], f"+2010666666{i:02d}", f"doc{i}@sihaos.com", item[2], "doctor", 1))

    cursor.executemany("INSERT INTO users (id, phone, email, name, role, is_verified) VALUES (?, ?, ?, ?, ?, ?)", users_data)

    # 2. Insert Doctors Detail
    doctors_data = []
    for i, item in enumerate(doc_specialties):
        doc_pk = str(uuid.uuid4())
        hours = {
            "sun": ["10:00-14:00", "18:00-21:00"],
            "mon": ["10:00-14:00", "18:00-21:00"],
            "tue": ["10:00-14:00", "18:00-21:00"],
            "wed": ["10:00-14:00", "18:00-21:00"],
            "thu": ["10:00-14:00", "16:00-19:00"]
        }
        doctors_data.append((
            doc_pk, doc_ids[i], item[0], item[1], f"LIC-{10000+i}", item[4], item[5], item[6], item[7], 'Available' if i%2==0 else 'Offline', 1 if i%3==0 else 0,
            f"Specialist in {item[0]} and {item[1]} with over {item[5]} years of clinical experience in Egypt.",
            json.dumps(hours)
        ))
    cursor.executemany("""
    INSERT INTO doctors (id, user_id, specialty, sub_specialty, license_number, hospital, years_experience, consultation_fee, rating, session_status, is_on_call, bio, available_hours)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, doctors_data)

    # 3. Insert Family Members (For Mohamed Ali)
    member_self = (str(uuid.uuid4()), p_id, "Mohamed Ali", "1990-05-15", "male", "A+", json.dumps(["Penicillin"]), json.dumps([]), "self")
    member_wife = (str(uuid.uuid4()), p_id, "Fatma Hassan", "1994-09-20", "female", "O-", json.dumps([]), json.dumps(["Asthma"]), "spouse")
    member_son = (str(uuid.uuid4()), p_id, "Ali Mohamed", "2018-03-12", "male", "A+", json.dumps(["Nuts"]), json.dumps([]), "child")
    cursor.executemany("""
    INSERT INTO family_members (id, user_id, name, dob, gender, blood_type, allergies, chronic_conditions, relationship)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [member_self, member_wife, member_son])

    # 4. Insert Pharmacies
    pharmacies_data = [
        (str(uuid.uuid4()), pharm_ids[0], "El-Ezaby Pharmacy", "15 El-Mansour St, Zamalek, Cairo", "+20227361111", 1, 1),
        (str(uuid.uuid4()), pharm_ids[1], "Seif Pharmacy", "45 El-Mirghani St, Heliopolis, Cairo", "+20224182222", 1, 1),
        (str(uuid.uuid4()), pharm_ids[2], "Care Pharmacy", "89 El-Nasr Ave, Maadi, Cairo", "+20225193333", 0, 1),
    ]
    cursor.executemany("""
    INSERT INTO pharmacies (id, user_id, name, address, phone, is_24h, delivery_available)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, pharmacies_data)

    # 5. Insert Pharmacy Inventories (drugs matching safe alternatives)
    # Get pharmacies
    cursor.execute("SELECT id, name FROM pharmacies;")
    pharms = cursor.fetchall()
    
    # SAFE_ALTERNATIVES definitions matching inference.py
    SAFE_ALTERNATIVES = {
        'Pain Relief': [
            ('Paracetamol 500mg', 'First-line; safest for liver', 15.50, 'Paracetamol'),
            ('Ibuprofen 400mg',   'Better anti-inflammatory', 22.00, 'Ibuprofen'),
            ('Diclofenac 50mg',   'Strong NSAID', 30.00, 'Diclofenac'),
        ],
        'Cold & Flu': [
            ('Paracetamol + Vit C', 'Symptomatic relief', 18.00, 'Paracetamol + Vit C'),
            ('Cetirizine 10mg',     'For allergic component', 25.00, 'Cetirizine'),
        ],
        'Allergy': [
            ('Cetirizine 10mg',   '2nd-gen antihistamine', 25.00, 'Cetirizine'),
            ('Loratadine 10mg',   'Non-drowsy', 28.50, 'Loratadine'),
            ('Fexofenadine 120mg','Least sedating', 45.00, 'Fexofenadine'),
        ],
        'Hypertension': [
            ('Amlodipine 5mg',    'CCB; well-tolerated', 38.00, 'Amlodipine'),
            ('Enalapril 10mg',    'ACE inhibitor', 29.00, 'Enalapril'),
            ('Losartan 50mg',     'ARB; use when cough occurs', 55.00, 'Losartan'),
        ],
        'Diabetes Type 2': [
            ('Metformin 500mg',   'First-line', 19.50, 'Metformin'),
            ('Sitagliptin 100mg', 'DPP-4 inhibitor', 120.00, 'Sitagliptin'),
            ('Empagliflozin 10mg','SGLT-2; renal protection', 150.00, 'Empagliflozin'),
        ],
        'Infection': [
            ('Amoxicillin 500mg',     'Broad-spectrum', 32.50, 'Amoxicillin'),
            ('Clarithromycin 500mg',  'For penicillin allergy', 85.00, 'Clarithromycin'),
            ('Ciprofloxacin 500mg',   'Urinary tract infections', 40.00, 'Ciprofloxacin'),
        ],
        'Gastroesophageal Reflux': [
            ('Omeprazole 20mg',     'PPI; acid suppression', 42.00, 'Omeprazole'),
            ('Pantoprazole 40mg',   'Longer duration PPI', 60.00, 'Pantoprazole'),
        ],
        'Asthma': [
            ('Salbutamol Inhaler',    'Short-acting rescue inhaler', 35.00, 'Salbutamol'),
            ('Fluticasone Inhaler',   'ICS controller (daily)', 95.00, 'Fluticasone'),
            ('Montelukast 10mg',      'Leukotriene antagonist', 70.00, 'Montelukast'),
        ],
        'Depression': [
            ('Sertraline 50mg',      'SSRI; first-line', 80.00, 'Sertraline'),
            ('Escitalopram 10mg',    'SSRI; clean profile', 90.00, 'Escitalopram'),
            ('Fluoxetine 20mg',      'Long half-life SSRI', 65.00, 'Fluoxetine'),
        ],
        'Arthritis': [
            ('Diclofenac 50mg',  'NSAID for flares', 30.00, 'Diclofenac'),
            ('Celecoxib 200mg',  'COX-2 selective', 75.00, 'Celecoxib'),
            ('Methotrexate 7.5mg','DMARD for RA (specialist)', 50.00, 'Methotrexate'),
        ],
    }

    inventory_data = []
    for ph in pharms:
        for condition, drugs in SAFE_ALTERNATIVES.items():
            for drug_tuple in drugs:
                drug_name = drug_tuple[0]
                generic = drug_tuple[3]
                price = drug_tuple[2]
                
                # Create alternatives list (excluding current drug)
                alts = [d[0] for d in drugs if d[0] != drug_name]
                
                inventory_data.append((
                    str(uuid.uuid4()), ph['id'], drug_name, generic, price,
                    100 if ph['name'] == "El-Ezaby Pharmacy" else (50 if ph['name'] == "Seif Pharmacy" else 20), # stock qty
                    1, json.dumps(alts)
                ))
    cursor.executemany("""
    INSERT INTO pharmacy_inventory (id, pharmacy_id, drug_name, generic_name, price, quantity, is_available, alternatives)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, inventory_data)

    # 6. Insert some sample past Appointments & Prescriptions for the patient
    # Get doctor list
    cursor.execute("SELECT id, specialty, user_id FROM doctors;")
    docs = cursor.fetchall()
    
    # Appointment with Dr. Sarah Hegazi (Cold & Flu) completed
    doc_cold = [d for d in docs if d['specialty'] == 'Cold & Flu'][0]
    app_id = str(uuid.uuid4())
    pres_id = str(uuid.uuid4())
    
    past_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO appointments (id, patient_id, family_member_id, doctor_id, type, status, severity, scheduled_at, started_at, ended_at, ai_summary, doctor_notes, diagnosis, prescription_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        app_id, p_id, member_self[0], doc_cold['id'], 'video', 'completed', 'mild', past_date, past_date, past_date,
        "Patient complains of cough, fever, runny nose and body ache for 2 days. Rule out pneumonia. Advised rest.",
        "Mild common cold. Patient advised to stay hydrated, rest, and check in if fever rises.",
        "Common Cold", pres_id, past_date
    ))
    
    # Medication list
    meds = [
        {"name": "Paracetamol + Vit C", "dosage": "1 tablet", "frequency": "3 times daily", "duration": "5 days", "notes": "After meals"},
        {"name": "Cetirizine 10mg", "dosage": "1 tablet", "frequency": "Once daily at night", "duration": "5 days", "notes": "May cause light drowsiness"}
    ]
    cursor.execute("""
    INSERT INTO prescriptions (id, appointment_id, doctor_id, patient_id, medications, status, pharmacy_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pres_id, app_id, doc_cold['id'], p_id, json.dumps(meds), 'dispensed', pharms[0]['id'], past_date
    ))

    # Follow-up for this appointment
    fu_id = str(uuid.uuid4())
    fu_schedule = [
        {"day": 1, "type": "medication_check", "prompt": "Did you start taking Paracetamol and Cetirizine?", "status": "completed", "response": "Yes, I started today."},
        {"day": 3, "type": "symptom_review", "prompt": "Are you feeling better? Is the fever gone?", "status": "completed", "response": "Yes, the fever is completely gone. Still have a mild cough."},
        {"day": 5, "type": "final_recovery", "prompt": "Are you fully recovered?", "status": "pending", "response": ""}
    ]
    cursor.execute("""
    INSERT INTO follow_ups (id, appointment_id, patient_id, schedule, status, responses, next_check_at, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fu_id, app_id, p_id, json.dumps(fu_schedule), 'active', json.dumps([]),
        (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), past_date
    ))

    conn.commit()
    conn.close()
    print("Database seeded successfully with premium startup data.")

if __name__ == '__main__':
    db_init()
    db_seed()
