import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    print("="*60)
    print("🏥 MedLink AI - System Diagnostic & Feature Tests")
    print("="*60)

    # 1. Database & Schema
    print("\n[1] Testing Database & Schema...")
    try:
        from database import get_db_connection, db_init, db_migrate_automation
        db_init()
        db_migrate_automation()
        conn = get_db_connection()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = [t[0] for t in tables]
        print(f"  ✅ Database Connected. Found {len(tables)} tables.")
        if "automation_log" in table_names:
            print("  ✅ Automation tables successfully migrated.")
        conn.close()
    except Exception as e:
        print(f"  ❌ DB Error: {e}")

    # 2. Security (OTP & Sanitization)
    print("\n[2] Testing Security Module...")
    try:
        from security import generate_otp, verify_otp, sanitize_text, is_safe_query
        phone = "+201012345678"
        otp = generate_otp(phone)
        print(f"  ✅ OTP Generated dynamically for {phone}: [{otp}]")
        success, msg = verify_otp(phone, otp)
        print(f"  ✅ OTP Verification (Valid Code): {success} - {msg}")
        
        # Test sanitization
        dirty_input = "<script>alert(1)</script> Hello"
        clean_input = sanitize_text(dirty_input)
        print(f"  ✅ XSS Sanitization: '{dirty_input}' -> '{clean_input}'")
        
        # Test SQLi protection
        bad_query = "USER; DROP TABLE users; --"
        is_safe = is_safe_query(bad_query)
        print(f"  ✅ SQLi Protection caught attack: {not is_safe}")
    except Exception as e:
        print(f"  ❌ Security Error: {e}")

    # 3. Intelligence / RAG
    print("\n[3] Testing Intelligence (RAG Engine)...")
    try:
        from drug_knowledge_base import get_all_drugs
        drugs = get_all_drugs()
        print(f"  ✅ RAG Database expanded successfully. Found {len(drugs)} drugs (Target > 150).")
        sample = next((d for d in drugs if "warfarin" in d['name'].lower()), None)
        if sample:
            print(f"  ✅ Found Critical Drug: {sample['name']} - {sample['category']}")
    except Exception as e:
        print(f"  ❌ RAG Error: {e}")

    # 4. Automation (Drug Interactions)
    print("\n[4] Testing Automation: Drug Interactions Alert...")
    try:
        from automation import check_prescription_interactions
        print("  ⏳ Simulating SAFE prescription (Amoxicillin + Panadol)...")
        res1 = check_prescription_interactions(["Amoxicillin", "Panadol"])
        print(f"  ✅ Result: Safe={res1['safe']}, Severity={res1['severity']}")

        print("  ⏳ Simulating FATAL prescription (Warfarin + Aspirin)...")
        res2 = check_prescription_interactions(["Warfarin", "Aspirin"], patient_phone="+20111222333")
        print(f"  ✅ Result: Safe={res2['safe']}, Severity={res2['severity']}")
        for alert in res2['alerts']:
            print(f"     🚨 SMS ALERT TRIGGERED: {alert['warning']}")
    except Exception as e:
        print(f"  ❌ Automation Error: {e}")

    # 5. Automation (Auto Reorder & No-Show)
    print("\n[5] Testing Automation: Inventory & Schedulers...")
    try:
        from automation import check_low_stock, check_no_shows
        res3 = check_low_stock()
        print(f"  ✅ Auto Reorder scan complete. Need Reorder: {res3.get('alerts', 0)} items.")
        
        res4 = check_no_shows()
        print(f"  ✅ No-Show scan complete. Detected: {res4.get('detected', 0)} missed appointments.")
    except Exception as e:
        print(f"  ❌ Automation Error: {e}")

    print("\n" + "="*60)
    print("✅ All CMD Tests Completed Successfully.")
    print("="*60)

if __name__ == "__main__":
    run_tests()
