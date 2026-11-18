"""
MedLink AI — Automation Engine
Handles:
  ⑧  Auto Reorder         — low stock → alert to distributor
  ⑨  Drug Interaction Alert — dangerous combo → SMS to patient
  ⑩  No-Show Detection    — missed appointment → reminder
  ㉒  Doctor Calendar Sync — check upcoming slots
"""

import os
import logging
from datetime import datetime, timedelta
from database import get_db_connection

logger = logging.getLogger('medlink.automation')


# ══════════════════════════════════════════════════════════════════════════════
# SHARED: SMS / WhatsApp sender (Twilio if configured, else console)
# ══════════════════════════════════════════════════════════════════════════════

def _send_sms(to_phone: str, message: str, category: str = "AUTO") -> bool:
    """Send SMS via Twilio or log to console in dev mode."""
    twilio_sid   = os.environ.get('TWILIO_ACCOUNT_SID', '')
    twilio_token = os.environ.get('TWILIO_AUTH_TOKEN', '')
    twilio_from  = os.environ.get('TWILIO_PHONE_NUMBER', '')

    if twilio_sid and twilio_token and twilio_from:
        try:
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_token)
            msg = client.messages.create(body=message, from_=twilio_from, to=to_phone)
            logger.info(f"[{category}] SMS sent → {to_phone[-4:]}xxxx | SID: {msg.sid}")
            return True
        except Exception as e:
            logger.error(f"[{category}] Twilio error: {e}")

    # Dev fallback — log to console
    print(f"\n📱 [{category}] SMS to {to_phone}:\n{message}\n")
    logger.info(f"[{category}] (DEV) SMS logged for {to_phone[-4:]}xxxx")
    return True


def _send_whatsapp(to_phone: str, message: str, category: str = "AUTO") -> bool:
    """Send WhatsApp via Twilio Sandbox or log to console."""
    twilio_sid   = os.environ.get('TWILIO_ACCOUNT_SID', '')
    twilio_token = os.environ.get('TWILIO_AUTH_TOKEN', '')
    twilio_from  = os.environ.get('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')

    if twilio_sid and twilio_token:
        try:
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_token)
            wa_to = f"whatsapp:{to_phone}" if not to_phone.startswith('whatsapp:') else to_phone
            msg = client.messages.create(body=message, from_=twilio_from, to=wa_to)
            logger.info(f"[{category}] WhatsApp sent → {to_phone[-4:]}xxxx | SID: {msg.sid}")
            return True
        except Exception as e:
            logger.error(f"[{category}] WhatsApp error: {e}")

    print(f"\n💬 [{category}] WhatsApp to {to_phone}:\n{message}\n")
    logger.info(f"[{category}] (DEV) WhatsApp logged for {to_phone[-4:]}xxxx")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ⑧  AUTO REORDER — Low Stock Detection
# ══════════════════════════════════════════════════════════════════════════════

# Default thresholds (can be overridden per pharmacy in DB)
DEFAULT_REORDER_THRESHOLD = 10   # units
DEFAULT_REORDER_QTY       = 100  # units to order

KNOWN_DISTRIBUTORS = {
    "default": {"name": "Cairo Medical Supplies", "phone": "+20223456789", "email": "orders@cairomed.eg"},
    "antibiotics": {"name": "PharmaCo Egypt", "phone": "+20212345678", "email": "orders@pharmacoeg.com"},
    "insulin": {"name": "Novo Nordisk Egypt", "phone": "+20210987654", "email": "orders@novonordisk.eg"},
}


def check_low_stock() -> dict:
    """
    ⑧ Scan pharmacy inventory — send reorder alert if below threshold.
    Returns summary of alerts sent.
    """
    logger.info("[AutoReorder] Scanning inventory for low stock...")
    alerts_sent = []
    conn = get_db_connection()

    try:
        # Get all pharmacy inventory items below threshold
        items = conn.execute("""
            SELECT
                pi.id, pi.drug_name, pi.quantity, pi.unit,
                pi.reorder_threshold, pi.reorder_quantity,
                pi.pharmacy_id,
                u.name  AS pharmacy_name,
                u.phone AS pharmacy_phone
            FROM pharmacy_inventory pi
            JOIN users u ON u.id = pi.pharmacy_id
            WHERE pi.quantity <= COALESCE(pi.reorder_threshold, ?)
        """, (DEFAULT_REORDER_THRESHOLD,)).fetchall()

        if not items:
            logger.info("[AutoReorder] All stock levels OK.")
            return {"status": "ok", "alerts": 0}

        for item in items:
            item = dict(item)
            drug     = item['drug_name']
            qty      = item['quantity']
            unit     = item.get('unit', 'units')
            pharm    = item.get('pharmacy_name', 'Pharmacy')
            pharm_ph = item.get('pharmacy_phone', '')
            reorder  = item.get('reorder_quantity') or DEFAULT_REORDER_QTY
            threshold = item.get('reorder_threshold') or DEFAULT_REORDER_THRESHOLD

            # Pick distributor by drug category (simple heuristic)
            dist_key = "antibiotics" if any(
                k in drug.lower() for k in ['cillin','mycin','cycline','floxacin','cef']
            ) else "insulin" if 'insulin' in drug.lower() else "default"
            distributor = KNOWN_DISTRIBUTORS[dist_key]

            # Compose message to pharmacy
            pharm_msg = (
                f"🔴 تنبيه مخزون منخفض — MedLink AI\n"
                f"الصيدلية: {pharm}\n"
                f"الدواء: {drug}\n"
                f"الكمية الحالية: {qty} {unit} (الحد الأدنى: {threshold})\n"
                f"تم إرسال طلب إعادة الطلب لـ {distributor['name']}\n"
                f"الكمية المطلوبة: {reorder} {unit}"
            )

            # Compose message to distributor
            dist_msg = (
                f"📦 طلب إعادة توريد — MedLink AI\n"
                f"الصيدلية: {pharm}\n"
                f"الدواء: {drug}\n"
                f"الكمية المطلوبة: {reorder} {unit}\n"
                f"يرجى التواصل مع الصيدلية على: {pharm_ph}\n"
                f"الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            # Send alerts
            if pharm_ph:
                _send_sms(pharm_ph, pharm_msg, "AutoReorder")
            _send_sms(distributor['phone'], dist_msg, "AutoReorder")

            # Log the reorder event in DB
            conn.execute("""
                INSERT INTO automation_log
                    (id, event_type, entity_id, message, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                _uuid(), 'low_stock_alert',
                item['pharmacy_id'],
                f"Low stock: {drug} ({qty} {unit})",
                'sent', datetime.now().isoformat()
            ))

            alerts_sent.append({"drug": drug, "qty": qty, "pharmacy": pharm})
            logger.info(f"[AutoReorder] Alert sent for {drug} — {qty} {unit} remaining")

        conn.commit()
        logger.info(f"[AutoReorder] Done. {len(alerts_sent)} alerts sent.")
        return {"status": "alerts_sent", "alerts": len(alerts_sent), "items": alerts_sent}

    except Exception as e:
        logger.error(f"[AutoReorder] Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# ⑨  DRUG INTERACTION ALERT — Dangerous Combo Detection
# ══════════════════════════════════════════════════════════════════════════════

# Critical drug-drug interaction pairs
# Format: (drug_a_keyword, drug_b_keyword, severity, arabic_warning)
CRITICAL_INTERACTIONS = [
    # FATAL / Life-threatening
    ("warfarin",   "aspirin",         "CRITICAL", "وارفارين + أسبرين = خطر نزيف شديد"),
    ("warfarin",   "ibuprofen",       "CRITICAL", "وارفارين + إيبوبروفين = خطر نزيف شديد"),
    ("warfarin",   "naproxen",        "CRITICAL", "وارفارين + نابروكسين = خطر نزيف"),
    ("warfarin",   "amiodarone",      "CRITICAL", "وارفارين + أميودارون = جرعة الوارفارين قد تتضاعف"),
    ("warfarin",   "clarithromycin",  "CRITICAL", "وارفارين + كلاريثروميسين = خطر نزيف"),
    ("warfarin",   "fluconazole",     "CRITICAL", "وارفارين + فلوكونازول = ارتفاع INR بشكل خطير"),
    ("sildenafil", "nitrate",         "CRITICAL", "فياجرا + نترات = انخفاض ضغط قاتل"),
    ("sildenafil", "isosorbide",      "CRITICAL", "فياجرا + إيزوسوربيد = انخفاض ضغط قاتل"),
    ("tramadol",   "sertraline",      "CRITICAL", "ترامادول + سيرترالين = متلازمة السيروتونين"),
    ("tramadol",   "fluoxetine",      "CRITICAL", "ترامادول + فلوكسيتين = متلازمة السيروتونين"),
    ("tramadol",   "escitalopram",    "CRITICAL", "ترامادول + إسيتالوبرام = متلازمة السيروتونين"),
    ("metformin",  "contrast",        "HIGH",     "ميتفورمين + صبغة تباين = خطر الحماض اللبني — وقف الميتفورمين"),
    ("methotrexate","nsaid",          "CRITICAL", "ميثوتريكسات + مضادات التهاب = تسمم الميثوتريكسات"),
    ("methotrexate","ibuprofen",      "CRITICAL", "ميثوتريكسات + إيبوبروفين = تسمم خطير"),
    ("lithium",    "nsaid",           "HIGH",     "ليثيوم + مضادات التهاب = ارتفاع مستوى الليثيوم"),
    ("lithium",    "ibuprofen",       "HIGH",     "ليثيوم + إيبوبروفين = تسمم الليثيوم"),
    ("colchicine", "clarithromycin",  "CRITICAL", "كولشيسين + كلاريثروميسين = تسمم كولشيسين مميت"),
    ("metronidazole","alcohol",       "HIGH",     "فلاجيل + كحول = تفاعل ديسولفيرام شديد"),
    ("allopurinol","azathioprine",    "CRITICAL", "ألوبيورينول + أزاثيوبرين = تسمم دموي خطير"),
    ("digoxin",    "amiodarone",      "CRITICAL", "ديجوكسين + أميودارون = ضعف القلب المضاعف"),
    ("ssri",       "maoi",            "CRITICAL", "مضادات اكتئاب SSRI + MAOI = متلازمة سيروتونين مميتة"),
    ("valproate",  "carbamazepine",   "HIGH",     "فالبروات + كاربامازيبين = تغيير مستويات الأدوية"),
    ("clopidogrel","omeprazole",      "MODERATE", "بلافيكس + أوميبرازول = تقليل فعالية بلافيكس — استخدم بانتوبرازول"),
    ("simvastatin","clarithromycin",  "CRITICAL", "سيمفاستاتين + كلاريثروميسين = خطر تحلل العضلات"),
    ("atorvastatin","clarithromycin", "HIGH",     "أتورفاستاتين + كلاريثروميسين = خطر ألم العضلات"),
    ("potassium",  "spironolactone",  "HIGH",     "مكمل بوتاسيوم + سبيرونولاكتون = ارتفاع بوتاسيوم خطير"),
    ("ace inhibitor","potassium",     "HIGH",     "مثبط ACE + بوتاسيوم = فرط بوتاسيوم الدم"),
]


def check_prescription_interactions(drug_list: list[str], patient_phone: str = None,
                                     patient_name: str = "المريض", doctor_name: str = "الطبيب") -> dict:
    """
    ⑨ Check a list of drugs for dangerous interactions.
    Sends SMS alert to patient and logs to DB.
    Returns: { alerts: [...], severity: 'CRITICAL'|'HIGH'|'MODERATE'|'NONE' }
    """
    alerts_found = []
    worst_severity = "NONE"
    severity_order = {"NONE": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

    drug_str = " ".join(d.lower() for d in drug_list)

    for pair in CRITICAL_INTERACTIONS:
        drug_a, drug_b, severity, warning = pair
        if drug_a in drug_str and drug_b in drug_str:
            alerts_found.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": severity,
                "warning": warning
            })
            if severity_order.get(severity, 0) > severity_order.get(worst_severity, 0):
                worst_severity = severity

    if not alerts_found:
        logger.info(f"[DrugInteraction] No dangerous interactions found in: {drug_list}")
        return {"alerts": [], "severity": "NONE", "safe": True}

    # Build alert message
    emoji = "🚨" if worst_severity == "CRITICAL" else "⚠️"
    severity_ar = "خطير جداً" if worst_severity == "CRITICAL" else "تحذير" if worst_severity == "HIGH" else "تنبيه"

    msg_lines = [
        f"{emoji} تحذير تفاعل دوائي — MedLink AI",
        f"المريض: {patient_name}",
        f"الدرجة: {severity_ar} ({worst_severity})",
        "",
        "التفاعلات المكتشفة:"
    ]
    for i, alert in enumerate(alerts_found, 1):
        msg_lines.append(f"{i}. {alert['warning']}")

    msg_lines.extend([
        "",
        f"يرجى مراجعة الطبيب: {doctor_name}",
        "⚕️ لا تتوقف عن الأدوية بدون مشورة طبية"
    ])

    message = "\n".join(msg_lines)

    # Send to patient
    if patient_phone:
        _send_sms(patient_phone, message, "DrugInteraction")

    # Log to DB
    try:
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO automation_log
                (id, event_type, entity_id, message, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            _uuid(), 'drug_interaction_alert',
            patient_phone or 'unknown',
            f"Severity: {worst_severity} | Drugs: {', '.join(drug_list)}",
            'sent', datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[DrugInteraction] Could not log to DB: {e}")

    logger.warning(f"[DrugInteraction] {worst_severity} interaction found! Drugs: {drug_list}")
    return {
        "alerts": alerts_found,
        "severity": worst_severity,
        "safe": False,
        "message_sent": bool(patient_phone)
    }


# ══════════════════════════════════════════════════════════════════════════════
# ⑩  NO-SHOW DETECTION — Missed Appointment Reminders
# ══════════════════════════════════════════════════════════════════════════════

def check_no_shows() -> dict:
    """
    ⑩ Find appointments that were missed (status=pending, time has passed).
    Send reminder/rebook SMS to patient.
    """
    logger.info("[NoShow] Scanning for missed appointments...")
    conn = get_db_connection()
    results = {"detected": 0, "messaged": 0, "rebooked": 0}

    try:
        now = datetime.now()
        cutoff = (now - timedelta(hours=2)).isoformat()  # 2 hours ago

        # Find appointments that were pending and have passed
        missed = conn.execute("""
            SELECT
                a.id, a.appointment_date, a.appointment_time,
                a.patient_id, a.doctor_id, a.status,
                a.reason,
                u_p.name  AS patient_name,
                u_p.phone AS patient_phone,
                u_d.name  AS doctor_name
            FROM appointments a
            JOIN users u_p ON u_p.id = a.patient_id
            JOIN users u_d ON u_d.id = a.doctor_id
            WHERE a.status = 'pending'
              AND (a.appointment_date || 'T' || COALESCE(a.appointment_time,'09:00')) <= ?
        """, (cutoff,)).fetchall()

        if not missed:
            logger.info("[NoShow] No missed appointments found.")
            return {"status": "ok", **results}

        for appt in missed:
            appt = dict(appt)
            patient_name  = appt.get('patient_name', 'المريض')
            patient_phone = appt.get('patient_phone')
            doctor_name   = appt.get('doctor_name', 'الطبيب')
            appt_date     = appt.get('appointment_date', '')
            appt_time     = appt.get('appointment_time', '')
            appt_id       = appt['id']

            # 1. Mark as no-show in DB
            conn.execute("""
                UPDATE appointments SET status = 'no_show' WHERE id = ?
            """, (appt_id,))

            results["detected"] += 1

            # 2. Send reminder message to patient
            if patient_phone:
                message = (
                    f"💙 مرحباً {patient_name} — MedLink AI\n\n"
                    f"لاحظنا أنك لم تحضر لموعدك:\n"
                    f"📅 التاريخ: {appt_date}\n"
                    f"⏰ الوقت: {appt_time}\n"
                    f"👨‍⚕️ الطبيب: {doctor_name}\n\n"
                    f"هل تريد إعادة الحجز؟\n"
                    f"رد بـ 'نعم' لإعادة الحجز\n"
                    f"أو تواصل معنا عبر التطبيق\n\n"
                    f"⚕️ صحتك تهمنا"
                )
                _send_sms(patient_phone, message, "NoShow")
                results["messaged"] += 1

            # 3. Log event
            conn.execute("""
                INSERT INTO automation_log
                    (id, event_type, entity_id, message, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                _uuid(), 'no_show_detected',
                appt_id,
                f"Patient {patient_name} missed appt with Dr. {doctor_name} on {appt_date}",
                'notified', datetime.now().isoformat()
            ))

            logger.info(f"[NoShow] {patient_name} missed appointment on {appt_date} with Dr. {doctor_name}")

        conn.commit()
        logger.info(f"[NoShow] Done. {results['detected']} no-shows, {results['messaged']} SMS sent.")
        return {"status": "done", **results}

    except Exception as e:
        logger.error(f"[NoShow] Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


def send_appointment_reminders() -> dict:
    """
    Send 24-hour advance reminder for upcoming appointments.
    Run daily at 9 AM.
    """
    logger.info("[Reminder] Sending appointment reminders...")
    conn = get_db_connection()
    sent = 0

    try:
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        upcoming = conn.execute("""
            SELECT
                a.id, a.appointment_date, a.appointment_time,
                u_p.name AS patient_name, u_p.phone AS patient_phone,
                u_d.name AS doctor_name
            FROM appointments a
            JOIN users u_p ON u_p.id = a.patient_id
            JOIN users u_d ON u_d.id = a.doctor_id
            WHERE a.appointment_date = ?
              AND a.status = 'confirmed'
              AND a.reminder_sent IS NOT 1
        """, (tomorrow,)).fetchall()

        for appt in upcoming:
            appt = dict(appt)
            patient_phone = appt.get('patient_phone')
            if not patient_phone:
                continue

            msg = (
                f"⏰ تذكير موعد — MedLink AI\n\n"
                f"مرحباً {appt['patient_name']}!\n"
                f"لديك موعد غداً:\n"
                f"👨‍⚕️ الطبيب: {appt['doctor_name']}\n"
                f"📅 التاريخ: {appt['appointment_date']}\n"
                f"⏰ الوقت: {appt.get('appointment_time', 'يُحدد لاحقاً')}\n\n"
                f"في حالة الإلغاء، يرجى التواصل مسبقاً.\n"
                f"⚕️ MedLink AI — رعاية صحية ذكية"
            )
            _send_sms(patient_phone, msg, "Reminder")

            # Mark reminder as sent
            conn.execute("UPDATE appointments SET reminder_sent = 1 WHERE id = ?", (appt['id'],))
            sent += 1

        conn.commit()
        logger.info(f"[Reminder] {sent} reminders sent for {tomorrow}")
        return {"status": "ok", "reminders_sent": sent, "date": tomorrow}

    except Exception as e:
        logger.error(f"[Reminder] Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# ㉒  DOCTOR CALENDAR — Slot Management
# ══════════════════════════════════════════════════════════════════════════════

def get_doctor_available_slots(doctor_id: str, date: str = None) -> list[dict]:
    """
    ㉒ Get available appointment slots for a doctor on a given date.
    Returns list of available time slots.
    """
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    conn = get_db_connection()
    try:
        # Get doctor's working hours
        doctor = conn.execute("""
            SELECT name, working_hours_start, working_hours_end,
                   appointment_duration_min, max_daily_appointments
            FROM users
            WHERE id = ? AND role = 'doctor'
        """, (doctor_id,)).fetchone()

        if not doctor:
            return []

        doctor = dict(doctor)
        start_h = int((doctor.get('working_hours_start') or '09:00').split(':')[0])
        end_h   = int((doctor.get('working_hours_end')   or '17:00').split(':')[0])
        duration = doctor.get('appointment_duration_min') or 30  # minutes

        # Generate all possible slots
        all_slots = []
        current = datetime.strptime(f"{date} {start_h:02d}:00", "%Y-%m-%d %H:%M")
        end_time = datetime.strptime(f"{date} {end_h:02d}:00", "%Y-%m-%d %H:%M")

        while current < end_time:
            all_slots.append(current.strftime('%H:%M'))
            current += timedelta(minutes=duration)

        # Get booked slots for this doctor on this date
        booked = conn.execute("""
            SELECT appointment_time
            FROM appointments
            WHERE doctor_id = ?
              AND appointment_date = ?
              AND status IN ('pending', 'confirmed')
        """, (doctor_id, date)).fetchall()

        booked_times = {row['appointment_time'] for row in booked}

        # Return available slots
        available = []
        for slot in all_slots:
            is_booked = slot in booked_times
            available.append({
                "time": slot,
                "available": not is_booked,
                "label": slot + (" — محجوز" if is_booked else " — متاح")
            })

        return available

    except Exception as e:
        logger.error(f"[Calendar] Error getting slots for doctor {doctor_id}: {e}")
        return []
    finally:
        conn.close()


def get_doctor_calendar_summary(doctor_id: str, days: int = 7) -> list[dict]:
    """
    ㉒ Get calendar summary for next N days for a doctor.
    Returns daily appointment counts and availability.
    """
    conn = get_db_connection()
    summary = []

    try:
        today = datetime.now().date()

        for i in range(days):
            day = today + timedelta(days=i)
            day_str = day.strftime('%Y-%m-%d')
            day_ar  = ['الإثنين','الثلاثاء','الأربعاء','الخميس','الجمعة','السبت','الأحد'][day.weekday()]

            # Count appointments
            booked = conn.execute("""
                SELECT COUNT(*) as count
                FROM appointments
                WHERE doctor_id = ?
                  AND appointment_date = ?
                  AND status IN ('pending', 'confirmed')
            """, (doctor_id, day_str)).fetchone()

            count = booked['count'] if booked else 0
            slots = get_doctor_available_slots(doctor_id, day_str)
            available_count = sum(1 for s in slots if s['available'])

            summary.append({
                "date": day_str,
                "day_ar": day_ar,
                "booked": count,
                "available": available_count,
                "total_slots": len(slots),
                "slots": slots,
                "is_full": available_count == 0
            })

        return summary

    except Exception as e:
        logger.error(f"[Calendar] Error getting summary: {e}")
        return []
    finally:
        conn.close()


def block_doctor_slots(doctor_id: str, date: str, reason: str = "إجازة") -> bool:
    """㉒ Block all slots for a doctor on a given date (vacation/leave)."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO doctor_unavailable (id, doctor_id, blocked_date, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (_uuid(), doctor_id, date, reason, datetime.now().isoformat()))
        conn.commit()
        logger.info(f"[Calendar] Blocked {date} for doctor {doctor_id} — {reason}")
        return True
    except Exception as e:
        logger.error(f"[Calendar] Error blocking slots: {e}")
        return False
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _uuid():
    import uuid
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCHEDULER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_automation_tasks() -> dict:
    """
    Run all automation tasks.
    Called by APScheduler every hour.
    """
    logger.info("[Automation] Running all automation tasks...")
    results = {}

    try:
        results['auto_reorder']  = check_low_stock()
    except Exception as e:
        results['auto_reorder']  = {"error": str(e)}

    try:
        results['no_show']       = check_no_shows()
    except Exception as e:
        results['no_show']       = {"error": str(e)}

    try:
        results['reminders']     = send_appointment_reminders()
    except Exception as e:
        results['reminders']     = {"error": str(e)}

    logger.info(f"[Automation] Tasks complete: {results}")
    return results
