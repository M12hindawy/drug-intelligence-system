"""
Scheduler Module — MedLink AI
===============================
Handles automated background tasks:
1. Follow-up notifications - check follow_ups.next_check_at
2. Appointment reminders - 24h and 1h before scheduled_at
3. Expired appointment cleanup - mark past appointments as 'missed'
"""

import json
import sqlite3
from datetime import datetime, timedelta
from database import get_db_connection


def check_follow_ups():
    """Check for due follow-ups and mark them for notification."""
    conn = get_db_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Find follow-ups that are due
    due_followups = conn.execute("""
    SELECT f.*, u.name as patient_name, u.phone as patient_phone
    FROM follow_ups f
    JOIN users u ON f.patient_id = u.id
    WHERE f.status = 'active' AND f.next_check_at <= ?
    """, (now,)).fetchall()
    
    notifications = []
    for fu in due_followups:
        schedule = json.loads(fu['schedule'] or '[]')
        
        # Find the next pending step
        pending_step = None
        for step in schedule:
            if step.get('status') == 'pending':
                pending_step = step
                break
        
        if pending_step:
            notifications.append({
                'type': 'follow_up',
                'patient_id': fu['patient_id'],
                'patient_name': fu['patient_name'],
                'patient_phone': fu['patient_phone'],
                'message': pending_step.get('prompt', 'Time for your follow-up check!'),
                'follow_up_id': fu['id'],
                'appointment_id': fu['appointment_id']
            })
            
            # Update next_check_at to next day
            next_check = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("UPDATE follow_ups SET next_check_at = ? WHERE id = ?", (next_check, fu['id']))
    
    conn.commit()
    conn.close()
    
    if notifications:
        print(f"[Scheduler] {len(notifications)} follow-up notification(s) due:")
        for n in notifications:
            print(f"  → {n['patient_name']}: {n['message']}")
    
    return notifications


def send_appointment_reminders():
    """Send reminders for upcoming appointments (24h and 1h before)."""
    conn = get_db_connection()
    now = datetime.now()
    
    # 24-hour reminders
    window_24h_start = (now + timedelta(hours=23, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    window_24h_end = (now + timedelta(hours=24, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    
    reminders_24h = conn.execute("""
    SELECT a.*, u.name as patient_name, u.phone as patient_phone,
           d_u.name as doctor_name
    FROM appointments a
    JOIN users u ON a.patient_id = u.id
    JOIN doctors d ON a.doctor_id = d.id
    JOIN users d_u ON d.user_id = d_u.id
    WHERE a.status IN ('pending', 'confirmed')
    AND a.scheduled_at BETWEEN ? AND ?
    """, (window_24h_start, window_24h_end)).fetchall()
    
    # 1-hour reminders
    window_1h_start = (now + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    window_1h_end = (now + timedelta(hours=1, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    
    reminders_1h = conn.execute("""
    SELECT a.*, u.name as patient_name, u.phone as patient_phone,
           d_u.name as doctor_name
    FROM appointments a
    JOIN users u ON a.patient_id = u.id
    JOIN doctors d ON a.doctor_id = d.id
    JOIN users d_u ON d.user_id = d_u.id
    WHERE a.status IN ('pending', 'confirmed')
    AND a.scheduled_at BETWEEN ? AND ?
    """, (window_1h_start, window_1h_end)).fetchall()
    
    conn.close()
    
    notifications = []
    
    for appt in reminders_24h:
        notifications.append({
            'type': 'reminder_24h',
            'patient_name': appt['patient_name'],
            'patient_phone': appt['patient_phone'],
            'doctor_name': appt['doctor_name'],
            'scheduled_at': appt['scheduled_at'],
            'message': f"Reminder: You have an appointment with {appt['doctor_name']} tomorrow at {appt['scheduled_at']}."
        })
    
    for appt in reminders_1h:
        notifications.append({
            'type': 'reminder_1h',
            'patient_name': appt['patient_name'],
            'patient_phone': appt['patient_phone'],
            'doctor_name': appt['doctor_name'],
            'scheduled_at': appt['scheduled_at'],
            'message': f"URGENT: Your appointment with {appt['doctor_name']} is in 1 hour! Get ready."
        })
    
    if notifications:
        print(f"[Scheduler] {len(notifications)} appointment reminder(s):")
        for n in notifications:
            print(f"  → {n['patient_name']}: {n['message']}")
    
    return notifications


def cleanup_expired_appointments():
    """Mark past pending appointments as 'cancelled' if they weren't attended."""
    conn = get_db_connection()
    now = datetime.now()
    cutoff = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    
    # Find expired appointments (2+ hours past scheduled time, still pending)
    expired = conn.execute("""
    SELECT id, patient_id, doctor_id, scheduled_at
    FROM appointments
    WHERE status = 'pending' AND scheduled_at < ?
    """, (cutoff,)).fetchall()
    
    count = 0
    for appt in expired:
        conn.execute("""
        UPDATE appointments SET status = 'cancelled' WHERE id = ?
        """, (appt['id'],))
        count += 1
    
    conn.commit()
    conn.close()
    
    if count > 0:
        print(f"[Scheduler] Cleaned up {count} expired appointment(s).")
    
    return count


def run_all_scheduled_tasks():
    """Run all scheduled tasks. Called by APScheduler every hour."""
    print(f"\n[Scheduler] Running scheduled tasks at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    followups = check_follow_ups()
    reminders = send_appointment_reminders()
    expired_count = cleanup_expired_appointments()
    
    print(f"[Scheduler] Complete: {len(followups)} follow-ups, {len(reminders)} reminders, {expired_count} expired cleaned.")
    
    return {
        'follow_ups': len(followups),
        'reminders': len(reminders),
        'expired_cleaned': expired_count
    }


if __name__ == '__main__':
    print("Running scheduler manually...")
    result = run_all_scheduled_tasks()
    print(f"Result: {result}")
