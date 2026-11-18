# MedLink AI - Core Flow API Documentation

## Waiting Room API

### 1. Join Waiting Room
**Endpoint:** `POST /api/waiting-room/join`  
**Description:** Adds a patient to the waiting room queue.  
**Request Body (JSON):**
```json
{
  "patient_id": "uuid-string",
  "family_member_id": "uuid-string (optional)",
  "severity": "urgent | moderate | mild"
}
```
**Response (JSON):**
```json
{
  "message": "Joined waiting room successfully",
  "id": "waiting-room-uuid"
}
```

### 2. Fetch Queue
**Endpoint:** `GET /api/waiting-room/queue`  
**Description:** Retrieves the ordered waiting room queue. Automatically routes any 'waiting' patient older than 5 minutes to 'in_call' for demo MVP purposes. Includes Risk Badges and Child Safety Alerts.  
**Response (JSON Array):**
```json
[
  {
    "id": "waiting-room-uuid",
    "patient_id": "uuid-string",
    "status": "waiting",
    "risk_level": "urgent",
    "risk_badge": "high",
    "child_safety_alert": 1,
    "patient_user_name": "John Doe",
    "family_member_name": "Baby Doe",
    "created_at": "2023-10-01 10:00:00"
  }
]
```

### 3. Update Status
**Endpoint:** `PATCH /api/waiting-room/status`  
**Description:** Updates the status of a patient in the waiting room.  
**Request Body (JSON):**
```json
{
  "waiting_room_id": "waiting-room-uuid",
  "status": "waiting | in_call | completed"
}
```
**Response (JSON):**
```json
{
  "message": "Status updated to in_call"
}
```

## Video Demo API

### 4. Start Video Demo
**Endpoint:** `GET /api/video/start/<waiting_room_id>`  
**Description:** Updates the waiting room status to `in_call` and returns a URL to a video demo page.  
**Response (JSON):**
```json
{
  "video_url": "/video_demo?room=<waiting_room_id>"
}
```

## Appointment API

### 5. Book Appointment
**Endpoint:** `POST /api/appointments/book`  
**Description:** Books a new appointment and calculates the `risk_badge` and `child_safety_alert`.  
**Request Body (JSON):**
```json
{
  "patient_id": "uuid",
  "doctor_id": "uuid",
  "type": "video",
  "scheduled_at": "YYYY-MM-DD HH:MM:SS",
  "severity": "urgent",
  "ai_summary": "High fever, needs attention"
}
```
**Response (JSON):**
```json
{
  "message": "Appointment booked successfully",
  "appointment_id": "appt-uuid"
}
```
