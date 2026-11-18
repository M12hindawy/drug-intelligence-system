@echo off
echo.
echo =====================================================
echo    MedLink AI - Full Test Suite
echo =====================================================
echo.

REM ── IDs من قاعدة البيانات ──────────────────────────
set PATIENT_ID=3a7741ea-c22a-4f3c-9b8f-8987adaf1865
set PHARMACY_ID=4ef7be23-2a5d-406d-a484-60ab16fa6930
set APPT_ID=e7523403-50b9-47b9-9366-d0f271acae67

echo [TEST 1] Drug Interaction - Amoxicillin + Penicillin Allergy
echo ─────────────────────────────────────────────────────────────
curl -s -X POST http://127.0.0.1:5000/api/drugs/check-interactions ^
  -H "Content-Type: application/json" ^
  -d "{\"medications\":[{\"name\":\"Amoxicillin 500mg\"},{\"name\":\"Paracetamol 500mg\"}],\"patient_id\":\"%PATIENT_ID%\"}"
echo.
echo.

echo [TEST 2] Drug Interaction - Two NSAIDs (Ibuprofen + Diclofenac)
echo ─────────────────────────────────────────────────────────────────
curl -s -X POST http://127.0.0.1:5000/api/drugs/check-interactions ^
  -H "Content-Type: application/json" ^
  -d "{\"medications\":[{\"name\":\"Ibuprofen 400mg\"},{\"name\":\"Diclofenac 50mg\"}],\"patient_id\":\"%PATIENT_ID%\"}"
echo.
echo.

echo [TEST 3] Drug Interaction - Asthma patient + Beta-blocker
echo ─────────────────────────────────────────────────────────
curl -s -X POST http://127.0.0.1:5000/api/drugs/check-interactions ^
  -H "Content-Type: application/json" ^
  -d "{\"medications\":[{\"name\":\"Salbutamol Inhaler\"},{\"name\":\"Metoprolol 50mg\"}],\"patient_id\":\"%PATIENT_ID%\"}"
echo.
echo.

echo [TEST 4] Rating System - Rate Doctor 5 Stars
echo ─────────────────────────────────────────────
curl -s -X POST http://127.0.0.1:5000/api/appointments/rate ^
  -H "Content-Type: application/json" ^
  -d "{\"appointment_id\":\"%APPT_ID%\",\"score\":5,\"comment\":\"Excellent doctor!\"}"
echo.
echo.

echo [TEST 5] Low Stock Alert - Care Pharmacy (qty 20 threshold 25)
echo ───────────────────────────────────────────────────────────────
curl -s "http://127.0.0.1:5000/api/pharmacy/low-stock?pharmacy_id=%PHARMACY_ID%&threshold=25"
echo.
echo.

echo [TEST 6] APScheduler - Run All Tasks
echo ─────────────────────────────────────
curl -s -X POST http://127.0.0.1:5000/api/admin/run-scheduler ^
  -H "Content-Type: application/json" ^
  -d "{}"
echo.
echo.

echo =====================================================
echo    Tests Complete!
echo =====================================================
pause
