"""
MedLink AI — Security Module
Handles: OTP store, input sanitization, rate limiting setup, CORS config
"""

import re
import time
import random
import string
import html
import logging
from datetime import datetime

logger = logging.getLogger('medlink.security')

# ─── OTP Store (in-memory, expires in 10 minutes) ─────────────────────────────
# Format: { phone: { 'code': '123456', 'expires': timestamp, 'attempts': 0 } }
OTP_STORE = {}

OTP_EXPIRY_SECONDS = 600   # 10 minutes
OTP_MAX_ATTEMPTS = 3
OTP_LENGTH = 6


def generate_otp(phone: str) -> str:
    """Generate a random OTP and store it with expiry."""
    code = ''.join(random.choices(string.digits, k=OTP_LENGTH))
    OTP_STORE[phone] = {
        'code': code,
        'expires': time.time() + OTP_EXPIRY_SECONDS,
        'attempts': 0,
        'created_at': datetime.utcnow().isoformat()
    }
    logger.info(f"[OTP] Generated OTP for {phone[-4:]}xxxx — expires in {OTP_EXPIRY_SECONDS}s")
    return code


def verify_otp(phone: str, code: str) -> tuple[bool, str]:
    """
    Verify OTP. Returns (success, message).
    Increments attempt counter. Invalidates after max attempts.
    """
    if phone not in OTP_STORE:
        return False, "لم يتم إرسال OTP لهذا الرقم. / OTP not sent for this number."

    record = OTP_STORE[phone]

    # Check expiry
    if time.time() > record['expires']:
        del OTP_STORE[phone]
        return False, "انتهت صلاحية الـ OTP. / OTP has expired. Please request a new one."

    # Check attempts
    if record['attempts'] >= OTP_MAX_ATTEMPTS:
        del OTP_STORE[phone]
        return False, "تجاوزت عدد المحاولات. / Too many failed attempts. Request a new OTP."

    # Verify code
    if record['code'] != code.strip():
        OTP_STORE[phone]['attempts'] += 1
        remaining = OTP_MAX_ATTEMPTS - OTP_STORE[phone]['attempts']
        return False, f"كود خاطئ. متبقي {remaining} محاولات. / Wrong OTP. {remaining} attempts left."

    # Success — clean up
    del OTP_STORE[phone]
    logger.info(f"[OTP] Verified successfully for {phone[-4:]}xxxx")
    return True, "تم التحقق بنجاح. / Verified successfully."


def cleanup_expired_otps():
    """Remove expired OTPs (called by scheduler)."""
    now = time.time()
    expired = [phone for phone, record in OTP_STORE.items() if now > record['expires']]
    for phone in expired:
        del OTP_STORE[phone]
    if expired:
        logger.info(f"[OTP] Cleaned up {len(expired)} expired OTPs")


# ─── Input Sanitization ────────────────────────────────────────────────────────

def sanitize_text(text: str, max_length: int = 2000) -> str:
    """
    Sanitize text input:
    - Escape HTML to prevent XSS
    - Strip leading/trailing whitespace
    - Limit length
    - Remove null bytes
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip whitespace
    text = text.strip()
    
    # Limit length
    text = text[:max_length]
    
    # Escape HTML (prevents XSS in any rendered output)
    text = html.escape(text)
    
    return text


def sanitize_phone(phone: str) -> str:
    """
    Sanitize and normalize phone number.
    Keeps only digits, +, -, spaces.
    """
    if not isinstance(phone, str):
        return ''
    
    # Keep only valid phone characters
    phone = re.sub(r'[^\d\+\-\s]', '', phone.strip())
    
    # Limit length
    phone = phone[:20]
    
    return phone


def sanitize_sql_param(value):
    """
    For use as SQLite parameter (already safe via parameterized queries).
    This is a secondary layer — just ensures type safety.
    """
    if isinstance(value, str):
        return value.replace('\x00', '').strip()[:500]
    return value


def is_valid_phone(phone: str) -> bool:
    """Validate Egyptian/international phone number format."""
    # Remove spaces and dashes
    cleaned = re.sub(r'[\s\-]', '', phone)
    # Must be 10-15 digits, optionally starting with +
    return bool(re.match(r'^\+?\d{10,15}$', cleaned))


def is_safe_query(text: str) -> bool:
    """
    Basic check for suspicious SQL injection patterns.
    Primary defense is parameterized queries — this is secondary.
    """
    suspicious = [
        r';\s*(drop|delete|truncate|alter|insert|update)\s',
        r'union\s+select',
        r'--\s*$',
        r'/\*.*\*/',
        r'xp_cmdshell',
        r'exec\s*\(',
    ]
    text_lower = text.lower()
    for pattern in suspicious:
        if re.search(pattern, text_lower):
            logger.warning(f"[Security] Suspicious input detected: {text[:100]}")
            return False
    return True
