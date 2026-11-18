"""
MedLink AI — Logging System
Configures file-based logging for all app modules.
"""

import os
import logging
import logging.handlers
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(app_name: str = 'medlink', level=logging.INFO):
    """
    Configure logging to both file and console.
    - INFO+ → logs/medlink.log (rotating, max 10MB × 5 backups)
    - ERROR+ → logs/errors.log (rotating, max 5MB × 3 backups)
    """
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ── Console Handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File Handler (all logs) ──────────────────────────────────────────────
    all_log_path = os.path.join(LOG_DIR, f'{app_name}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        all_log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ── Error File Handler ───────────────────────────────────────────────────
    error_log_path = os.path.join(LOG_DIR, 'errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)

    logging.info(f"[Logging] System initialized — logs at: {LOG_DIR}")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a module."""
    return logging.getLogger(f'medlink.{name}')
