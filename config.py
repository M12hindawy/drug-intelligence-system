#!/usr/bin/env python3
"""
config.py — Centralized configuration for Drug Intelligence System
All models and scripts import from here to ensure consistency.
"""
import sys
import os
import torch

# ─── Windows UTF-8 fix (prevents UnicodeEncodeError with Unicode chars) ────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
EXP_DIR   = os.path.join(BASE_DIR, 'experiments')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXP_DIR,  exist_ok=True)

# ─── Data files ──────────────────────────────────────────────────────────────
RAW_CSV     = os.path.join(DATA_DIR, 'raw_drugs.csv')
TRAIN_CSV   = os.path.join(DATA_DIR, 'train.csv')
VAL_CSV     = os.path.join(DATA_DIR, 'val.csv')
TEST_CSV    = os.path.join(DATA_DIR, 'test.csv')
TFIDF_PKL   = os.path.join(DATA_DIR, 'tfidf_vectorizer.pkl')
VOCAB_JSON  = os.path.join(DATA_DIR, 'vocab.json')
LABEL_JSON  = os.path.join(DATA_DIR, 'label_encoder.json')

# ─── Dataset constants ───────────────────────────────────────────────────────
SEED        = 42
NUM_CLASSES = 10
LABEL_CLASSES = [
    'Pain Relief',
    'Cold & Flu',
    'Allergy',
    'Hypertension',
    'Diabetes Type 2',
    'Infection',
    'Gastroesophageal Reflux',
    'Asthma',
    'Depression',
    'Arthritis',
]

# ─── Model hyperparameters ───────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model 1 — Basic DNN
M1_EPOCHS     = 30
M1_BATCH      = 32
M1_LR         = 1e-3

# Model 2 — DNN + Regularization
M2_EPOCHS     = 30
M2_BATCH      = 32
M2_LR         = 5e-4

# Model 3 — LSTM
M3_EPOCHS     = 40
M3_BATCH      = 32
M3_LR         = 5e-3
M3_EMBED_DIM  = 128
M3_MAX_LEN    = 8   # actual symptom length is 2-5 words

# Model 4 — BiLSTM
M4_EPOCHS     = 40
M4_BATCH      = 32
M4_LR         = 5e-3
M4_EMBED_DIM  = 128
M4_MAX_LEN    = 8   # actual symptom length is 2-5 words

# Model 5 — CNN1D
M5_EPOCHS     = 40
M5_BATCH      = 32
M5_LR         = 5e-3
M5_EMBED_DIM  = 128
M5_MAX_LEN    = 50
M5_KERNELS    = [3, 4, 5]

# Model 6 — BERT
M6_EPOCHS_P1  = 6
M6_EPOCHS_P2  = 12
M6_BATCH      = 16
M6_LR_P1      = 1e-3
M6_LR_P2      = 2e-5
M6_MAX_LEN    = 64
M6_BERT_NAME  = 'bert-base-uncased'

# ─── Training ─────────────────────────────────────────────────────────────────
PATIENCE      = 8   # early stopping patience
CLIP_GRAD     = 1.0

print(f"[Config] Device: {DEVICE} | Seed: {SEED} | Classes: {NUM_CLASSES}")
