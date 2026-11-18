#!/usr/bin/env python3
"""
inference.py — Drug Intelligence System Inference Pipeline

Given a free-text symptom description (English or Arabic transliterated),
predicts the most likely drug conditions, suggests safe alternatives,
and flags potential side effects.

Supports:
  - All 6 trained models (select via --model flag)
  - Batch inference from a file
  - JSON output for API integration

Usage:
  python inference.py --symptoms "headache fever muscle pain"
  python inference.py --symptoms "chest tightness wheezing shortness of breath" --model 4
  python inference.py --batch symptoms_list.txt --model 6
  python inference.py --interactive
"""

import sys, os, argparse, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

from config import (
    DATA_DIR, EXP_DIR, DEVICE, SEED,
    TFIDF_PKL, VOCAB_JSON, LABEL_JSON,
    NUM_CLASSES, LABEL_CLASSES,
    M3_EMBED_DIM, M3_MAX_LEN, M4_EMBED_DIM, M4_MAX_LEN,
    M5_EMBED_DIM, M5_MAX_LEN, M6_MAX_LEN, M6_BERT_NAME,
)

torch.manual_seed(SEED)

# ─── Import model architectures ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

# Drug alternative knowledge base (rule-based safe alternatives)
SAFE_ALTERNATIVES = {
    'Pain Relief': [
        ('Paracetamol 500mg', 'First-line; safest for liver, avoid in heavy drinkers'),
        ('Ibuprofen 400mg',   'Better anti-inflammatory; avoid with peptic ulcer'),
        ('Diclofenac 50mg',   'Strong NSAID; avoid with renal impairment'),
    ],
    'Cold & Flu': [
        ('Paracetamol + Vit C', 'Symptomatic relief; rest + hydration recommended'),
        ('Cetirizine 10mg',     'For allergic component; non-drowsy antihistamine'),
    ],
    'Allergy': [
        ('Cetirizine 10mg',   '2nd-gen antihistamine; non-sedating'),
        ('Loratadine 10mg',   'Non-drowsy; good for daytime use'),
        ('Fexofenadine 120mg','Least sedating; preferred for drivers'),
    ],
    'Hypertension': [
        ('Amlodipine 5mg',    'CCB; well-tolerated, good for elderly'),
        ('Enalapril 10mg',    'ACE inhibitor; protective for kidneys (diabetes)'),
        ('Losartan 50mg',     'ARB; use when ACE-inhibitor causes cough'),
    ],
    'Diabetes Type 2': [
        ('Metformin 500mg',   'First-line; reduces cardiovascular risk'),
        ('Sitagliptin 100mg', 'DPP-4 inhibitor; weight-neutral'),
        ('Empagliflozin 10mg','SGLT-2; cardiovascular + renal protection'),
    ],
    'Infection': [
        ('Amoxicillin 500mg',     'Broad-spectrum; first-line community infections'),
        ('Clarithromycin 500mg',  'For penicillin allergy; respiratory infections'),
        ('Ciprofloxacin 500mg',   'Fluoroquinolone; urinary tract infections'),
    ],
    'Gastroesophageal Reflux': [
        ('Omeprazole 20mg',     'PPI; highly effective for acid suppression'),
        ('Pantoprazole 40mg',   'Longer duration than omeprazole'),
        ('Ranitidine (if available)', 'H2 blocker; milder cases'),
    ],
    'Asthma': [
        ('Salbutamol Inhaler',    'Short-acting beta-agonist; rescue inhaler'),
        ('Fluticasone Inhaler',   'ICS; controller medication (daily)'),
        ('Montelukast 10mg',      'Leukotriene antagonist; add-on therapy'),
    ],
    'Depression': [
        ('Sertraline 50mg',      'SSRI; first-line; good tolerability'),
        ('Escitalopram 10mg',    'SSRI; fewer drug interactions'),
        ('Fluoxetine 20mg',      'Long half-life; less withdrawal risk'),
    ],
    'Arthritis': [
        ('Diclofenac 50mg',  'NSAID; anti-inflammatory for acute flares'),
        ('Celecoxib 200mg',  'COX-2 selective; less GI risk than NSAIDs'),
        ('Methotrexate 7.5mg','DMARD; for rheumatoid arthritis (specialist only)'),
    ],
}

# Known side effects for detected conditions
SIDE_EFFECT_WARNINGS = {
    'Pain Relief':              ['Paracetamol overdose → liver failure', 'NSAIDs → GI bleeding, renal impairment'],
    'Cold & Flu':               ['Decongestants → hypertension', 'Antihistamines → sedation'],
    'Allergy':                  ['First-gen antihistamines → sedation, anticholinergic effects'],
    'Hypertension':             ['ACE inhibitors → dry cough', 'Beta-blockers → bradycardia, fatigue'],
    'Diabetes Type 2':          ['Metformin → lactic acidosis (rare, with renal failure)', 'Sulfonylureas → hypoglycemia'],
    'Infection':                ['Antibiotics → C. difficile colitis', 'Quinolones → tendinopathy', 'Aminoglycosides → ototoxicity'],
    'Gastroesophageal Reflux':  ['PPIs → hypomagnesemia (long-term)', 'Rebound acid hypersecretion on stop'],
    'Asthma':                   ['Long-acting beta-agonists → paradoxical bronchospasm (rare)', 'ICS → oral thrush'],
    'Depression':               ['SSRIs → serotonin syndrome (with MAOI)', 'Initial anxiety/insomnia first 2 weeks'],
    'Arthritis':                ['NSAIDs → peptic ulcers', 'Methotrexate → hepatotoxicity, pulmonary toxicity'],
}


# ─── Text preprocessing (matches preprocessing.py) ───────────────────────────
import re

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'it', 'its',
    'this', 'that', 'these', 'those', 'as', 'if', 'then', 'than',
    'when', 'where', 'which', 'who', 'what', 'how', 'not', 'also',
}

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = text.lower().strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ─── Model loaders ────────────────────────────────────────────────────────────
def _load_tfidf():
    with open(TFIDF_PKL, 'rb') as f:
        return pickle.load(f)

def _load_vocab():
    with open(VOCAB_JSON) as f:
        return json.load(f)

def _tokenize_sequence(text: str, vocab: dict, max_len: int) -> torch.Tensor:
    tokens = text.split()[:max_len]
    ids    = [vocab.get(t, 1) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def load_model(model_id: int):
    """Load a trained model by ID."""
    ckpt_map = {
        1: 'model1_best.pth',
        2: 'model2_best.pth',
        3: 'model3_best.pth',
        4: 'model4_best.pth',
        5: 'model5_best.pth',
        6: 'model6_best.pth',
    }
    ckpt_path = os.path.join(EXP_DIR, ckpt_map[model_id])
    if not os.path.exists(ckpt_path):
        return None, None

    if model_id == 1:
        from models.model1_basic_dnn import BasicDNN
        tfidf = _load_tfidf()
        dim   = len(tfidf.vocabulary_)
        model = BasicDNN(input_dim=dim).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model, ('tfidf', tfidf)

    elif model_id == 2:
        from models.model2_dnn_regularized import RegularizedDNN
        tfidf = _load_tfidf()
        dim   = len(tfidf.vocabulary_)
        model = RegularizedDNN(input_dim=dim).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model, ('tfidf', tfidf)

    elif model_id == 3:
        from models.model3_lstm import LSTMModel
        vocab      = _load_vocab()
        vocab_size = len(vocab) + 2
        model = LSTMModel(vocab_size=vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model, ('seq', vocab, M3_MAX_LEN)

    elif model_id == 4:
        from models.model4_bilstm import BiLSTMModel
        vocab      = _load_vocab()
        vocab_size = len(vocab) + 2
        model = BiLSTMModel(vocab_size=vocab_size).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model, ('seq', vocab, M4_MAX_LEN)

    elif model_id == 5:
        from models.model5_cnn1d import CNN1DModel
        vocab      = _load_vocab()
        vocab_size = len(vocab) + 2
        model = CNN1DModel(vocab_size=vocab_size, kernel_size=3).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model, ('seq', vocab, M5_MAX_LEN)

    elif model_id == 6:
        try:
            from transformers import BertTokenizer
            from models.model6_bert import BERTClassifier
            tokenizer = BertTokenizer.from_pretrained(M6_BERT_NAME)
            model     = BERTClassifier().to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            return model, ('bert', tokenizer, M6_MAX_LEN)
        except ImportError:
            print('[ERROR] transformers not installed.')
            return None, None

    return None, None


def predict(model, preprocessor, text: str, threshold: float = 0.4):
    """Run inference and return list of (class, confidence) tuples."""
    model.eval()
    cleaned = clean_text(text)
    kind    = preprocessor[0]

    with torch.no_grad():
        if kind == 'tfidf':
            tfidf = preprocessor[1]
            x     = torch.tensor(
                tfidf.transform([cleaned]).toarray().astype(np.float32)
            ).to(DEVICE)
            probs = model(x).cpu().numpy()[0]

        elif kind == 'seq':
            _, vocab, max_len = preprocessor
            x     = _tokenize_sequence(cleaned, vocab, max_len).to(DEVICE)
            probs = model(x).cpu().numpy()[0]

        elif kind == 'bert':
            _, tokenizer, max_len = preprocessor
            enc   = tokenizer(cleaned, max_length=max_len, padding='max_length',
                              truncation=True, return_tensors='pt')
            ids   = enc['input_ids'].to(DEVICE)
            mask  = enc['attention_mask'].to(DEVICE)
            probs = model(ids, mask).cpu().numpy()[0]

    predictions = [
        (LABEL_CLASSES[i], float(probs[i]))
        for i in range(NUM_CLASSES)
        if probs[i] >= threshold
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)

    if not predictions:
        # Return top-1 even below threshold
        top = int(np.argmax(probs))
        predictions = [(LABEL_CLASSES[top], float(probs[top]))]

    return predictions, probs.tolist()


def format_output(symptoms: str, predictions, all_probs, model_id: int) -> dict:
    """Build structured output with predictions, alternatives, and warnings."""
    result = {
        'input_symptoms': symptoms,
        'model_used':     f'Model {model_id}',
        'predictions':    [],
    }

    for condition, confidence in predictions:
        alts     = SAFE_ALTERNATIVES.get(condition, [])
        warnings = SIDE_EFFECT_WARNINGS.get(condition, [])
        result['predictions'].append({
            'condition':        condition,
            'confidence':       round(confidence, 4),
            'confidence_pct':   f'{confidence * 100:.1f}%',
            'safe_alternatives': [{'drug': d, 'note': n} for d, n in alts],
            'side_effect_warnings': warnings,
        })

    result['all_class_probs'] = {
        cls: round(p, 4) for cls, p in zip(LABEL_CLASSES, all_probs)
    }
    return result


def print_result(result: dict):
    print('\n' + '═' * 60)
    print(f'  DRUG INTELLIGENCE SYSTEM — {result["model_used"]}')
    print('═' * 60)
    print(f'  Symptoms: "{result["input_symptoms"]}"')
    print()

    for i, pred in enumerate(result['predictions'], 1):
        print(f'  [{i}] {pred["condition"]}  ({pred["confidence_pct"]} confidence)')
        print(f'      Safe alternatives:')
        for alt in pred['safe_alternatives'][:3]:
            print(f'        • {alt["drug"]} — {alt["note"]}')
        print(f'      ⚠ Side-effect warnings:')
        for w in pred['side_effect_warnings'][:2]:
            print(f'        ! {w}')
        print()
    print('═' * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Drug Intelligence System — Inference')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--symptoms',     type=str, help='Symptom description (free text)')
    group.add_argument('--batch',        type=str, help='Path to text file, one symptom per line')
    group.add_argument('--interactive',  action='store_true', help='Interactive mode')
    parser.add_argument('--model',       type=int, default=4, choices=[1,2,3,4,5,6],
                        help='Which model to use (default: 4 = BiLSTM)')
    parser.add_argument('--threshold',   type=float, default=0.4,
                        help='Confidence threshold for positive predictions (default: 0.4)')
    parser.add_argument('--json-output', action='store_true', help='Output raw JSON')
    args = parser.parse_args()

    # Verify data files exist
    for path, name in [(TFIDF_PKL, 'TF-IDF'), (VOCAB_JSON, 'Vocab'), (LABEL_JSON, 'Labels')]:
        if not os.path.exists(path):
            print(f'[ERROR] {name} file not found: {path}')
            print('Run data/preprocessing.py first.')
            return 1

    print(f'Loading Model {args.model}...')
    model, preprocessor = load_model(args.model)

    if model is None:
        print(f'[ERROR] Model {args.model} checkpoint not found in {EXP_DIR}')
        print('Train the model first using models/model{N}_*.py')
        return 1

    model.eval()
    print('Model loaded ✅')

    # ── Single symptom ────────────────────────────────────────────────────────
    if args.symptoms:
        preds, all_probs = predict(model, preprocessor, args.symptoms, args.threshold)
        result = format_output(args.symptoms, preds, all_probs, args.model)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print_result(result)

    # ── Batch mode ────────────────────────────────────────────────────────────
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f'[ERROR] Batch file not found: {args.batch}')
            return 1
        with open(args.batch) as f:
            lines = [l.strip() for l in f if l.strip()]
        all_results = []
        for line in lines:
            preds, all_probs = predict(model, preprocessor, line, args.threshold)
            result = format_output(line, preds, all_probs, args.model)
            all_results.append(result)
            if not args.json_output:
                print_result(result)
        if args.json_output:
            print(json.dumps(all_results, indent=2))

    # ── Interactive mode ──────────────────────────────────────────────────────
    elif args.interactive:
        print('\n' + '═' * 60)
        print('  DRUG INTELLIGENCE SYSTEM — Interactive Mode')
        print(f'  Using Model {args.model} | Type "quit" to exit')
        print('═' * 60)
        while True:
            try:
                text = input('\nEnter symptoms: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if not text:
                continue
            preds, all_probs = predict(model, preprocessor, text, args.threshold)
            result = format_output(text, preds, all_probs, args.model)
            print_result(result)

    return 0


if __name__ == '__main__':
    sys.exit(main())
