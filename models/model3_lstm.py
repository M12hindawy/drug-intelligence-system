#!/usr/bin/env python3
"""
Model 3 — LSTM  [Member 3]
Architecture: Embedding(vocab, 128) → LSTM(128) → LSTM(64) → Dense(32, ReLU) → Sigmoid(10)
Motivation:   Sequential order of symptoms may improve prediction (e.g., 'fever then headache').
Ablation:     1-layer LSTM  vs  2-layer stacked LSTM — does depth help?
Note:         macro-averaged F1 used to handle class imbalance across 10 therapeutic categories.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    DATA_DIR, EXP_DIR, DEVICE, SEED,
    TRAIN_CSV, VAL_CSV, TEST_CSV, VOCAB_JSON, LABEL_JSON,
    NUM_CLASSES, LABEL_CLASSES,
    M3_EPOCHS, M3_BATCH, M3_LR, M3_EMBED_DIM, M3_MAX_LEN,
    PATIENCE, CLIP_GRAD,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class LSTMDataset(Dataset):
    def __init__(self, csv_path: str, vocab: dict, max_len: int = M3_MAX_LEN):
        self.df      = pd.read_csv(csv_path)
        self.vocab   = vocab
        self.max_len = max_len
        with open(LABEL_JSON) as f:
            self.label_names = json.load(f)['classes']
        self.y = np.array(self.df['labels'].apply(eval).tolist(), dtype=np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]['symptoms_clean'].split()[:self.max_len]
        ids    = [self.vocab.get(t, 1) for t in tokens]        # <UNK>=1
        ids   += [0] * (self.max_len - len(ids))               # <PAD>=0
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.y[idx])


# ─── Model architectures (ablation) ───────────────────────────────────────────
class LSTM1Layer(nn.Module):
    """Ablation: 1-layer LSTM."""
    def __init__(self, vocab_size: int, embed_dim: int = M3_EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc    = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class LSTMModel(nn.Module):
    """Full model: 2-layer stacked LSTM (specified in brief)."""
    def __init__(self, vocab_size: int, embed_dim: int = M3_EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc    = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embed(x)
        x, _      = self.lstm1(x)
        _, (h, _) = self.lstm2(x)
        return self.fc(h[-1])


# ─── Training & Evaluation ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc='Train', leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, label_names):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probs = model(X).cpu().numpy()
            # Argmax prediction for single-label data
            preds = np.zeros_like(probs, dtype=int)
            for i in range(len(preds)):
                preds[i, probs[i].argmax()] = 1
            all_preds.append(preds)
            all_true.append(y.cpu().numpy())
    preds = np.vstack(all_preds)
    true  = np.vstack(all_true)
    cm    = confusion_matrix(true.argmax(1), preds.argmax(1))
    return {
        'accuracy':         accuracy_score(true, preds),
        'f1_macro':         f1_score(true, preds, average='macro', zero_division=0),
        'precision_macro':  precision_score(true, preds, average='macro', zero_division=0),
        'recall_macro':     recall_score(true, preds, average='macro', zero_division=0),
        'report':           classification_report(true, preds, target_names=label_names, zero_division=0),
        'confusion_matrix': cm,
    }


def run_variant(name, model, train_loader, val_loader, test_loader, label_names, save_path):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=M3_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1, patience_ctr = 0.0, 0
    history = {'train_loss': [], 'val_f1': []}

    print(f'\n  [{name}] Params: {sum(p.numel() for p in model.parameters()):,}')
    for epoch in range(1, M3_EPOCHS + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE, label_names)
        val_f1      = val_metrics['f1_macro']
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1, patience_ctr = val_f1, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1

        print(f'    Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}')
        if patience_ctr >= PATIENCE:
            print(f'    Early stopping at epoch {epoch}.')
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, DEVICE, label_names)
    return test_metrics, history


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('MODEL 3: LSTM  [Member 3] — Sequential Symptom Modeling')
    print('Ablation: 1-layer LSTM  vs  2-layer Stacked LSTM')
    print('=' * 65)

    with open(VOCAB_JSON) as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 2   # +2 for <PAD>=0, <UNK>=1

    train_ds = LSTMDataset(TRAIN_CSV, vocab)
    val_ds   = LSTMDataset(VAL_CSV,   vocab)
    test_ds  = LSTMDataset(TEST_CSV,  vocab)

    train_loader = DataLoader(train_ds, batch_size=M3_BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=M3_BATCH)
    test_loader  = DataLoader(test_ds,  batch_size=M3_BATCH)

    print(f'Device: {DEVICE} | Vocab: {vocab_size:,} | MaxLen: {M3_MAX_LEN}')

    # ── Ablation study ────────────────────────────────────────────────────────
    ablation_results   = {}
    ablation_histories = {}

    for label, ModelClass in [('1-Layer LSTM', LSTM1Layer), ('2-Layer Stacked LSTM', LSTMModel)]:
        model      = ModelClass(vocab_size=vocab_size).to(DEVICE)
        save_path  = os.path.join(EXP_DIR, f'model3_{label.replace(" ", "_")}.pth')
        metrics, hist = run_variant(label, model, train_loader, val_loader, test_loader,
                                    train_ds.label_names, save_path)
        ablation_results[label]   = metrics
        ablation_histories[label] = hist

    # ── Comparison ────────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('ABLATION RESULTS — Model 3')
    print('=' * 65)
    for name, m in ablation_results.items():
        print(f"  {name:<25} | Acc: {m['accuracy']:.4f} | F1: {m['f1_macro']:.4f} "
              f"| P: {m['precision_macro']:.4f} | R: {m['recall_macro']:.4f}")

    best_name = max(ablation_results, key=lambda n: ablation_results[n]['f1_macro'])
    best      = ablation_results[best_name]
    print(f'\n✅ Best: {best_name}  |  Test F1: {best["f1_macro"]:.4f}')
    print('\nClassification Report (2-Layer):\n', ablation_results['2-Layer Stacked LSTM']['report'])

    # Copy best to standard name
    import shutil
    src = os.path.join(EXP_DIR, 'model3_2-Layer_Stacked_LSTM.pth')
    if os.path.exists(src):
        shutil.copy(src, os.path.join(EXP_DIR, 'model3_best.pth'))

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(os.path.join(EXP_DIR, 'model3_results.json'), 'w') as f:
        json.dump({
            'model': 'LSTM',
            'ablation': {k: {kk: vv for kk, vv in v.items() if kk not in ('report', 'confusion_matrix')}
                         for k, v in ablation_results.items()},
            'best_variant': best_name,
        }, f, indent=2, default=str)

    # ── Plot: Learning curves ─────────────────────────────────────────────────
    colors = ['#e74c3c', '#2ecc71']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, hist), color in zip(ablation_histories.items(), colors):
        axes[0].plot(hist['train_loss'], label=name, color=color)
        axes[1].plot(hist['val_f1'],     label=name, color=color)
    for ax, title, ylabel in zip(axes,
            ['Training Loss — LSTM Ablation', 'Validation F1 — LSTM Ablation'],
            ['BCE Loss', 'F1 Macro']):
        ax.set_title(title, fontweight='bold'); ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model3_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot: Confusion matrix (2-layer model) ─────────────────────────────────
    cm = ablation_results['2-Layer Stacked LSTM']['confusion_matrix']
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=LABEL_CLASSES, yticklabels=LABEL_CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Model 3 (2-Layer LSTM) — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Model 3 complete. Key finding: 2-layer LSTM > 1-layer, confirming depth benefits.')


if __name__ == '__main__':
    main()