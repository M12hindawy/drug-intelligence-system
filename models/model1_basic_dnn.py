#!/usr/bin/env python3
"""
Model 1 — Basic DNN Baseline  [Member 2]
Architecture: Input (TF-IDF) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Sigmoid(10)
Motivation:   Establish a simple Dense baseline; all 5 other models are benchmarked against this.
Task:         Multi-label classification (10 conditions from symptoms)
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
import pickle
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
    TRAIN_CSV, VAL_CSV, TEST_CSV, TFIDF_PKL, LABEL_JSON,
    NUM_CLASSES, LABEL_CLASSES,
    M1_EPOCHS, M1_BATCH, M1_LR, PATIENCE, CLIP_GRAD,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Dataset ──────────────────────────────────────────────────────────────────
class DrugDataset(Dataset):
    """TF-IDF based dataset for DNN models."""
    def __init__(self, csv_path: str, tfidf_path: str):
        self.df = pd.read_csv(csv_path)
        with open(LABEL_JSON) as f:
            self.label_names = json.load(f)['classes']
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)
        self.X = self.tfidf.transform(self.df['symptoms_clean']).toarray().astype(np.float32)
        self.y = np.array(self.df['labels'].apply(eval).tolist(), dtype=np.float32)

    def __len__(self):  return len(self.df)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ─── Model ────────────────────────────────────────────────────────────────────
class BasicDNN(nn.Module):
    """
    Baseline DNN: exactly as specified in project brief.
    Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Sigmoid(10)
    """
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ─── Training ────────────────────────────────────────────────────────────────
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
            # Primary: threshold at 0.3; fallback: argmax when all below threshold
            preds = (probs > 0.3).astype(int)
            for i in range(len(preds)):
                if preds[i].sum() == 0:
                    preds[i, probs[i].argmax()] = 1
            all_preds.append(preds)
            all_true.append(y.cpu().numpy())
    preds = np.vstack(all_preds)
    true  = np.vstack(all_true)
    cm = confusion_matrix(true.argmax(1), preds.argmax(1))
    return {
        'accuracy':         accuracy_score(true, preds),
        'f1_macro':         f1_score(true, preds, average='macro', zero_division=0),
        'precision_macro':  precision_score(true, preds, average='macro', zero_division=0),
        'recall_macro':     recall_score(true, preds, average='macro', zero_division=0),
        'report':           classification_report(true, preds, target_names=label_names, zero_division=0),
        'confusion_matrix': cm,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('MODEL 1: Basic DNN Baseline  [Member 2]')
    print('Architecture: Input → 128 → 64 → 32 → Sigmoid(10)')
    print('=' * 65)

    if not os.path.exists(TFIDF_PKL):
        print('[ERROR] Run data/preprocessing.py first to generate TF-IDF!')
        return

    train_ds = DrugDataset(TRAIN_CSV, TFIDF_PKL)
    val_ds   = DrugDataset(VAL_CSV,   TFIDF_PKL)
    test_ds  = DrugDataset(TEST_CSV,  TFIDF_PKL)

    train_loader = DataLoader(train_ds, batch_size=M1_BATCH, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=M1_BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=M1_BATCH, shuffle=False)

    model     = BasicDNN(input_dim=train_ds.X.shape[1]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=M1_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Device: {DEVICE} | Params: {total_params:,} | Input dim: {train_ds.X.shape[1]:,}')

    best_f1, patience_ctr = 0.0, 0
    history = {'train_loss': [], 'val_f1': [], 'val_loss': []}
    best_path = os.path.join(EXP_DIR, 'model1_best.pth')

    for epoch in range(1, M1_EPOCHS + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE, train_ds.label_names)
        val_f1      = val_metrics['f1_macro']

        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1, patience_ctr = val_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ctr += 1

        print(f'Epoch {epoch:02d}/{M1_EPOCHS} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}')
        if patience_ctr >= PATIENCE:
            print(f'Early stopping at epoch {epoch}.')
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, DEVICE, train_ds.label_names)

    print('\n' + '=' * 45)
    print('MODEL 1 TEST RESULTS (Baseline Reference)')
    print('=' * 45)
    print(f'Accuracy   : {test_metrics["accuracy"]:.4f}')
    print(f'F1 (macro) : {test_metrics["f1_macro"]:.4f}')
    print(f'Precision  : {test_metrics["precision_macro"]:.4f}')
    print(f'Recall     : {test_metrics["recall_macro"]:.4f}')
    print(f'Parameters : {total_params:,}')
    print('\nClassification Report:\n', test_metrics['report'])

    # ── Save JSON ────────────────────────────────────────────────────────────
    save_data = {
        'model': 'BasicDNN',
        'architecture': 'Input → 128 → 64 → 32 → Sigmoid(10)',
        'total_params': total_params,
        'test_metrics': {k: v for k, v in test_metrics.items()
                         if k not in ('report', 'confusion_matrix')},
        'note': 'Reference baseline; all other models compared against this.',
    }
    with open(os.path.join(EXP_DIR, 'model1_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # ── Plot: Learning curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    axes[0].set_title('Model 1 — Training Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_f1'], 'g-o', markersize=4, label='Val F1 Macro')
    axes[1].axhline(y=test_metrics['f1_macro'], color='r', linestyle='--', label=f'Test F1={test_metrics["f1_macro"]:.3f}')
    axes[1].set_title('Model 1 — Validation F1', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Macro')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model1_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot: Confusion matrix ────────────────────────────────────────────────
    cm = test_metrics['confusion_matrix']
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_CLASSES, yticklabels=LABEL_CLASSES, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title('Model 1 (BasicDNN) — Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model1_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print('\n✅ Model 1 complete. This is the REFERENCE baseline.')
    print(f'   → Best checkpoint: {best_path}')
    print(f'   → Plots saved to : {EXP_DIR}')


if __name__ == '__main__':
    main()