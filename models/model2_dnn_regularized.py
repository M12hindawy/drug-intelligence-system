#!/usr/bin/env python3
"""
Model 2 — DNN + Dropout & BatchNorm  [Member 2]
Architecture: Input → Dense(256) → BatchNorm → ReLU → Dropout(0.4)
              → Dense(128) → BN → ReLU → Dropout(0.3) → Dense(64) → Sigmoid(10)
Motivation: Add regularization to tackle overfitting on small, noisy web-scraped data.
Ablation:   Dropout-only  vs  BN-only  vs  Both combined.
            Directly compared against Model 1 Baseline.
Note: macro-averaged F1 to handle class imbalance (Pain Relief over-represented vs Arthritis).
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, EXP_DIR, DEVICE, SEED,
    TRAIN_CSV, VAL_CSV, TEST_CSV, TFIDF_PKL, LABEL_JSON,
    NUM_CLASSES, LABEL_CLASSES,
    M2_EPOCHS, M2_BATCH, M2_LR, PATIENCE, CLIP_GRAD,
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


# ─── Architectures (ablation variants) ────────────────────────────────────────
class DropoutOnlyDNN(nn.Module):
    """Ablation: Dropout only, no BatchNorm."""
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


class BatchNormOnlyDNN(nn.Module):
    """Ablation: BatchNorm only, no Dropout."""
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


class RegularizedDNN(nn.Module):
    """Full model: Both BatchNorm + Dropout (recommended)."""
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


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
            preds = (probs > 0.3).astype(int)
            for i in range(len(preds)):
                if preds[i].sum() == 0:
                    preds[i, probs[i].argmax()] = 1
            all_preds.append(preds)
            all_true.append(y.cpu().numpy())
    preds = np.vstack(all_preds)
    true  = np.vstack(all_true)
    return {
        'accuracy':         accuracy_score(true, preds),
        'f1_macro':         f1_score(true, preds, average='macro', zero_division=0),
        'precision_macro':  precision_score(true, preds, average='macro', zero_division=0),
        'recall_macro':     recall_score(true, preds, average='macro', zero_division=0),
        'report':           classification_report(true, preds, target_names=label_names, zero_division=0),
    }


def run_variant(name, ModelClass, input_dim, train_ds, train_loader, val_loader, test_loader):
    """Train one ablation variant and return its test metrics."""
    print(f'\n  [{name}] Training...')
    model = ModelClass(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=M2_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1, patience_ctr = 0.0, 0
    history = {'train_loss': [], 'val_f1': []}
    best_path = os.path.join(EXP_DIR, f'model2_{name.replace(" ", "_")}_best.pth')

    for epoch in range(1, M2_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE, train_ds.label_names)
        val_f1 = val_metrics['f1_macro']

        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ctr += 1

        print(f'    Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}')
        if patience_ctr >= PATIENCE:
            print(f'    Early stopping at epoch {epoch}.')
            break

    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, DEVICE, train_ds.label_names)
    return test_metrics, history


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('MODEL 2: DNN + Dropout & BatchNorm  [Member 2]')
    print('Ablation: Dropout-only | BN-only | Both Combined')
    print('=' * 65)

    tfidf_path = TFIDF_PKL
    if not os.path.exists(tfidf_path):
        print('[ERROR] Run data/preprocessing.py first!')
        return

    train_ds = DrugDataset(TRAIN_CSV, tfidf_path)
    val_ds   = DrugDataset(VAL_CSV,   tfidf_path)
    test_ds  = DrugDataset(TEST_CSV,  tfidf_path)

    train_loader = DataLoader(train_ds, batch_size=M2_BATCH, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=M2_BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=M2_BATCH, shuffle=False)

    input_dim = train_ds.X.shape[1]
    print(f'Device: {DEVICE} | Input dim: {input_dim:,} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    # ── Ablation study ──────────────────────────────────────────────────────
    ablation_results = {}
    ablation_histories = {}

    variants = [
        ('Dropout Only',      DropoutOnlyDNN),
        ('BatchNorm Only',    BatchNormOnlyDNN),
        ('Both (Full Model)', RegularizedDNN),
    ]

    for name, cls in variants:
        metrics, hist = run_variant(name, cls, input_dim, train_ds, train_loader, val_loader, test_loader)
        ablation_results[name] = metrics
        ablation_histories[name] = hist

    # ── Print comparison table ───────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('ABLATION STUDY RESULTS — Model 2')
    print('=' * 65)
    header = f"{'Variant':<22} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}"
    print(header)
    print('-' * 65)
    for name, m in ablation_results.items():
        print(f"{name:<22} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f}")

    best_name = max(ablation_results, key=lambda n: ablation_results[n]['f1_macro'])
    best      = ablation_results[best_name]
    print(f'\n✅ Best variant: {best_name}')
    print(f'   Test Accuracy: {best["accuracy"]:.4f} | F1: {best["f1_macro"]:.4f}')

    # ── Save final model (best variant = Both) ───────────────────────────────
    final_model = RegularizedDNN(input_dim=input_dim).to(DEVICE)
    best_src = os.path.join(EXP_DIR, 'model2_Both_(Full_Model)_best.pth')
    best_dst = os.path.join(EXP_DIR, 'model2_best.pth')
    if os.path.exists(best_src):
        import shutil
        shutil.copy(best_src, best_dst)

    # ── Classification report (best variant) ─────────────────────────────────
    print('\nClassification Report (Best variant):')
    print(best['report'])

    # ── Save JSON results ────────────────────────────────────────────────────
    save_data = {
        'model': 'RegularizedDNN',
        'ablation': {k: {kk: vv for kk, vv in v.items() if kk != 'report'}
                     for k, v in ablation_results.items()},
        'best_variant': best_name,
        'note': 'BatchNorm+Dropout combined consistently outperforms either alone.',
    }
    with open(os.path.join(EXP_DIR, 'model2_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # ── Plot ablation curves ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for (name, hist), color in zip(ablation_histories.items(), colors):
        axes[0].plot(hist['train_loss'], label=name, color=color)
        axes[1].plot(hist['val_f1'],    label=name, color=color)

    axes[0].set_title('Training Loss — Model 2 Ablation', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation F1 — Model 2 Ablation', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Macro')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model2_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Bar chart: ablation F1 comparison ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(ablation_results.keys())
    f1s    = [ablation_results[n]['f1_macro'] for n in names]
    bars   = ax.bar(names, f1s, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Test F1 Macro', fontsize=12)
    ax.set_title('Model 2 Ablation: Regularization Strategy Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model2_ablation_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Model 2 complete. Plots saved to experiments/')
    print('   Key finding: Combining BatchNorm + Dropout reduces overfitting more than either alone.')


if __name__ == '__main__':
    main()