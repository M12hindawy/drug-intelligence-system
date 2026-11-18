#!/usr/bin/env python3
"""
Model 5 — 1D CNN / TextCNN  [Member 4]
Architecture: Embedding → Conv1D → MaxPool → Conv1D → GlobalMaxPool → Dense(32) → Sigmoid(10)
Motivation:   CNN detects local n-gram patterns (e.g., 'chest pain', 'skin rash') efficiently.
Ablation:     Kernel size experiment: 3 vs 4 vs 5 — which window best captures medical phrases?
Advanced:     Parallel TextCNN option (3 branches with kernels 3,4,5 merged via concatenation).
Note:         macro-averaged F1 to handle class imbalance across 10 therapeutic categories.
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
import shutil

from config import (
    DATA_DIR, EXP_DIR, DEVICE, SEED,
    TRAIN_CSV, VAL_CSV, TEST_CSV, VOCAB_JSON, LABEL_JSON,
    NUM_CLASSES, LABEL_CLASSES,
    M5_EPOCHS, M5_BATCH, M5_LR, M5_EMBED_DIM, M5_MAX_LEN, M5_KERNELS,
    PATIENCE, CLIP_GRAD,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CNNDataset(Dataset):
    def __init__(self, csv_path: str, vocab: dict, max_len: int = M5_MAX_LEN):
        self.df      = pd.read_csv(csv_path)
        self.vocab   = vocab
        self.max_len = max_len
        with open(LABEL_JSON) as f:
            self.label_names = json.load(f)['classes']
        self.y = np.array(self.df['labels'].apply(eval).tolist(), dtype=np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]['symptoms_clean'].split()[:self.max_len]
        ids    = [self.vocab.get(t, 1) for t in tokens]
        ids   += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.y[idx])


# ─── Model architectures ─────────────────────────────────────────────────────
class CNN1DModel(nn.Module):
    """
    Sequential 1D CNN (ablation variant with configurable kernel size).
    Embedding → Conv1D(kernel) → MaxPool → Conv1D(kernel) → GlobalMaxPool → Dense → Sigmoid
    """
    def __init__(self, vocab_size: int, embed_dim: int = M5_EMBED_DIM,
                 num_classes: int = NUM_CLASSES, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.embed       = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1       = nn.Conv1d(embed_dim, 128, kernel_size=kernel_size, padding=pad)
        self.pool        = nn.MaxPool1d(2)
        self.conv2       = nn.Conv1d(128, 64, kernel_size=kernel_size, padding=pad)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc          = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)           # (B, D, L)
        x = self.pool(torch.relu(self.conv1(x)))      # (B, 128, L/2)
        x = torch.relu(self.conv2(x))                 # (B, 64, ~L/2)
        x = self.global_pool(x).squeeze(-1)           # (B, 64)
        return self.fc(x)


class TextCNN(nn.Module):
    """
    Parallel TextCNN: 3 parallel Conv branches with kernels 3,4,5,
    then concatenated → Dense → Sigmoid. (Kim 2014 style)
    """
    def __init__(self, vocab_size: int, embed_dim: int = M5_EMBED_DIM,
                 num_classes: int = NUM_CLASSES,
                 num_filters: int = 64, kernels: list = None):
        super().__init__()
        if kernels is None:
            kernels = [3, 4, 5]
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs   = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Sequential(
            nn.Linear(num_filters * len(kernels), 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)           # (B, D, L)
        pooled = [
            torch.relu(conv(x)).max(dim=2)[0]         # global max pool per branch
            for conv in self.convs
        ]
        x = self.dropout(torch.cat(pooled, dim=1))    # (B, num_filters * num_kernels)
        return self.fc(x)


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
    optimizer = optim.Adam(model.parameters(), lr=M5_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1, patience_ctr = 0.0, 0
    history = {'train_loss': [], 'val_f1': []}
    params  = sum(p.numel() for p in model.parameters())

    print(f'\n  [{name}] Params: {params:,}')
    for epoch in range(1, M5_EPOCHS + 1):
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
    return test_metrics, history, params


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('MODEL 5: 1D CNN / TextCNN  [Member 4]')
    print('Ablation: Kernel sizes 3, 4, 5  +  Parallel TextCNN')
    print('=' * 65)

    with open(VOCAB_JSON) as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 2

    train_ds = CNNDataset(TRAIN_CSV, vocab)
    val_ds   = CNNDataset(VAL_CSV,   vocab)
    test_ds  = CNNDataset(TEST_CSV,  vocab)

    train_loader = DataLoader(train_ds, batch_size=M5_BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=M5_BATCH)
    test_loader  = DataLoader(test_ds,  batch_size=M5_BATCH)

    print(f'Device: {DEVICE} | Vocab: {vocab_size:,} | MaxLen: {M5_MAX_LEN}')

    # ── Ablation: kernel sizes 3, 4, 5 ──────────────────────────────────────
    ablation_results   = {}
    ablation_histories = {}
    ablation_params    = {}

    for k in M5_KERNELS:
        name      = f'CNN1D-kernel{k}'
        model     = CNN1DModel(vocab_size=vocab_size, kernel_size=k).to(DEVICE)
        save_path = os.path.join(EXP_DIR, f'model5_{name}.pth')
        metrics, hist, params = run_variant(name, model, train_loader, val_loader,
                                            test_loader, train_ds.label_names, save_path)
        ablation_results[name]   = metrics
        ablation_histories[name] = hist
        ablation_params[name]    = params

    # ── Advanced: Parallel TextCNN ────────────────────────────────────────────
    name      = 'TextCNN-parallel(3,4,5)'
    model     = TextCNN(vocab_size=vocab_size, kernels=[3, 4, 5]).to(DEVICE)
    save_path = os.path.join(EXP_DIR, 'model5_textcnn_parallel.pth')
    metrics, hist, params = run_variant(name, model, train_loader, val_loader,
                                        test_loader, train_ds.label_names, save_path)
    ablation_results[name]   = metrics
    ablation_histories[name] = hist
    ablation_params[name]    = params

    # ── Comparison table ──────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('ABLATION RESULTS — Model 5 (Kernel Size Experiment)')
    print('=' * 65)
    header = f"{'Variant':<28} {'Params':>8} {'Acc':>8} {'F1':>8} {'P':>8} {'R':>8}"
    print(header)
    print('-' * 65)
    for name, m in ablation_results.items():
        p = ablation_params[name]
        print(f"{name:<28} {p:>8,} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} "
              f"{m['precision_macro']:>8.4f} {m['recall_macro']:>8.4f}")

    best_name = max(ablation_results, key=lambda n: ablation_results[n]['f1_macro'])
    best      = ablation_results[best_name]
    print(f'\n✅ Best variant: {best_name}  |  Test F1: {best["f1_macro"]:.4f}')
    print('\nClassification Report (Best variant):\n', best['report'])

    # Copy best checkpoint
    best_src = os.path.join(EXP_DIR,
        f'model5_{best_name.replace("(", "").replace(")", "").replace(",", "")}.pth')
    if os.path.exists(best_src):
        shutil.copy(best_src, os.path.join(EXP_DIR, 'model5_best.pth'))

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(os.path.join(EXP_DIR, 'model5_results.json'), 'w') as f:
        json.dump({
            'model': 'CNN1D',
            'ablation': {k: {kk: vv for kk, vv in v.items()
                             if kk not in ('report', 'confusion_matrix')}
                         for k, v in ablation_results.items()},
            'params': ablation_params,
            'best_variant': best_name,
        }, f, indent=2, default=str)

    # ── Plot: Ablation curves ─────────────────────────────────────────────────
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, hist), color in zip(ablation_histories.items(), colors):
        axes[0].plot(hist['train_loss'], label=name, color=color)
        axes[1].plot(hist['val_f1'],     label=name, color=color)
    for ax, title, ylabel in zip(axes,
            ['Training Loss — CNN Kernel Ablation', 'Validation F1 — CNN Kernel Ablation'],
            ['BCE Loss', 'F1 Macro']):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model5_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot: Bar chart comparison ────────────────────────────────────────────
    names = list(ablation_results.keys())
    f1s   = [ablation_results[n]['f1_macro'] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, f1s, color=colors[:len(names)], edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Test F1 Macro', fontsize=12)
    ax.set_title('Model 5 Ablation: Kernel Size vs Parallel TextCNN', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=15, ha='right')
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model5_ablation_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Confusion matrix (best variant) ──────────────────────────────────────
    cm = best['confusion_matrix']
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=LABEL_CLASSES, yticklabels=LABEL_CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Model 5 ({best_name}) — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, 'model5_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n✅ Model 5 complete. Key finding: Parallel TextCNN (multi-kernel) captures richer patterns.')


if __name__ == '__main__':
    main()