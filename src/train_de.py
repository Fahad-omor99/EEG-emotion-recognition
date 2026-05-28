# Improvement 1: Training with Differential Entropy features
import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.insert(0, 'src')

from model_de import DEClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def compute_class_weights(y_train, n_classes=4, device='cpu'):
    class_counts  = np.bincount(y_train, minlength=n_classes)
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    class_weights = class_weights / class_weights.sum() * n_classes
    print(f"Class counts  : {class_counts}")
    print(f"Class weights : {class_weights.round(3)}")
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        preds          = out.argmax(dim=1)
        total_loss    += loss.item() * len(y_batch)
        total_correct += (preds == y_batch).sum().item()
        total_samples += len(y_batch)
    return total_loss / total_samples, total_correct / total_samples * 100


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            preds          = out.argmax(dim=1)
            total_loss    += loss.item() * len(y_batch)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)
    return total_loss / total_samples, total_correct / total_samples * 100


def train_de(n_epochs=50, batch_size=64, lr=0.001):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # ── Load and extract DE features ───────────────────────────────
    print("\nLoading DEAP data...")
    if os.path.exists('data/data_preprocessed_python/s01.dat'):
        from load_data import load_all_subjects
        from label_utils import convert_labels
        from features import extract_de_features

        all_eeg, all_labels = load_all_subjects()
        discrete_labels     = convert_labels(all_labels)
        X, y                = extract_de_features(all_eeg, discrete_labels)
    else:
        print("Using mock data...")
        X = np.random.randn(2000, 160).astype(np.float32)
        y = np.random.randint(0, 4, 2000).astype(np.int64)

    print(f"\nFeature shape: {X.shape}")
    print(f"Label shape  : {y.shape}")

    # ── Split ──────────────────────────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.125, stratify=y_tv, random_state=42)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── DataLoaders ────────────────────────────────────────────────
    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # ── Model ──────────────────────────────────────────────────────
    model     = DEClassifier(input_dim=160, n_classes=4).to(device)
    weights   = compute_class_weights(y_train, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5)

    # ── Training Loop ──────────────────────────────────────────────
    print(f"\nTraining for {n_epochs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("-" * 52)

    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader,   criterion, device)
        scheduler.step(vl_acc)

        print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc:>9.2f}% {vl_loss:>10.4f} {vl_acc:>8.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_acc'    : vl_acc,
            }, 'models/best_model_de.pt')
            print(f"         ✓ Best DE model saved (val_acc={vl_acc:.2f}%)")

    # ── Final Test ─────────────────────────────────────────────────
    checkpoint = torch.load('models/best_model_de.pt')
    model.load_state_dict(checkpoint['model_state'])
    _, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*52}")
    print(f"Training complete!")
    print(f"Best val accuracy : {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Test accuracy     : {test_acc:.2f}%")
    print(f"{'='*52}")

    return model


if __name__ == '__main__':
    train_de(n_epochs=100, batch_size=64, lr=0.0005)