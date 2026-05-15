import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.insert(0, 'src')

from model import EEGNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over training data."""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss    = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        preds = outputs.argmax(dim=1)
        total_loss    += loss.item() * len(batch_y)
        total_correct += (preds == batch_y).sum().item()
        total_samples += len(batch_y)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on val or test set."""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss    = criterion(outputs, batch_y)

            preds = outputs.argmax(dim=1)
            total_loss    += loss.item() * len(batch_y)
            total_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


def train(
    n_epochs=30,
    batch_size=64,
    learning_rate=0.001,
    dropout=0.5,
    save_path='models/best_model.pt'
):
    """Full training pipeline."""

    # ── Device ─────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # ── Data ───────────────────────────────────────────────────────
    print("\nLoading data...")
    import os
    if os.path.exists('data/data_preprocessed_python/s01.dat'):
        from dataset import build_dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            batch_size=batch_size
        )
    else:
        # Mock data for testing
        print("No DEAP data found — using mock data")
        from dataset import EEGEmotionDataset
        from torch.utils.data import DataLoader
        from sklearn.model_selection import train_test_split

        X = np.random.randn(2000, 32, 256).astype(np.float32)
        y = np.random.randint(0, 4, 2000).astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.1)

        train_loader = DataLoader(EEGEmotionDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(EEGEmotionDataset(X_val,   y_val),   batch_size=batch_size)
        test_loader  = DataLoader(EEGEmotionDataset(X_test,  y_test),  batch_size=batch_size)

    # ── Model ──────────────────────────────────────────────────────
    model = EEGNet(n_classes=4, n_channels=32, n_timepoints=256, dropout=dropout)
    model = model.to(device)

    # ── Loss and Optimizer ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── Training Loop ──────────────────────────────────────────────
    print(f"\nTraining for {n_epochs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 62)

    best_val_acc  = 0.0
    best_epoch    = 0
    train_history = []
    val_history   = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_history.append((train_loss, train_acc))
        val_history.append((val_loss, val_acc))

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>10.4f} {val_acc:>8.2f}% {current_lr:>10.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc'   : val_acc,
            }, save_path)
            print(f"         ✓ Best model saved (val_acc={val_acc:.2f}%)")

    # ── Final Evaluation ───────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"Training complete!")
    print(f"Best val accuracy : {best_val_acc:.2f}% at epoch {best_epoch}")

    # Load best model and test
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy     : {test_acc:.2f}%")
    print(f"{'='*62}")

    return model, train_history, val_history


if __name__ == '__main__':
    model, train_hist, val_hist = train(
        n_epochs=30,
        batch_size=64,
        learning_rate=0.001
    )