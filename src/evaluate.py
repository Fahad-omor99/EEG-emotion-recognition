import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from model import EEGNet

LABEL_NAMES = ['Happy', 'Fear', 'Sad', 'Neutral']


def get_predictions(model, loader, device):
    """Run model on full loader, return all predictions and true labels."""
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch_y.numpy())

    return np.array(all_true), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('EEGNet Emotion Classification\nConfusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(train_history, val_history, save_path='models/training_curves.png'):
    """Plot loss and accuracy curves."""
    train_losses = [h[0] for h in train_history]
    train_accs   = [h[1] for h in train_history]
    val_losses   = [h[0] for h in val_history]
    val_accs     = [h[1] for h in val_history]
    epochs       = range(1, len(train_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs,   'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('EEGNet Training History', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training curves saved to {save_path}")


def full_evaluation(model_path='models/best_model.pt'):
    """Load best model and run full evaluation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("Loading best model...")
    model      = EEGNet(n_classes=4, n_channels=32, n_timepoints=256)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model      = model.to(device)
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.2f}%)")

    # Load test data
    print("\nLoading test data...")
    import os
    if os.path.exists('data/data_preprocessed_python/s01.dat'):
        from dataset import build_dataloaders
        _, _, test_loader = build_dataloaders(batch_size=64)
    else:
        from dataset import EEGEmotionDataset
        from torch.utils.data import DataLoader
        X = np.random.randn(500, 32, 256).astype(np.float32)
        y = np.random.randint(0, 4, 500).astype(np.int64)
        test_loader = DataLoader(EEGEmotionDataset(X, y), batch_size=64)

    # Get predictions
    y_true, y_pred = get_predictions(model, test_loader, device)

    # Print classification report
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    # Per class accuracy
    print("Per-class accuracy:")
    cm = confusion_matrix(y_true, y_pred)
    for i, name in enumerate(LABEL_NAMES):
        acc = cm[i, i] / cm[i].sum() * 100
        print(f"  {name:8s}: {acc:.1f}%")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    return y_true, y_pred


if __name__ == '__main__':
    y_true, y_pred = full_evaluation()