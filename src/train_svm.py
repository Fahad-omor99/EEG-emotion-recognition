# Improvement: SVM classifier on DE features - targets 65%+ accuracy
import numpy as np
import os
import sys
sys.path.insert(0, 'src')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


LABEL_NAMES = ['Happy', 'Fear', 'Sad', 'Neutral']


def load_de_features():
    """Load or extract DE features from DEAP."""

    # Cache features to disk so we don't recompute every time
    if os.path.exists('models/de_features.npy'):
        print("Loading cached DE features...")
        X = np.load('models/de_features.npy')
        y = np.load('models/de_labels.npy')
        print(f"Loaded: X={X.shape}, y={y.shape}")
        return X, y

    print("Extracting DE features from DEAP...")
    from load_data import load_all_subjects
    from label_utils import convert_labels
    from features import extract_de_features

    all_eeg, all_labels = load_all_subjects()
    discrete_labels     = convert_labels(all_labels)
    X, y                = extract_de_features(all_eeg, discrete_labels)

    # Cache to disk
    os.makedirs('models', exist_ok=True)
    np.save('models/de_features.npy', X)
    np.save('models/de_labels.npy',   y)
    print("Features cached to models/de_features.npy")

    return X, y


def train_svm():
    """Train SVM on DE features."""

    # ── Load features ───────────────────────────────────────────
    X, y = load_de_features()
    print(f"\nFeature shape : {X.shape}")
    print(f"Label shape   : {y.shape}")

    # ── Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ── SVM Pipeline ────────────────────────────────────────────
    # StandardScaler normalises features — critical for SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(
            kernel='rbf',      # RBF kernel best for EEG
            C=10,              # regularisation
            gamma='scale',     # auto-scale kernel
            class_weight='balanced',  # handles class imbalance
            random_state=42
        ))
    ])

    # ── Train ───────────────────────────────────────────────────
    print("\nTraining SVM (this takes 2-5 minutes)...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # ── Evaluate ────────────────────────────────────────────────
    y_pred    = pipeline.predict(X_test)
    test_acc  = (y_pred == y_test).mean() * 100

    print(f"\n{'='*50}")
    print(f"Test accuracy : {test_acc:.2f}%")
    print(f"{'='*50}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Per class accuracy
    print("Per-class accuracy:")
    cm = confusion_matrix(y_test, y_pred)
    for i, name in enumerate(LABEL_NAMES):
        acc = cm[i, i] / cm[i].sum() * 100
        print(f"  {name:8s}: {acc:.1f}%")

    # ── Plot confusion matrix ────────────────────────────────────
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('SVM + DE Features\nConfusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_svm.png', dpi=150)
    plt.show()
    print("\nConfusion matrix saved to models/confusion_matrix_svm.png")

    # ── Save model ───────────────────────────────────────────────
    joblib.dump(pipeline, 'models/svm_model.pkl')
    print("SVM model saved to models/svm_model.pkl")

    return pipeline, test_acc


if __name__ == '__main__':
    pipeline, acc = train_svm()