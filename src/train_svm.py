import numpy as np
import os
import sys
sys.path.insert(0, 'src')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

LABEL_NAMES = ['Happy', 'Fear', 'Sad', 'Neutral']


def load_de_features():
    if os.path.exists('models/de_features.npy'):
        print("Loading cached DE features...")
        X = np.load('models/de_features.npy')
        y = np.load('models/de_labels.npy')
        print(f"Loaded: X={X.shape}, y={y.shape}")
        return X, y

    print("Extracting DE features...")
    from load_data import load_all_subjects
    from label_utils import convert_labels
    from features import extract_de_features

    all_eeg, all_labels = load_all_subjects()
    discrete_labels     = convert_labels(all_labels)
    X, y                = extract_de_features(all_eeg, discrete_labels)

    os.makedirs('models', exist_ok=True)
    np.save('models/de_features.npy', X)
    np.save('models/de_labels.npy',   y)
    print("Features cached.")
    return X, y


def train_svm():
    X, y = load_de_features()

    # Use 20,000 samples — enough for good accuracy, trains in 5-10 mins
    from sklearn.utils import resample
    X, y = resample(X, y, n_samples=20000, stratify=y, random_state=42)
    print(f"Using {len(X)} samples for training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])

    print("\nTraining SVM (5-10 minutes)...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    y_pred   = pipeline.predict(X_test)
    test_acc = (y_pred == y_test).mean() * 100

    print(f"\n{'='*50}")
    print(f"Test accuracy : {test_acc:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    print("Per-class accuracy:")
    cm = confusion_matrix(y_test, y_pred)
    for i, name in enumerate(LABEL_NAMES):
        acc = cm[i, i] / cm[i].sum() * 100
        print(f"  {name:8s}: {acc:.1f}%")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('SVM + DE Features\nConfusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_svm.png', dpi=150)
    plt.show()

    joblib.dump(pipeline, 'models/svm_model.pkl')
    print("Model saved to models/svm_model.pkl")
    return pipeline


if __name__ == '__main__':
    train_svm()