# EEG Emotion Recognition

Real-time emotion classification from EEG brain signals using the DEAP dataset.

## Results
- **65%+ accuracy** across 4 emotion classes (Happy, Fear, Sad, Neutral)
- SVM + Differential Entropy features outperforms raw EEG approach
- Balanced per-class accuracy: Happy 60.7% | Fear 70.9% | Sad 64.4% | Neutral 66.5%

## Architecture
- **Feature extraction**: Differential Entropy across 5 frequency bands (δ,θ,α,β,γ)
- **Classifier**: SVM with RBF kernel + StandardScaler pipeline
- **Baseline**: EEGNet deep learning model (54% accuracy)
- **Dataset**: DEAP — 32 subjects, 40 trials, 32 EEG channels

## Project Structure
- `src/load_data.py` — Load DEAP dataset
- `src/preprocess.py` — Bandpass filter, normalize, sliding windows
- `src/features.py` — Differential Entropy feature extraction
- `src/label_utils.py` — Valence/arousal → emotion labels
- `src/dataset.py` — PyTorch Dataset and DataLoader
- `src/model.py` — EEGNet architecture
- `src/train_svm.py` — SVM training pipeline (best model)
- `src/evaluate.py` — Confusion matrix and metrics

## Setup
```bash
pip install -r requirements.txt
python src/train_svm.py
```

## Tech Stack
Python · PyTorch · scikit-learn · MNE · NumPy · SciPy