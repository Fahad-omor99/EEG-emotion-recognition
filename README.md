# EEG Emotion Recognition

Real-time emotion classification from EEG brain signals using the DEAP dataset.

## Results
- **62% balanced accuracy** across 4 emotion classes
- Happy: 50% | Fear: 62.8% | Sad: 64.2% | Neutral: 61.3%
- SVM + Differential Entropy features outperforms raw EEG baseline (54%)

## Pipeline
Raw EEG → Bandpass Filter → DE Features → SVM → Live Streamlit Dashboard

## Project Structure
src/
├── load_data.py      # Phase 1: Load DEAP dataset (32 subjects)
├── preprocess.py     # Phase 2: Bandpass filter, normalize, windowing
├── label_utils.py    # Phase 6: Valence/arousal → emotion labels
├── dataset.py        # Phase 5: PyTorch Dataset and DataLoader
├── model.py          # Phase 7: EEGNet deep learning model
├── train.py          # Phase 8: EEGNet training loop
├── evaluate.py       # Phase 9: Confusion matrix and metrics
├── features.py       # Improvement 1: Differential Entropy extraction
├── model_de.py       # Improvement 1: MLP classifier for DE features
├── train_de.py       # Improvement 2: Cross-subject validation
├── train_svm.py      # Best model: SVM + DE features (62% accuracy)
├── realtime_sim.py   # Phase 10+11: Real-time EEG simulation pipeline
└── dashboard.py      # Phase 12: Live Streamlit emotion dashboard

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the SVM model
python src/train_svm.py

# Launch live dashboard
streamlit run src/dashboard.py
```

## Model Comparison
| Model | Accuracy | Notes |
|-------|----------|-------|
| EEGNet (raw EEG) | 54% | Deep learning baseline |
| MLP + DE features | 57% | Faster convergence |
| **SVM + DE features** | **62%** | **Best — balanced per class** |

## Dataset
DEAP dataset — 32 subjects, 40 trials each, 32 EEG channels, 128Hz sampling rate.
Download from: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

## Tech Stack
Python · PyTorch · scikit-learn · MNE · SciPy · NumPy · Streamlit

## Author
Fahad Bin Omor — github.com/Fahad-omor99