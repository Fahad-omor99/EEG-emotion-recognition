# EEG Emotion Recognition

Real-time emotion classification from EEG brain signals using the DEAP dataset.

## Results
- **62% balanced accuracy** across 4 emotion classes
- Happy: 50% | Fear: 62.8% | Sad: 64.2% | Neutral: 61.3%
- SVM + Differential Entropy features outperforms raw EEG baseline (54%)

## Pipeline
Raw EEG → Bandpass Filter → DE Features → SVM → Live Streamlit Dashboard


## Project Structure
```
src/
├── load_data.py       # Phase 1: Load DEAP dataset
├── preprocess.py      # Phase 2: Bandpass filter
├── label_utils.py     # Phase 6: Emotion labels
├── dataset.py         # Phase 5: PyTorch Dataset
├── model.py           # Phase 7: EEGNet model
├── train.py           # Phase 8: Training loop
├── evaluate.py        # Phase 9: Evaluation
├── features.py        # Improvement 1: DE features
├── model_de.py        # Improvement 1: MLP classifier
├── train_de.py        # Improvement 2: Cross-subject
├── train_svm.py       # Best model: SVM 62%
├── realtime_sim.py    # Phase 10+11: Real-time pipeline
└── dashboard.py       # Phase 12: Streamlit dashboard
```

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