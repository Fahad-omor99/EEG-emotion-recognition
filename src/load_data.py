import pickle
import numpy as np
import os

DATA_PATH = 'data/data_preprocessed_python/'

def load_subject(subject_id):
    filepath = os.path.join(DATA_PATH, f's{subject_id:02d}.dat')
    with open(filepath, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')
    
    eeg    = subject['data'][:, :32, 384:]  # (40, 32, 7680)
    labels = subject['labels']              # (40, 4)
    return eeg, labels

def load_all_subjects():
    all_eeg, all_labels = [], []
    for s in range(1, 33):
        eeg, labels = load_subject(s)
        all_eeg.append(eeg)
        all_labels.append(labels)
        print(f"  Loaded subject {s:02d} — EEG: {eeg.shape}")
    return np.array(all_eeg), np.array(all_labels)

if __name__ == '__main__':
    print("Loading DEAP dataset...")
    all_eeg, all_labels = load_all_subjects()
    print("\n--- Done ---")
    print(f"EEG shape    : {all_eeg.shape}")    # (32, 40, 32, 7680)
    print(f"Labels shape : {all_labels.shape}") # (32, 40, 4)
    print(f"EEG range    : {all_eeg.min():.1f} to {all_eeg.max():.1f}")