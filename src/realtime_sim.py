# Phase 10: Real-time EEG simulation pipeline
import numpy as np
import time
import sys
import os
sys.path.insert(0, 'src')

from preprocess import bandpass_filter, normalize, WINDOW_SAMPLES, STEP_SAMPLES
from features import compute_de
from label_utils import LABEL_NAMES

SFREQ        = 128   # samples per second
CHUNK_SIZE   = 64    # samples per chunk (0.5 seconds)
BUFFER_SIZE  = WINDOW_SAMPLES  # 256 samples = 2 seconds


class EEGBuffer:
    """Ring buffer that holds 2 seconds of EEG data."""

    def __init__(self, n_channels=32, buffer_size=BUFFER_SIZE):
        self.n_channels  = n_channels
        self.buffer_size = buffer_size
        self.buffer      = np.zeros((n_channels, buffer_size))
        self.is_ready    = False
        self.n_received  = 0

    def add_chunk(self, chunk):
        """Add new chunk of EEG data to buffer.
        chunk shape: (n_channels, chunk_size)
        """
        chunk_len = chunk.shape[1]
        # Shifting buffer left and add new data at the end
        self.buffer = np.roll(self.buffer, -chunk_len, axis=1)
        self.buffer[:, -chunk_len:] = chunk
        self.n_received += chunk_len

        # Buffer is ready after it fills up once
        if self.n_received >= self.buffer_size:
            self.is_ready = True

    def get_window(self):
        """Return current buffer as a window."""
        return self.buffer.copy()


class RealTimeSimulator:
    """Simulates real-time EEG stream from DEAP data."""

    def __init__(self, model, subject_id=0, trial_id=0):
        self.model      = model
        self.buffer     = EEGBuffer()
        self.subject_id = subject_id
        self.trial_id   = trial_id
        self.eeg_data   = None
        self.true_label = None
        self.position   = 0

    def load_trial(self, all_eeg, all_labels):
        """Load one trial from DEAP data."""
        self.eeg_data   = all_eeg[self.subject_id, self.trial_id]    # (32, 7680)
        self.true_label = all_labels[self.subject_id, self.trial_id]
        self.position   = 0
        print(f"Loaded: Subject {self.subject_id+1}, "
              f"Trial {self.trial_id+1}, "
              f"True emotion: {LABEL_NAMES[self.true_label]}")

    def get_next_chunk(self):
        """Get next chunk of EEG data (simulates live stream)."""
        if self.position + CHUNK_SIZE > self.eeg_data.shape[1]:
            return None  # End of trial

        chunk = self.eeg_data[:, self.position:self.position + CHUNK_SIZE]
        self.position += CHUNK_SIZE
        return chunk

    def predict(self, window):
        """Preprocess window and predict emotion."""
        # Step 1: bandpassing filter
        filtered = bandpass_filter(window)

        # Step 2: normalizing
        normalized = normalize(filtered)

        # Step 3: extracting DE features
        de_features = compute_de(normalized)  # (160,)

        # Step 4: predict
        de_features_2d = de_features.reshape(1, -1)  # (1, 160)
        prediction      = self.model.predict(de_features_2d)[0]
        probabilities   = self.model.predict_proba(de_features_2d)[0]

        return prediction, probabilities

    def run(self, all_eeg, all_labels, speed=1.0, callback=None):
        """Run simulation loop.
        
        speed   : 1.0 = real-time, 2.0 = 2x speed
        callback: function called with (prediction, probs, true_label) each step
        """
        self.load_trial(all_eeg, all_labels)
        predictions = []
        step        = 0

        print(f"\nStarting real-time simulation...")
        print(f"{'Step':>5} {'Predicted':>12} {'Confidence':>12} {'True':>10}")
        print("-" * 45)

        while True:
            chunk = self.get_next_chunk()
            if chunk is None:
                break

            # Adding to buffer
            self.buffer.add_chunk(chunk)

            # Only predict when buffer is full
            if self.buffer.is_ready:
                window                 = self.buffer.get_window()
                prediction, probs      = self.predict(window)
                confidence             = probs.max() * 100
                predicted_name         = LABEL_NAMES[prediction]
                true_name              = LABEL_NAMES[self.true_label]

                predictions.append(prediction)
                step += 1

                print(f"{step:>5} {predicted_name:>12} {confidence:>11.1f}% {true_name:>10}")

                if callback:
                    callback(prediction, probs, self.true_label, step)

            # Simulating real-time delay
            time.sleep(CHUNK_SIZE / SFREQ / speed)

        # Summary
        if predictions:
            from collections import Counter
            most_common = Counter(predictions).most_common(1)[0][0]
            print(f"\n{'='*45}")
            print(f"Trial summary:")
            print(f"  Most predicted : {LABEL_NAMES[most_common]}")
            print(f"  True emotion   : {LABEL_NAMES[self.true_label]}")
            print(f"  Match          : {'✓ YES' if most_common == self.true_label else '✗ NO'}")
            print(f"{'='*45}")

        return predictions


if __name__ == '__main__':
    import joblib

    print("Loading SVM model...")
    model = joblib.load('models/svm_model.pkl')

    # Enabling probability estimates — needed for confidence scores
    # Checking if model supports predict_proba
    if not hasattr(model.named_steps['svm'], 'probability') or \
       not model.named_steps['svm'].probability:
        print("Note: Retraining SVM with probability=True for confidence scores...")
        from train_svm import train_svm
        model = train_svm()
        import joblib
        joblib.dump(model, 'models/svm_model.pkl')

    print("Loading DEAP data...")
    from load_data import load_all_subjects
    from label_utils import convert_labels

    all_eeg, all_labels  = load_all_subjects()
    discrete_labels      = convert_labels(all_labels)

    # Run simulation on subject 1, trial 1
    simulator = RealTimeSimulator(model, subject_id=0, trial_id=0)
    simulator.run(all_eeg, discrete_labels, speed=5.0)  