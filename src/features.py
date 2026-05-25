# Improvement 1: Differential Entropy feature extraction
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import differential_entropy

SFREQ = 128

# Frequency bands
BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta' : (13, 30),
    'gamma': (30, 45),
}


def bandpass(data, low, high, fs=SFREQ, order=5):
    """Bandpass filter for one frequency band."""
    nyq  = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_de(window):
    """Compute Differential Entropy for one window.
    
    window shape : (n_channels, n_timepoints) e.g. (32, 256)
    Returns      : (n_channels * n_bands,) e.g. (32*5,) = (160,)
    """
    n_channels = window.shape[0]
    n_bands    = len(BANDS)
    de_features = np.zeros((n_channels, n_bands))

    for b_idx, (band_name, (low, high)) in enumerate(BANDS.items()):
        filtered = bandpass(window, low, high)          # (32, 256)
        for ch in range(n_channels):
            # DE of a Gaussian signal = 0.5 * log(2*pi*e*variance)
            var = np.var(filtered[ch])
            de_features[ch, b_idx] = 0.5 * np.log(2 * np.pi * np.e * (var + 1e-6))

    return de_features.flatten()   # (160,)


def extract_de_features(all_eeg, all_labels_discrete):
    """Extract DE features for all subjects and trials with sliding windows.
    
    all_eeg    : (32, 40, 32, 7680)
    all_labels : (32, 40)
    Returns    : X (N, 160), y (N,)
    """
    from preprocess import WINDOW_SAMPLES, STEP_SAMPLES, normalize, bandpass_filter

    X, y = [], []
    n_subjects, n_trials = all_eeg.shape[0], all_eeg.shape[1]

    for s in range(n_subjects):
        for t in range(n_trials):
            trial = all_eeg[s, t]             # (32, 7680)
            label = all_labels_discrete[s, t]

            # Normalize first
            trial = normalize(trial)

            # Sliding window
            start = 0
            while start + WINDOW_SAMPLES <= trial.shape[-1]:
                window = trial[:, start:start + WINDOW_SAMPLES]
                de     = compute_de(window)    # (160,)
                X.append(de)
                y.append(label)
                start += STEP_SAMPLES

        print(f"  Extracted DE features: subject {s+1:02d}")

    X = np.array(X, dtype=np.float32)   # (N, 160)
    y = np.array(y, dtype=np.int64)     # (N,)

    print(f"\nDE feature shape : {X.shape}")
    print(f"Label shape      : {y.shape}")
    return X, y


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')

    print("Testing DE feature extraction with mock data...")
    mock_eeg    = np.random.randn(2, 40, 32, 7680).astype(np.float32)
    mock_labels = np.random.randint(0, 4, (2, 40))

    X, y = extract_de_features(mock_eeg, mock_labels)
    print(f"\nSample features  : {X[0].round(3)}")
    print(f"Feature range    : {X.min():.2f} to {X.max():.2f}")
    print("\nDE extraction working correctly!")