import numpy as np
from scipy.signal import butter, filtfilt

SFREQ = 128          # sampling frequency (Hz)
LOW_FREQ = 4         # bandpass low cut
HIGH_FREQ = 45       # bandpass high cut
WINDOW_SEC = 2       # window size in seconds
WINDOW_SAMPLES = WINDOW_SEC * SFREQ   # 256 samples
STEP_SAMPLES = WINDOW_SAMPLES // 2    # 50% overlap = 128 samples

def bandpass_filter(data, low=LOW_FREQ, high=HIGH_FREQ, fs=SFREQ, order=5):
    """Apply bandpass filter to EEG data.
    data shape: (n_channels, n_timepoints)
    """
    nyq = fs / 2
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, data, axis=-1)


def normalize(data):
    """Normalize each channel to zero mean and unit std.
    data shape: (n_channels, n_timepoints)
    """
    mean = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True)
    std[std == 0] = 1  # avoid division by zero
    return (data - mean) / std


def make_windows(trial, label):
    """Slide a window over one trial, return list of (window, label).
    trial shape: (32, 7680)
    Returns list of windows shape: (32, 256)
    """
    windows, labels = [], []
    start = 0
    while start + WINDOW_SAMPLES <= trial.shape[-1]:
        window = trial[:, start:start + WINDOW_SAMPLES]
        windows.append(window)
        labels.append(label)
        start += STEP_SAMPLES
    return windows, labels


def preprocess_all(all_eeg, all_labels_discrete):
    """Full pipeline: filter → normalize → window all subjects.
    all_eeg shape: (32, 40, 32, 7680)
    all_labels_discrete shape: (32, 40) — integer class labels
    Returns:
        X: (N, 32, 256) — all windows
        y: (N,)         — corresponding labels
    """
    X, y = [], []

    n_subjects, n_trials = all_eeg.shape[0], all_eeg.shape[1]

    for s in range(n_subjects):
        for t in range(n_trials):
            trial = all_eeg[s, t]          # (32, 7680)
            label = all_labels_discrete[s, t]

            # Step 1: bandpass filter
            trial = bandpass_filter(trial)

            # Step 2: normalize
            trial = normalize(trial)

            # Step 3: sliding windows
            windows, lbls = make_windows(trial, label)
            X.extend(windows)
            y.extend(lbls)

        print(f"  Preprocessed subject {s+1:02d}")

    X = np.array(X, dtype=np.float32)  # (N, 32, 256)
    y = np.array(y, dtype=np.int64)    # (N,)

    print(f"\nTotal windows : {X.shape[0]}")
    print(f"Window shape  : {X.shape[1:]}")
    print(f"Labels shape  : {y.shape}")
    print(f"Classes       : {np.unique(y)}")

    return X, y


if __name__ == '__main__':
    # Test with mock data first
    print("Testing preprocessor with mock data...")
    mock_eeg    = np.random.randn(2, 40, 32, 7680).astype(np.float32)
    mock_labels = np.random.randint(0, 4, (2, 40))

    X, y = preprocess_all(mock_eeg, mock_labels)
    print("\nPreprocessor working correctly!")