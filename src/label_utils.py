import numpy as np

LABEL_NAMES = {0: 'Happy', 1: 'Fear', 2: 'Sad', 3: 'Neutral'}
THRESHOLD = 5.0

def valence_arousal_to_emotion(valence, arousal):
    high_valence = valence >= THRESHOLD
    high_arousal = arousal >= THRESHOLD

    if high_valence and high_arousal:
        return 0  # Happy
    elif not high_valence and high_arousal:
        return 1  # Fear
    elif not high_valence and not high_arousal:
        return 2  # Sad
    else:
        return 3  # Neutral


def convert_labels(all_labels):
    """
    all_labels shape: (32, 40, 4)
    Returns: (32, 40) integer labels 0-3
    """
    n_subjects, n_trials = all_labels.shape[0], all_labels.shape[1]
    discrete = np.zeros((n_subjects, n_trials), dtype=np.int64)

    for s in range(n_subjects):
        for t in range(n_trials):
            valence = all_labels[s, t, 0]
            arousal = all_labels[s, t, 1]
            discrete[s, t] = valence_arousal_to_emotion(valence, arousal)

    return discrete


def print_label_distribution(discrete_labels):
    flat = discrete_labels.flatten()
    total = len(flat)
    print("\nLabel distribution:")
    for label_id, name in LABEL_NAMES.items():
        count = (flat == label_id).sum()
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {name:8s} ({label_id}): {count:4d} trials ({pct:.1f}%) {bar}")
    print()


if __name__ == '__main__':
    print("Testing with mock data...")
    mock_labels = np.random.uniform(1, 9, (32, 40, 4))
    discrete = convert_labels(mock_labels)

    print(f"Input shape  : {mock_labels.shape}")
    print(f"Output shape : {discrete.shape}")
    print(f"Unique labels: {np.unique(discrete)}")
    print_label_distribution(discrete)

    # Test with real DEAP if available
    import os, pickle
    if os.path.exists('data/data_preprocessed_python/s01.dat'):
        print("Testing with real DEAP subject 1...")
        with open('data/data_preprocessed_python/s01.dat', 'rb') as f:
            s = pickle.load(f, encoding='latin1')
        real_labels = s['labels'][np.newaxis, ...]
        discrete_real = convert_labels(real_labels)
        print_label_distribution(discrete_real)