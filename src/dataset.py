import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from load_data import load_all_subjects
from preprocess import preprocess_all
from label_utils import convert_labels, print_label_distribution


class EEGEmotionDataset(Dataset):
    """PyTorch Dataset for EEG emotion windows."""

    def __init__(self, X, y):
        """
        X shape: (N, 32, 256) — EEG windows
        y shape: (N,)         — emotion labels 0-3
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(batch_size=64, test_size=0.2, val_size=0.1, random_state=42):
    """Full pipeline: load → label → preprocess → split → DataLoader.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 50)
    print("Step 1: Loading DEAP data...")
    print("=" * 50)
    all_eeg, all_labels = load_all_subjects()

    print("\n" + "=" * 50)
    print("Step 2: Converting labels...")
    print("=" * 50)
    discrete_labels = convert_labels(all_labels)
    print_label_distribution(discrete_labels)

    print("=" * 50)
    print("Step 3: Preprocessing EEG...")
    print("=" * 50)
    X, y = preprocess_all(all_eeg, discrete_labels)

    print("\n" + "=" * 50)
    print("Step 4: Splitting data...")
    print("=" * 50)

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # keep class balance in each split
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_trainval
    )

    print(f"  Train : {X_train.shape[0]:6d} windows")
    print(f"  Val   : {X_val.shape[0]:6d} windows")
    print(f"  Test  : {X_test.shape[0]:6d} windows")

    print("\n" + "=" * 50)
    print("Step 5: Creating DataLoaders...")
    print("=" * 50)

    train_dataset = EEGEmotionDataset(X_train, y_train)
    val_dataset   = EEGEmotionDataset(X_val,   y_val)
    test_dataset  = EEGEmotionDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"  Batch size     : {batch_size}")
    print(f"  Train batches  : {len(train_loader)}")
    print(f"  Val batches    : {len(val_loader)}")
    print(f"  Test batches   : {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, 'src')

    # Use mock data if real data not available
    if not os.path.exists('data/data_preprocessed_python/s01.dat'):
        print("Using mock data for testing...")
        X = np.random.randn(1000, 32, 256).astype(np.float32)
        y = np.random.randint(0, 4, 1000).astype(np.int64)
    else:
        print("Real DEAP data found — running full pipeline...")
        from load_data import load_all_subjects
        from preprocess import preprocess_all
        from label_utils import convert_labels

        all_eeg, all_labels = load_all_subjects()
        discrete_labels     = convert_labels(all_labels)
        X, y                = preprocess_all(all_eeg, discrete_labels)

    # Test Dataset
    dataset = EEGEmotionDataset(X, y)
    print(f"\nDataset size   : {len(dataset)}")
    print(f"Sample X shape : {dataset[0][0].shape}")
    print(f"Sample y       : {dataset[0][1]}")

    # Test DataLoader
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    train_ds = EEGEmotionDataset(X_train, y_train)
    loader   = DataLoader(train_ds, batch_size=64, shuffle=True)

    batch_X, batch_y = next(iter(loader))
    print(f"\nBatch X shape  : {batch_X.shape}")   # (64, 32, 256)
    print(f"Batch y shape  : {batch_y.shape}")   # (64,)
    print(f"Batch y sample : {batch_y[:8]}")
    print("\nDataset and DataLoader working correctly!")