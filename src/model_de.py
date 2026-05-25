# Improvement 1: MLP classifier for Differential Entropy features
import torch
import torch.nn as nn


class DEClassifier(nn.Module):
    """MLP classifier for Differential Entropy EEG features.
    
    Input : (batch, 160) — DE features (32 channels x 5 bands)
    Output: (batch, 4)   — emotion class logits
    """

    def __init__(self, input_dim=160, n_classes=4, dropout=0.5):
        super(DEClassifier, self).__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropout),

            # Output
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    print("Testing DEClassifier...")
    model = DEClassifier(input_dim=160, n_classes=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")

    # Test forward pass
    batch = torch.randn(64, 160)
    out   = model(batch)
    print(f"Input shape      : {batch.shape}")
    print(f"Output shape     : {out.shape}")
    print(f"Output sample    : {out[0].detach().numpy().round(3)}")
    print("\nDEClassifier working correctly!")