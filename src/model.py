import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet: A compact CNN for EEG-based BCIs.
    Designed specifically for EEG signals.
    
    Input shape : (batch, 1, n_channels, n_timepoints)
                = (batch, 1, 32, 256)
    Output shape: (batch, n_classes)
                = (batch, 4)
    """

    def __init__(
        self,
        n_classes=4,
        n_channels=32,
        n_timepoints=256,
        dropout=0.5,
        F1=8,       # number of temporal filters
        D=2,        # depth multiplier (spatial filters per temporal)
        F2=16,      # number of pointwise filters (= F1 * D)
    ):
        super(EEGNet, self).__init__()

        # ── Block 1: Temporal Convolution ──────────────────────────
        # Learns frequency-specific features across time
        self.block1 = nn.Sequential(
            # Temporal conv: kernel spans 0.5 seconds (128/2 = 64)
            nn.Conv2d(1, F1, kernel_size=(1, n_timepoints // 2),
                      padding=(0, n_timepoints // 4), bias=False),
            nn.BatchNorm2d(F1),

            # Depthwise conv: learns spatial filter per channel
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )

        # ── Block 2: Separable Convolution ─────────────────────────
        # Learns temporal relationships between spatial features
        self.block2 = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise conv
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )

        # ── Classifier ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_flatten_size(n_channels, n_timepoints, F1, D, F2), n_classes)
        )

    def _get_flatten_size(self, n_channels, n_timepoints, F1, D, F2):
        """Compute flattened size dynamically by doing a dummy forward pass."""
        dummy = torch.zeros(1, 1, n_channels, n_timepoints)
        x = self.block1(dummy)
        x = self.block2(x)
        return x.numel()

    def forward(self, x):
        """
        x shape: (batch, n_channels, n_timepoints) — from DataLoader
        """
        x = x.unsqueeze(1)   # → (batch, 1, n_channels, n_timepoints)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    print("Testing EEGNet...")

    model = EEGNet(
        n_classes=4,
        n_channels=32,
        n_timepoints=256,
        dropout=0.5
    )

    # Print model summary
    print(model)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test forward pass
    batch = torch.randn(64, 32, 256)   # (batch, channels, timepoints)
    out   = model(batch)
    print(f"\nInput shape  : {batch.shape}")
    print(f"Output shape : {out.shape}")  # should be (64, 4)

    # Test with device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device : {device}")
    model  = model.to(device)
    batch  = batch.to(device)
    out    = model(batch)
    print(f"Forward pass on {device} successful!")
    print(f"Output sample: {out[0].detach().cpu().numpy().round(3)}")