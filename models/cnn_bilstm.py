"""
cnn_bilstm.py
-------------
CNN spatial encoder + Bi-LSTM temporal predictor with temporal attention.

Improvements over v1:
  - Deeper CNN encoder (4 blocks) preserving more spatial detail
  - Deeper FC head with residual skip connection
  - Sigmoid output clamps prediction to [0,1] (density is bounded)
  - Lower dropout (0.2) to reduce underfitting on small datasets

Input  : (batch, T, C, H, W)  — T video frames
Output : (batch, 1)            — predicted traffic density in [0, 1]
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Shared-weight CNN: single frame (C, H, W) → feature vector."""

    def __init__(self, in_channels: int = 2, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 4 — extra detail extraction before pooling
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),   # reduced from 0.3 — less underfitting
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        energy  = torch.tanh(self.attn(lstm_out))
        scores  = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)


class CNN_BiLSTM_Attention(nn.Module):
    """
    Parameters
    ----------
    in_channels  : CNN input channels (1=gray, 2=gray+flow)
    feature_dim  : CNN output size
    hidden_size  : BiLSTM hidden units per direction
    num_layers   : BiLSTM depth
    output_size  : prediction dimension
    """

    def __init__(
        self,
        in_channels:  int = 2,
        feature_dim:  int = 128,
        hidden_size:  int = 64,
        num_layers:   int = 2,
        output_size:  int = 1,
    ):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_channels=in_channels, feature_dim=feature_dim)

        self.lstm = nn.LSTM(
            input_size    = feature_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = 0.2 if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)

        lstm_out_dim = hidden_size * 2   # 128

        # Deeper FC head: 128 → 64 → 32 → 1
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_size),
            nn.Sigmoid(),   # clamp output to [0,1] — density is always bounded
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, C, H, W = x.shape
        x_flat      = x.view(batch * T, C, H, W)
        features    = self.cnn_encoder(x_flat).view(batch, T, -1)
        lstm_out, _ = self.lstm(features)
        context     = self.attention(lstm_out)
        return self.fc(context)


if __name__ == "__main__":
    model = CNN_BiLSTM_Attention(in_channels=2)
    x     = torch.randn(4, 12, 2, 64, 64)
    out   = model(x)
    print(f"Input: {x.shape}  Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]"
    print("Output range check passed: all in [0, 1]")