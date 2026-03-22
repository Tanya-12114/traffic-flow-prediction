"""
train_video.py
--------------
Training loop for CNN_BiLSTM_Attention on video data.

Improvements over v1:
  - Huber loss instead of MSE  — less sensitive to outlier frames
  - AdamW + weight decay       — better generalisation
  - Cosine annealing LR        — smoother convergence than ReduceLROnPlateau
  - Early stopping             — stops when val loss stops improving
  - Label smoothing            — prevents overconfidence on boundary values
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.cnn_bilstm import CNN_BiLSTM_Attention


# ── Huber loss with density-aware weighting ───────────────────────────────────
class WeightedHuberLoss(nn.Module):
    """
    Huber loss (smooth L1) that upweights high-density frames.
    High congestion frames matter more for traffic management —
    a 0.05 error at density=0.8 is more critical than at density=0.1.
    """
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss    = self.huber(pred, target)
        # Weight: 1.0 for low density, up to 2.0 for high density
        weights = 1.0 + target.detach()
        return (loss * weights).mean()


def train_video_model(
    dataset,
    epochs:           int   = 30,
    batch_size:       int   = 16,
    lr:               float = 3e-4,      # slightly lower default for AdamW
    val_split:        float = 0.15,      # slightly larger val set
    device:           str   = None,
    progress_callback       = None,
    early_stop_patience: int = 8,        # stop if val loss doesn't improve for N epochs
):
    """
    Train the CNN-BiLSTM model on a VideoTrafficDataset.

    Parameters
    ----------
    dataset              : VideoTrafficDataset instance
    epochs               : max training epochs (early stopping may end sooner)
    batch_size           : mini-batch size
    lr                   : AdamW learning rate
    val_split            : fraction held out for validation
    device               : 'cpu' / 'cuda' / None (auto-detect)
    progress_callback    : called every epoch with (epoch, epochs, train_loss, val_loss)
    early_stop_patience  : stop training if val loss doesn't improve for this many epochs

    Returns
    -------
    model   : best trained CNN_BiLSTM_Attention (lowest val loss checkpoint)
    history : dict with 'train_loss' and 'val_loss'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ── Train / val split ─────────────────────────────────────────────────────
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    _, C, H, W = dataset.input_shape
    model = CNN_BiLSTM_Attention(in_channels=C).to(device)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = WeightedHuberLoss(delta=0.1)

    # AdamW — Adam + weight decay decoupled from gradient, better generalisation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = 1e-4,
    )

    # Cosine annealing — smooth LR decay, avoids the "cliff" of step schedulers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = epochs,
        eta_min = lr * 0.01,   # LR never drops below 1% of start
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history      = {"train_loss": [], "val_loss": []}
    best_val     = float("inf")
    best_weights = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # Validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                v_loss += criterion(model(X).squeeze(-1), y).item()
        v_loss /= max(len(val_loader), 1)

        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        print(f"Epoch {epoch:3d}/{epochs} | Train: {t_loss:.6f} | Val: {v_loss:.6f}")

        if progress_callback:
            progress_callback(epoch, epochs, t_loss, v_loss)

        # ── Checkpoint best model ─────────────────────────────────────────────
        if v_loss < best_val:
            best_val     = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= early_stop_patience:
            print(f"  Early stop at epoch {epoch} — val loss hasn't improved "
                  f"for {early_stop_patience} epochs.")
            break

    # Restore best weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model.cpu(), history