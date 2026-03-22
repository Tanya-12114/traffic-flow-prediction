"""
online_learner.py  —  Gap 3: Real-time adaptive retraining
----------------------------------------------------------
Continuously fine-tunes the model on the most recent N clips
without full retraining.  Uses a fixed-size replay buffer so
the model never forgets older patterns while still adapting
to new traffic conditions (construction, seasonal changes, etc.)

How it works
────────────
  1. After each prediction, new (X, y) pairs are added to a ring buffer
  2. Every `update_every` new samples, run a mini fine-tune step
  3. Learning rate is small (lr_finetune << lr_train) to avoid forgetting
  4. Old samples are evicted from the buffer (FIFO)

This implements a simplified version of online / continual learning.
"""

import torch
import torch.nn as nn
from collections import deque
import numpy as np
from typing import Optional


class OnlineLearner:
    """
    Wraps a trained CNN_BiLSTM_Attention model and fine-tunes it
    incrementally as new video frames arrive.

    Parameters
    ----------
    model         : trained CNN_BiLSTM_Attention instance
    buffer_size   : max samples kept in the replay buffer
    update_every  : fine-tune after every N new samples
    lr_finetune   : learning rate for online updates (keep small: 1e-5 to 1e-4)
    device        : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model,
        buffer_size:  int   = 200,
        update_every: int   = 20,
        lr_finetune:  float = 1e-5,
        device:       str   = None,
    ):
        self.device       = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model        = model.to(self.device)
        self.buffer_size  = buffer_size
        self.update_every = update_every

        # Replay buffer: stores (X_tensor, y_scalar) tuples
        self.buffer: deque = deque(maxlen=buffer_size)

        self.optimizer    = torch.optim.Adam(
            self.model.parameters(), lr=lr_finetune
        )
        self.criterion    = nn.MSELoss()
        self._since_update = 0
        self.update_count  = 0
        self.loss_history  = []

    def add_sample(self, X: torch.Tensor, y: float):
        """
        Add one new (clip, label) pair to the buffer.
        Triggers a fine-tune step every `update_every` samples.

        Parameters
        ----------
        X : (clip_len, C, H, W) tensor — one video clip
        y : float — ground truth density for next frame
        """
        self.buffer.append((X.cpu(), torch.tensor(y, dtype=torch.float32)))
        self._since_update += 1

        if self._since_update >= self.update_every:
            self._finetune_step()
            self._since_update = 0

    def add_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Add a batch of samples at once.
        X_batch : (N, clip_len, C, H, W)
        y_batch : (N,)
        """
        for i in range(len(X_batch)):
            self.add_sample(X_batch[i], y_batch[i].item())

    def _finetune_step(self, n_steps: int = 3, batch_size: int = 16):
        """
        Run a few gradient steps on a random mini-batch from the buffer.
        """
        if len(self.buffer) < batch_size:
            return

        self.model.train()
        total_loss = 0.0

        for _ in range(n_steps):
            # Sample random mini-batch from buffer
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            X_list  = [self.buffer[i][0] for i in indices]
            y_list  = [self.buffer[i][1] for i in indices]

            X_batch = torch.stack(X_list).to(self.device)
            y_batch = torch.stack(y_list).to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X_batch).squeeze(-1)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_steps
        self.loss_history.append(avg_loss)
        self.update_count += 1
        print(f"[OnlineLearner] Update #{self.update_count} | "
              f"buffer={len(self.buffer)} | loss={avg_loss:.6f}")

        self.model.eval()

    def predict(self, X: torch.Tensor) -> float:
        """
        Run inference with the current (continuously updated) model.

        Parameters
        ----------
        X : (clip_len, C, H, W) or (1, clip_len, C, H, W)

        Returns
        -------
        density : float prediction
        """
        self.model.eval()
        with torch.no_grad():
            if X.dim() == 4:
                X = X.unsqueeze(0)
            X   = X.to(self.device)
            out = self.model(X)
            return out.item()

    def get_stats(self) -> dict:
        """Return current buffer and update statistics."""
        return {
            "buffer_size":    len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "update_count":   self.update_count,
            "recent_loss":    self.loss_history[-1] if self.loss_history else None,
            "loss_trend":     "improving" if (
                len(self.loss_history) >= 5 and
                self.loss_history[-1] < self.loss_history[-5]
            ) else "stable",
        }

    def save_checkpoint(self, path: str):
        """Save the current model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"[OnlineLearner] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load weights back into model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[OnlineLearner] Checkpoint loaded from {path}")


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.cnn_bilstm import CNN_BiLSTM_Attention

    model   = CNN_BiLSTM_Attention(in_channels=2)
    learner = OnlineLearner(model, buffer_size=100, update_every=10)

    # Simulate 50 incoming samples
    for i in range(50):
        X = torch.randn(12, 2, 64, 64)
        y = float(np.random.uniform(0, 1))
        learner.add_sample(X, y)

    print(learner.get_stats())