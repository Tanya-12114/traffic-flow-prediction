"""
kalman.py
---------
Implements the Rauch-Tung-Striebel (RTS) smoother via pykalman.

Difference from plain Kalman filter
────────────────────────────────────
  kf.filter()  → forward pass only  (uses past observations only)
  kf.smooth()  → forward + backward  (uses past AND future observations)

The RTS smoother (kf.smooth) is what the research paper specifies.
It produces a cleaner, more accurate smoothed curve — especially around
sudden traffic spikes — because each point benefits from context on
both sides, not just from what came before it.
"""

from pykalman import KalmanFilter
import numpy as np


class KalmanSmoother:
    """
    RTS (Rauch-Tung-Striebel) smoother wrapping pykalman.

    Usage
    -----
    ks = KalmanSmoother()
    smoothed = ks.smooth(raw_predictions)   # list or 1-D array
    """

    def __init__(
        self,
        transition_var: float = 1e-4,   # how much the signal is expected to change
        observation_var: float = 1e-2,  # how noisy the observations are
    ):
        self.kf = KalmanFilter(
            transition_matrices   = [1],
            observation_matrices  = [1],
            transition_covariance = [[transition_var]],
            observation_covariance= [[observation_var]],
            initial_state_mean    = 0,
            initial_state_covariance = [[1]],
        )

    def smooth(self, predictions) -> np.ndarray:
        """
        Apply RTS smoothing (forward + backward pass).

        Parameters
        ----------
        predictions : list or np.ndarray of shape (T,)

        Returns
        -------
        smoothed : np.ndarray of shape (T,)  — RTS-smoothed values
        """
        preds = np.array(predictions, dtype=float).reshape(-1, 1)

        # kf.smooth() = full RTS (forward Kalman + backward Rauch-Tung-Striebel)
        # This is the algorithm the paper explicitly names.
        smoothed_means, _ = self.kf.smooth(preds)

        return smoothed_means.flatten()

    def filter_only(self, predictions) -> np.ndarray:
        """
        Forward-only Kalman filter (kept for comparison / ablation studies).
        Lower quality than smooth() — included so you can compare both.
        """
        preds = np.array(predictions, dtype=float).reshape(-1, 1)
        filtered_means, _ = self.kf.filter(preds)
        return filtered_means.flatten()