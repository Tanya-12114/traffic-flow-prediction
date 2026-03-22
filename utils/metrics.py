import numpy as np


def MAE(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def RMSE(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def MAPE(y_true, y_pred, eps: float = 1e-2):
    """
    Mean Absolute Percentage Error — robust to near-zero actuals.

    Traffic density lives in [0, 1]. Values very close to 0 (empty road)
    cause division-by-near-zero which explodes MAPE into millions of percent.

    Fix: only compute MAPE on samples where actual >= eps (default 0.01).
    This ignores frames where the road is essentially empty — a prediction
    error of 0.05 on actual=0.001 is meaningless for traffic management.

    Parameters
    ----------
    eps : minimum actual value to include (default 0.01 = 1% density)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Only evaluate on frames where actual density is meaningful
    mask = np.abs(y_true) >= eps
    if mask.sum() == 0:
        return 0.0   # all actuals near zero — MAPE undefined, return 0

    return float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    )) * 100)