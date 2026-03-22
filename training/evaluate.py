"""
evaluate.py
-----------
Standalone evaluation helper — used by main.py CLI.
"""

import numpy as np
import torch
from utils.metrics import MAE, RMSE, MAPE
from models.kalman import KalmanSmoother


def evaluate_model(model, dataset):
    """Run inference on every sample and return MAE / RMSE / MAPE."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataset:
            preds.append(model(X.unsqueeze(0)).item())
            trues.append(y.item())

    smoothed = KalmanSmoother().smooth(preds)
    return {
        "MAE":  MAE(np.array(trues), smoothed),
        "RMSE": RMSE(np.array(trues), smoothed),
        "MAPE": MAPE(np.array(trues), smoothed),
    }