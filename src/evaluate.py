"""
Evaluation metrics for Food Delivery ETA ML System.
Implements standard metrics (MAE, RMSE, R2, MAPE) plus production domain-specific metrics:
- Asymmetric Pinball Loss (penalizing late deliveries 3x compared to early deliveries)
- On-Time Delivery Rate (orders delivered within +5 mins of predicted ETA)
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import ASYMMETRIC_TAU

def asymmetric_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float = ASYMMETRIC_TAU) -> float:
    """
    Quantile / Pinball loss function:
    L_tau(y, y_hat) = max(tau * (y - y_hat), (1 - tau) * (y_hat - y))
    """
    residual = y_true - y_pred
    loss = np.maximum(tau * residual, (1.0 - tau) * -residual)
    return float(np.mean(loss))

def calculate_eta_metrics(y_true: np.ndarray, y_pred: np.ndarray, tau: float = ASYMMETRIC_TAU) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics for delivery predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid zero division
    safe_y_true = np.where(y_true == 0, 1e-5, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100.0)
    
    pinball = asymmetric_pinball_loss(y_true, y_pred, tau=tau)
    
    # Domain specific SLA metrics:
    # On-Time buffer: arriving between -3 mins early to +5 mins late of prediction
    diff = y_true - y_pred  # positive means delivered later than predicted
    on_time_mask = (diff <= 5.0) & (diff >= -5.0)
    on_time_rate = float(np.mean(on_time_mask) * 100.0)
    
    # Tail late rate: delivered > 5 mins later than promised
    severe_late_rate = float(np.mean(diff > 5.0) * 100.0)

    return {
        "mae_min": round(mae, 4),
        "rmse_min": round(rmse, 4),
        "r2_score": round(r2, 4),
        "mape_percent": round(mape, 2),
        "asymmetric_pinball_loss": round(pinball, 4),
        "on_time_rate_percent": round(on_time_rate, 2),
        "severe_late_rate_percent": round(severe_late_rate, 2)
    }
