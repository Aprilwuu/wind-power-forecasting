from typing import Union, Iterable
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ArrayLike = Union[Iterable[float], np.ndarray]

def _to_1d_aray(x: ArrayLike) -> np.ndarray:
    """make sure input is transformed to numpy array"""
    return np.asarray(x).reshape(-1)

def evaluate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error (RMSE)

    Parameters
    -----------
    y_true : array-like
    y_pred : array-like

    Returns
    --------
    float
        RMSE value.
    """
    y_true = _to_1d_aray(y_true)
    y_pred = _to_1d_aray(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_mae(y_true: ArrayLike, y_pred:ArrayLike) -> float:
    """
    Mean Absolute Error (MAE)
    """
    y_true = _to_1d_aray(y_true)
    y_pred = _to_1d_aray(y_pred)
    return float(mean_absolute_error(y_true, y_pred))

def evaluate_r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of Determination (R^2)
    """
    y_true = _to_1d_aray(y_true)
    y_pred = _to_1d_aray(y_pred)
    return float(r2_score(y_true, y_pred))

