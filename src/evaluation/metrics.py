"""Evaluation metrics for building energy forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score as sklearn_r2

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Root Mean Squared Logarithmic Error.
    
    This is the primary metric for the ASHRAE Great Energy Predictor III competition.
    RMSLE treats relative errors fairly - being off by 10 when actual is 20 is worse
    than being off by 10 when actual is 1000.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid log(0). Default is 1.0 (adds 1 to both values)
        
    Returns:
        RMSLE score
    """
    # Clip negative predictions to 0
    y_pred = np.clip(y_pred, 0, None)
    
    # Calculate RMSLE
    log_true = np.log1p(y_true)  # log(1 + y)
    log_pred = np.log1p(y_pred)
    
    squared_log_error = (log_true - log_pred) ** 2
    mean_squared_log_error = np.mean(squared_log_error)
    rmsle_score = np.sqrt(mean_squared_log_error)
    
    return rmsle_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Simple to interpret: the average absolute difference between predictions and actuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Penalizes large errors more than MAE.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero
        
    Returns:
        MAPE score (as percentage, 0-100)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination).
    
    Measures proportion of variance explained by the model.
    1.0 = perfect predictions, 0.0 = model as good as predicting the mean.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R2 score
    """
    return sklearn_r2(y_true, y_pred)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metrics: Optional[list] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metric names to calculate. If None, calculates all.
        prefix: Prefix to add to metric names (e.g., "train_" or "test_")
        
    Returns:
        Dictionary of metric names to scores
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Default metrics
    if metrics is None:
        metrics = ['rmsle', 'mae', 'rmse', 'mape', 'r2']
    
    results = {}
    
    metric_functions = {
        'rmsle': rmsle,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2_score,
    }
    
    for metric_name in metrics:
        if metric_name not in metric_functions:
            logger.warning(f"Unknown metric: {metric_name}")
            continue
        
        try:
            score = metric_functions[metric_name](y_true, y_pred)
            results[f"{prefix}{metric_name}"] = score
        except Exception as e:
            logger.error(f"Error calculating {metric_name}: {e}")
            results[f"{prefix}{metric_name}"] = np.nan
    
    return results


def evaluate_by_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_col: pd.Series,
    slice_name: str = "slice",
    primary_metric: str = "rmsle"
) -> pd.DataFrame:
    """
    Calculate metrics for each slice (e.g., by building type, season, etc.).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        slice_col: Column defining slices (e.g., building_type)
        slice_name: Name of the slice dimension
        primary_metric: Primary metric to calculate
        
    Returns:
        DataFrame with metrics for each slice
    """
    results = []
    
    for slice_value in slice_col.unique():
        mask = slice_col == slice_value
        
        y_true_slice = y_true[mask]
        y_pred_slice = y_pred[mask]
        
        metrics = evaluate_predictions(y_true_slice, y_pred_slice)
        metrics[slice_name] = slice_value
        metrics['n_samples'] = mask.sum()
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Sort by primary metric
    if primary_metric in results_df.columns:
        results_df = results_df.sort_values(primary_metric)
    
    return results_df


def print_metrics(metrics_dict: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics_dict: Dictionary of metric names to scores
        title: Title to print
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    for metric_name, score in metrics_dict.items():
        if isinstance(score, (int, float)):
            print(f"  {metric_name:20s}: {score:10.4f}")
        else:
            print(f"  {metric_name:20s}: {score}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    """Test metrics with synthetic data."""
    
    # Create synthetic data
    np.random.seed(42)
    y_true = np.random.rand(1000) * 100 + 50  # Values between 50-150
    y_pred = y_true + np.random.randn(1000) * 10  # Add noise
    
    print("Testing evaluation metrics with synthetic data...")
    print(f"y_true range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"y_pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Calculate all metrics
    metrics = evaluate_predictions(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")
    
    # Test with slices
    slice_col = pd.Series(['A'] * 500 + ['B'] * 500)
    slice_metrics = evaluate_by_slice(y_true, y_pred, slice_col, slice_name="group")
    print("\nMetrics by slice:")
    print(slice_metrics)

