"""Explainable Boosting Machine (EBM) Model Wrapper."""

from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class EBMModel:
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize EBM model.
        
        Args:
            params: EBM parameters
        """
        self.default_params = {
            'interactions': 10,
            'learning_rate': 0.01,
            'max_bins': 256,
            'random_state': 42,
            'n_jobs': -1
        }
        self.params = {**self.default_params, **(params or {})}
        self.model = ExplainableBoostingRegressor(**self.params)
        
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: Optional[pd.DataFrame] = None, 
        y_val: Optional[pd.Series] = None
    ):
        """
        Train the model.
        """
        logger.info(f"Training EBM model with {X_train.shape[1]} features...")
        
        # EBM handles validation internally if not provided, but we can't explicitly pass it easily 
        # in the standard fit method like LightGBM without merging.
        # For simplicity, we just fit on train.
        self.model.fit(X_train, y_train)
        
        logger.info("Training complete.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
        
    def explain_global(self, name: str = None):
        """Return global explanation."""
        return self.model.explain_global(name=name)
        
    def explain_local(self, X: pd.DataFrame, y: pd.Series = None, name: str = None):
        """Return local explanation."""
        return self.model.explain_local(X, y, name=name)

    def save(self, path: str):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {path}")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        preds = self.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        
        y_test_safe = np.maximum(y_test, 0)
        preds_safe = np.maximum(preds, 0)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_test_safe), np.log1p(preds_safe)))
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'RMSLE': rmsle
        }
        
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics
