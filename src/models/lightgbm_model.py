"""LightGBM Model Wrapper."""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class LightGBMModel:
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LightGBM model.
        
        Args:
            params: LightGBM parameters
        """
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        self.params = {**self.default_params, **(params or {})}
        self.model = None
        self.feature_importance_ = None
        
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: Optional[pd.DataFrame] = None, 
        y_val: Optional[pd.Series] = None,
        categorical_features: List[str] = 'auto'
    ):
        """
        Train the model.
        """
        logger.info(f"Training LightGBM model with {X_train.shape[1]} features...")
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
            valid_sets.append(val_data)
            valid_names.append('val')
            
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Save feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Training complete.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)
        
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
        
        # Handle log-transformed targets if necessary (assuming input is already transformed)
        # Metrics
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        
        # RMSLE (Root Mean Squared Logarithmic Error)
        # Ensure non-negative for log
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
