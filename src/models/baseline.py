"""Baseline models for building energy forecasting."""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV

from ..utils.logging_utils import get_logger
from ..evaluation.metrics import evaluate_predictions

logger = get_logger(__name__)


class NaiveWeeklyForecast:
    """
    Naive baseline: Predict same value as same hour last week (168 hours ago).
    
    This is a common baseline for time-series forecasting.
    """
    
    def __init__(self):
        self.name = "Naive Weekly"
    
    def fit(self, X, y):
        """No training needed for naive forecast."""
        return self
    
    def predict(self, X):
        """
        Predict using lag_168h feature (same hour last week).
        
        Args:
            X: Feature matrix with 'meter_reading_lag_168h' column
            
        Returns:
            Predictions
        """
        if 'meter_reading_lag_168h' not in X.columns:
            logger.warning("lag_168h not found, using lag_24h instead")
            if 'meter_reading_lag_24h' in X.columns:
                return X['meter_reading_lag_24h'].values
            else:
                logger.error("No lag features found!")
                return np.zeros(len(X))
        
        return X['meter_reading_lag_168h'].values


class RollingMeanForecast:
    """
    Rolling mean baseline: Predict using rolling average.
    
    Uses the rolling_24h_mean feature.
    """
    
    def __init__(self, window: int = 24):
        self.window = window
        self.name = f"Rolling Mean ({window}h)"
    
    def fit(self, X, y):
        """No training needed for rolling mean."""
        return self
    
    def predict(self, X):
        """
        Predict using rolling mean feature.
        
        Args:
            X: Feature matrix with rolling mean column
            
        Returns:
            Predictions
        """
        col_name = f'meter_reading_rolling_{self.window}h_mean'
        
        if col_name not in X.columns:
            logger.warning(f"{col_name} not found, using lag_24h instead")
            if 'meter_reading_lag_24h' in X.columns:
                return X['meter_reading_lag_24h'].values
            else:
                return np.zeros(len(X))
        
        return X[col_name].values


class BaselineRidge:
    """
    Ridge regression baseline with hyperparameter tuning.
    
    This is a simple linear model with L2 regularization.
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        param_grid: Optional[dict] = None,
        cv: int = 3
    ):
        """
        Initialize Ridge model.
        
        Args:
            alpha: Regularization strength. If None, will tune via CV.
            param_grid: Parameter grid for tuning
            cv: Number of CV folds for tuning
        """
        self.alpha = alpha
        self.param_grid = param_grid
        self.cv = cv
        self.model = None
        self.name = "Ridge Regression"
    
    def fit(self, X, y):
        """
        Train Ridge model.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self
        """
        logger.info(f"Training Ridge model on {len(X):,} samples")
        
        if self.alpha is not None:
            # Use specified alpha
            self.model = Ridge(alpha=self.alpha, random_state=42)
            self.model.fit(X, y)
            logger.info(f"  Trained with alpha={self.alpha}")
        else:
            # Tune alpha via cross-validation
            if self.param_grid is None:
                self.param_grid = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            
            logger.info(f"  Tuning hyperparameters: {self.param_grid}")
            
            base_model = Ridge(random_state=42)
            grid_search = GridSearchCV(
                base_model,
                self.param_grid,
                cv=self.cv,
                scoring='neg_mean_squared_error',
                n_jobs=1,  # Use 1 for testing, can set to -1 for production
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.alpha = grid_search.best_params_['alpha']
            
            logger.info(f"  Best alpha: {self.alpha}")
            logger.info(f"  Best CV score: {-grid_search.best_score_:.4f} (MSE)")
        
        # Log feature importance (absolute coefficients)
        if hasattr(X, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': self.model.coef_
            })
            feature_importance['abs_coef'] = np.abs(feature_importance['coefficient'])
            feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
            
            logger.info("  Top 5 features by coefficient magnitude:")
            for idx, row in feature_importance.head(5).iterrows():
                logger.info(f"    {row['feature']}: {row['coefficient']:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, X):
        """
        Get feature importance (absolute coefficients).
        
        Args:
            X: Feature matrix (to get column names)
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if hasattr(X, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': self.model.coef_,
                'abs_coefficient': np.abs(self.model.coef_)
            })
            return feature_importance.sort_values('abs_coefficient', ascending=False)
        else:
            return None


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Train all baseline models and evaluate.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with models and results
    """
    logger.info("="*60)
    logger.info("Training Baseline Models")
    logger.info("="*60)
    
    results = {}
    
    # 1. Naive Weekly
    logger.info("\n1. Naive Weekly Forecast")
    naive_model = NaiveWeeklyForecast()
    naive_model.fit(X_train, y_train)
    
    naive_pred_train = naive_model.predict(X_train)
    naive_pred_test = naive_model.predict(X_test)
    
    # Remove NaN predictions
    valid_train = ~np.isnan(naive_pred_train)
    valid_test = ~np.isnan(naive_pred_test)
    
    naive_metrics_train = evaluate_predictions(
        y_train[valid_train], 
        naive_pred_train[valid_train],
        prefix="train_"
    )
    naive_metrics_test = evaluate_predictions(
        y_test[valid_test], 
        naive_pred_test[valid_test],
        prefix="test_"
    )
    
    results['naive_weekly'] = {
        'model': naive_model,
        'train_metrics': naive_metrics_train,
        'test_metrics': naive_metrics_test
    }
    
    logger.info(f"  Train RMSLE: {naive_metrics_train['train_rmsle']:.4f}")
    logger.info(f"  Test RMSLE:  {naive_metrics_test['test_rmsle']:.4f}")
    
    # 2. Rolling Mean
    logger.info("\n2. Rolling Mean Forecast")
    rolling_model = RollingMeanForecast(window=24)
    rolling_model.fit(X_train, y_train)
    
    rolling_pred_train = rolling_model.predict(X_train)
    rolling_pred_test = rolling_model.predict(X_test)
    
    valid_train = ~np.isnan(rolling_pred_train)
    valid_test = ~np.isnan(rolling_pred_test)
    
    rolling_metrics_train = evaluate_predictions(
        y_train[valid_train], 
        rolling_pred_train[valid_train],
        prefix="train_"
    )
    rolling_metrics_test = evaluate_predictions(
        y_test[valid_test], 
        rolling_pred_test[valid_test],
        prefix="test_"
    )
    
    results['rolling_mean'] = {
        'model': rolling_model,
        'train_metrics': rolling_metrics_train,
        'test_metrics': rolling_metrics_test
    }
    
    logger.info(f"  Train RMSLE: {rolling_metrics_train['train_rmsle']:.4f}")
    logger.info(f"  Test RMSLE:  {rolling_metrics_test['test_rmsle']:.4f}")
    
    # 3. Ridge Regression
    logger.info("\n3. Ridge Regression")
    ridge_model = BaselineRidge()
    ridge_model.fit(X_train, y_train)
    
    ridge_pred_train = ridge_model.predict(X_train)
    ridge_pred_test = ridge_model.predict(X_test)
    
    ridge_metrics_train = evaluate_predictions(
        y_train, 
        ridge_pred_train,
        prefix="train_"
    )
    ridge_metrics_test = evaluate_predictions(
        y_test, 
        ridge_pred_test,
        prefix="test_"
    )
    
    results['ridge'] = {
        'model': ridge_model,
        'train_metrics': ridge_metrics_train,
        'test_metrics': ridge_metrics_test
    }
    
    logger.info(f"  Train RMSLE: {ridge_metrics_train['train_rmsle']:.4f}")
    logger.info(f"  Test RMSLE:  {ridge_metrics_test['test_rmsle']:.4f}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Baseline Model Summary (Test Set)")
    logger.info("="*60)
    
    for model_name, result in results.items():
        test_rmsle = result['test_metrics']['test_rmsle']
        test_mae = result['test_metrics']['test_mae']
        logger.info(f"{model_name:20s} - RMSLE: {test_rmsle:.4f}, MAE: {test_mae:.2f}")
    
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    """Test baseline models with synthetic data."""
    
    logger.info("Testing baseline models with synthetic data...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = pd.DataFrame({
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'temperature': np.random.rand(n_samples) * 30 + 50,
        'meter_reading_lag_24h': np.random.rand(n_samples) * 100,
        'meter_reading_lag_168h': np.random.rand(n_samples) * 100,
        'meter_reading_rolling_24h_mean': np.random.rand(n_samples) * 100,
    })
    
    # Create target (linear combination + noise)
    y = (
        2.0 * X['temperature'] +
        0.5 * X['meter_reading_lag_24h'] +
        np.random.randn(n_samples) * 10
    )
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train models
    results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    logger.info("\n✓ Baseline models test complete!")

