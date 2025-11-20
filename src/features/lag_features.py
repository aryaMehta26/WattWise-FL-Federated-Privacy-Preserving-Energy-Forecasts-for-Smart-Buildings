"""Lag and rolling window features for time-series forecasting."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = 'meter_reading',
    lag_hours: List[int] = [1, 24, 168],
    group_cols: List[str] = ['building_id']
) -> pd.DataFrame:
    """
    Add lag features (past values) for time-series forecasting.
    
    CRITICAL: These are "leakage-safe" lags that use only past data!
    - lag_1h: Value from 1 hour ago
    - lag_24h: Value from 24 hours ago (same time yesterday)
    - lag_168h: Value from 168 hours ago (same time last week)
    
    Args:
        df: Dataframe with timestamp and target column
        target_col: Column to create lags for
        lag_hours: List of lag periods in hours
        group_cols: Columns to group by (e.g., building_id)
        
    Returns:
        Dataframe with lag features added
    """
    logger.info(f"Adding lag features: {lag_hours} hours")
    
    df = df.sort_values(group_cols + ['timestamp']).copy()
    
    for lag in lag_hours:
        lag_col_name = f'{target_col}_lag_{lag}h'
        
        # Create lag within each group (building)
        df[lag_col_name] = df.groupby(group_cols)[target_col].shift(lag)
        
        n_missing = df[lag_col_name].isna().sum()
        logger.info(f"  Created {lag_col_name} ({n_missing:,} initial NaN values)")
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'meter_reading',
    windows: List[Dict] = None,
    group_cols: List[str] = ['building_id']
) -> pd.DataFrame:
    """
    Add rolling window features (moving averages, stdev, etc.).
    
    CRITICAL: Uses .shift(1) to ensure we only use PAST data!
    The window includes past data, NOT the current value.
    
    Args:
        df: Dataframe with timestamp and target column
        target_col: Column to create rolling features for
        windows: List of window configs. Each dict should have:
                 - 'window': int (window size in hours)
                 - 'functions': list of function names ('mean', 'std', 'min', 'max')
        group_cols: Columns to group by
        
    Returns:
        Dataframe with rolling features added
    """
    if windows is None:
        windows = [
            {'window': 24, 'functions': ['mean', 'std', 'min', 'max']},
            {'window': 168, 'functions': ['mean', 'std']},
        ]
    
    logger.info(f"Adding rolling window features: {len(windows)} windows")
    
    df = df.sort_values(group_cols + ['timestamp']).copy()
    
    for window_config in windows:
        window = window_config['window']
        functions = window_config['functions']
        
        for func_name in functions:
            col_name = f'{target_col}_rolling_{window}h_{func_name}'
            
            # Create rolling feature within each group
            # IMPORTANT: shift(1) ensures we don't include current value!
            grouped = df.groupby(group_cols)[target_col].shift(1)
            
            if func_name == 'mean':
                df[col_name] = grouped.rolling(window=window, min_periods=1).mean().values
            elif func_name == 'std':
                df[col_name] = grouped.rolling(window=window, min_periods=1).std().values
            elif func_name == 'min':
                df[col_name] = grouped.rolling(window=window, min_periods=1).min().values
            elif func_name == 'max':
                df[col_name] = grouped.rolling(window=window, min_periods=1).max().values
            else:
                logger.warning(f"Unknown function: {func_name}")
                continue
            
            n_missing = df[col_name].isna().sum()
            logger.info(f"  Created {col_name} ({n_missing:,} initial NaN values)")
    
    return df


def add_diff_features(
    df: pd.DataFrame,
    target_col: str = 'meter_reading',
    diff_hours: List[int] = [1, 24],
    group_cols: List[str] = ['building_id']
) -> pd.DataFrame:
    """
    Add difference features (rate of change).
    
    diff_1h: Change from 1 hour ago
    diff_24h: Change from 24 hours ago
    
    Args:
        df: Dataframe with timestamp and target column
        diff_hours: List of difference periods in hours
        group_cols: Columns to group by
        
    Returns:
        Dataframe with difference features added
    """
    logger.info(f"Adding difference features: {diff_hours} hours")
    
    df = df.sort_values(group_cols + ['timestamp']).copy()
    
    for diff_h in diff_hours:
        diff_col_name = f'{target_col}_diff_{diff_h}h'
        
        # Calculate difference within each group
        df[diff_col_name] = df.groupby(group_cols)[target_col].diff(diff_h)
        
        n_missing = df[diff_col_name].isna().sum()
        logger.info(f"  Created {diff_col_name} ({n_missing:,} initial NaN values)")
    
    return df


def add_expanding_features(
    df: pd.DataFrame,
    target_col: str = 'meter_reading',
    functions: List[str] = ['mean', 'std'],
    group_cols: List[str] = ['building_id']
) -> pd.DataFrame:
    """
    Add expanding window features (cumulative statistics).
    
    Expanding mean/std uses ALL historical data up to current point.
    Useful for capturing long-term trends.
    
    Args:
        df: Dataframe with timestamp and target column
        functions: List of function names
        group_cols: Columns to group by
        
    Returns:
        Dataframe with expanding features added
    """
    logger.info(f"Adding expanding window features: {functions}")
    
    df = df.sort_values(group_cols + ['timestamp']).copy()
    
    for func_name in functions:
        col_name = f'{target_col}_expanding_{func_name}'
        
        # Create expanding feature within each group
        # shift(1) to avoid leakage
        grouped = df.groupby(group_cols)[target_col].shift(1)
        
        if func_name == 'mean':
            df[col_name] = grouped.expanding(min_periods=1).mean().values
        elif func_name == 'std':
            df[col_name] = grouped.expanding(min_periods=1).std().values
        elif func_name == 'min':
            df[col_name] = grouped.expanding(min_periods=1).min().values
        elif func_name == 'max':
            df[col_name] = grouped.expanding(min_periods=1).max().values
        else:
            logger.warning(f"Unknown function: {func_name}")
            continue
        
        logger.info(f"  Created {col_name}")
    
    return df


def create_all_temporal_features(
    df: pd.DataFrame,
    target_col: str = 'meter_reading',
    config: Optional[Dict] = None,
    group_cols: List[str] = ['building_id']
) -> pd.DataFrame:
    """
    Create all temporal features (lags, rolling, diffs).
    
    This is a convenience function that creates the standard set of
    leakage-safe temporal features.
    
    Args:
        df: Dataframe with timestamp and target column
        target_col: Column to create features for
        config: Configuration dictionary. If None, uses defaults.
        group_cols: Columns to group by
        
    Returns:
        Dataframe with all temporal features added
    """
    logger.info("="*60)
    logger.info("Creating All Temporal Features")
    logger.info("="*60)
    
    initial_cols = len(df.columns)
    initial_rows = len(df)
    
    # Default configuration
    if config is None:
        config = {
            'lags': {
                'enabled': True,
                'lag_hours': [1, 24, 168]  # 1h, 1 day, 1 week
            },
            'rolling': {
                'enabled': True,
                'windows': [
                    {'window': 24, 'functions': ['mean', 'std', 'min', 'max']},
                    {'window': 168, 'functions': ['mean', 'std']}
                ]
            },
            'diffs': {
                'enabled': False,  # Optional
                'diff_hours': [1, 24]
            },
            'expanding': {
                'enabled': False,  # Optional
                'functions': ['mean', 'std']
            }
        }
    
    # Add features
    if config.get('lags', {}).get('enabled', True):
        df = add_lag_features(
            df,
            target_col=target_col,
            lag_hours=config['lags'].get('lag_hours', [1, 24, 168]),
            group_cols=group_cols
        )
    
    if config.get('rolling', {}).get('enabled', True):
        df = add_rolling_features(
            df,
            target_col=target_col,
            windows=config['rolling'].get('windows'),
            group_cols=group_cols
        )
    
    if config.get('diffs', {}).get('enabled', False):
        df = add_diff_features(
            df,
            target_col=target_col,
            diff_hours=config['diffs'].get('diff_hours', [1, 24]),
            group_cols=group_cols
        )
    
    if config.get('expanding', {}).get('enabled', False):
        df = add_expanding_features(
            df,
            target_col=target_col,
            functions=config['expanding'].get('functions', ['mean', 'std']),
            group_cols=group_cols
        )
    
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols
    
    logger.info("="*60)
    logger.info(f"Temporal Feature Creation Complete")
    logger.info(f"  Initial columns: {initial_cols}")
    logger.info(f"  Final columns: {final_cols}")
    logger.info(f"  New features: {new_features}")
    logger.info(f"  Rows: {initial_rows} → {len(df)}")
    logger.info("="*60)
    
    return df


if __name__ == "__main__":
    """Test lag and rolling features with synthetic data."""
    
    # Create synthetic time-series data
    dates = pd.date_range('2016-01-01', '2016-01-31', freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'building_id': [1] * len(dates),
        'meter_reading': np.random.rand(len(dates)) * 100 + 50
    })
    
    print(f"Test data: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Add features
    df = create_all_temporal_features(df, target_col='meter_reading')
    
    print(f"\nAfter adding features:")
    print(f"Columns: {len(df.columns)}")
    print(f"New feature columns: {[col for col in df.columns if col not in ['timestamp', 'building_id', 'meter_reading']]}")
    
    # Show sample
    print(f"\nSample data (first 5 rows):")
    print(df.head())
    
    # Check for leakage (lag features should be shifted correctly)
    print(f"\nLeakage check:")
    print(f"  Row 0 meter_reading: {df.iloc[0]['meter_reading']:.2f}")
    print(f"  Row 1 lag_1h (should be Row 0's value): {df.iloc[1]['meter_reading_lag_1h']:.2f}")
    print(f"  Match: {abs(df.iloc[0]['meter_reading'] - df.iloc[1]['meter_reading_lag_1h']) < 0.01}")

