"""Calendar and time-based features."""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Add calendar-based features (hour, day, month, etc.).
    
    Args:
        df: Dataframe with timestamp column
        timestamp_col: Name of timestamp column
        
    Returns:
        Dataframe with calendar features added
    """
    logger.info("Adding calendar features...")
    
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    dt = df[timestamp_col].dt
    
    # Basic components
    df['hour'] = dt.hour
    df['day'] = dt.day
    df['dayofweek'] = dt.dayofweek
    df['month'] = dt.month
    df['year'] = dt.year
    df['dayofyear'] = dt.dayofyear
    
    # Derived features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for periodic features (hour, dayofweek, month)
    # This helps models understand that hour 23 is close to hour 0
    for col, max_val in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        logger.info(f"  Added cyclical features for {col}")
        
    logger.info(f"Calendar features added. New columns: {list(df.columns[-10:])}")
    
    return df
