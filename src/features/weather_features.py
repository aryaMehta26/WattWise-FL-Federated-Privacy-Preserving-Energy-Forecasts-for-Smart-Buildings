"""Weather-based features."""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def add_weather_features(
    df: pd.DataFrame,
    weather_cols: List[str] = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed'],
    group_cols: List[str] = ['site_id']
) -> pd.DataFrame:
    """
    Add weather-derived features.
    
    Args:
        df: Dataframe containing weather columns
        weather_cols: List of weather columns to process
        group_cols: Columns to group by for rolling stats (usually site_id)
        
    Returns:
        Dataframe with weather features added
    """
    logger.info("Adding weather features...")
    
    df = df.copy()
    
    # Relative Humidity approximation
    # Formula: 100 * (EXP((17.625 * TD) / (243.04 + TD)) / EXP((17.625 * T) / (243.04 + T)))
    if 'air_temperature' in df.columns and 'dew_temperature' in df.columns:
        t = df['air_temperature']
        td = df['dew_temperature']
        
        # Avoid division by zero or invalid values
        try:
            # August-Roche-Magnus approximation
            a = 17.625
            b = 243.04
            
            # Calculate saturation vapor pressure
            es = 6.1094 * np.exp((a * t) / (b + t))
            # Calculate actual vapor pressure
            e = 6.1094 * np.exp((a * td) / (b + td))
            
            df['relative_humidity'] = 100 * (e / es)
            df['relative_humidity'] = df['relative_humidity'].clip(0, 100)
            logger.info("  Added relative_humidity")
        except Exception as e:
            logger.warning(f"Could not calculate relative humidity: {e}")

    # Rolling weather stats (e.g., 24h average temperature)
    # This captures thermal inertia of buildings
    if 'air_temperature' in df.columns:
        logger.info("  Calculating rolling temperature features...")
        # Sort to ensure correct rolling
        df = df.sort_values(group_cols + ['timestamp'])
        
        # 24h moving average
        df['air_temperature_rolling_24h'] = df.groupby(group_cols)['air_temperature'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        
        # 72h moving average (longer term thermal mass)
        df['air_temperature_rolling_72h'] = df.groupby(group_cols)['air_temperature'].transform(
            lambda x: x.rolling(window=72, min_periods=1).mean()
        )
        
        # Difference from 24h average (is it hotter/colder than recently?)
        df['air_temperature_diff_24h_mean'] = df['air_temperature'] - df['air_temperature_rolling_24h']

    logger.info("Weather features added.")
    return df
