"""Building metadata features."""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def add_building_features(
    df: pd.DataFrame,
    cat_cols: List[str] = ['primary_use']
) -> pd.DataFrame:
    """
    Add building metadata features.
    
    Args:
        df: Dataframe containing building metadata
        cat_cols: Categorical columns to encode
        
    Returns:
        Dataframe with building features added
    """
    logger.info("Adding building features...")
    
    df = df.copy()
    
    # One-hot encoding for primary_use
    if 'primary_use' in df.columns:
        # Get top N uses, group others as 'other' to prevent explosion
        top_uses = df['primary_use'].value_counts().nlargest(10).index
        df['primary_use_grouped'] = df['primary_use'].apply(lambda x: x if x in top_uses else 'other')
        
        dummies = pd.get_dummies(df['primary_use_grouped'], prefix='use')
        df = pd.concat([df, dummies], axis=1)
        logger.info(f"  One-hot encoded primary_use into {dummies.shape[1]} columns")
        
    # Interaction features
    # Example: square_feet per floor (if floor_count exists)
    if 'square_feet' in df.columns and 'floor_count' in df.columns:
        # Fill floor_count NaN with 1 to avoid division by zero
        floors = df['floor_count'].fillna(1)
        df['sqft_per_floor'] = df['square_feet'] / floors
        logger.info("  Added sqft_per_floor")
        
    return df
