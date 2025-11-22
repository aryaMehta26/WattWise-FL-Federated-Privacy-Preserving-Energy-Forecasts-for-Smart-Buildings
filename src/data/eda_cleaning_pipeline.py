"""
Enhanced Data Cleaning Pipeline with EDA-Driven Insights.
This module implements a comprehensive cleaning pipeline based on EDA findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """
    Professional data cleaning pipeline with EDA-driven decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cleaning pipeline.
        
        Args:
            config: Configuration dictionary with cleaning parameters
        """
        self.config = config or {}
        self.stats = {}
        
    def remove_buildings_high_missing(
        self, 
        df: pd.DataFrame, 
        building_cols: List[str],
        threshold: float = 0.90
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove buildings with high percentage of missing data.
        
        Args:
            df: Wide format dataframe
            building_cols: List of building column names
            threshold: Maximum allowed missing percentage (default 0.90)
            
        Returns:
            Tuple of (cleaned dataframe, removed building IDs)
        """
        logger.info(f"Analyzing missing data for {len(building_cols)} buildings...")
        
        missing_pct = {}
        for col in building_cols:
            missing_pct[col] = (df[col].isnull().sum() / len(df)) * 100
        
        missing_series = pd.Series(missing_pct)
        high_missing = missing_series[missing_series > threshold * 100]
        
        if len(high_missing) > 0:
            logger.info(f"Removing {len(high_missing)} buildings with >{threshold*100:.0f}% missing data")
            removed_buildings = high_missing.index.tolist()
            df_cleaned = df.drop(columns=removed_buildings)
            return df_cleaned, removed_buildings
        else:
            logger.info("No buildings exceed missing data threshold")
            return df, []
    
    def detect_and_handle_outliers(
        self,
        df: pd.DataFrame,
        value_col: str = 'meter_reading',
        method: str = 'iqr',
        iqr_multiplier: float = 3.0,
        action: str = 'clip'
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in consumption data.
        
        Args:
            df: Long format dataframe
            value_col: Name of value column
            method: Detection method ('iqr' or 'zscore')
            iqr_multiplier: IQR multiplier for bounds
            action: Action to take ('clip', 'remove', or 'flag')
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        df = df.copy()
        initial_count = len(df)
        
        if method == 'iqr':
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
        else:  # zscore
            mean = df[value_col].mean()
            std = df[value_col].std()
            lower_bound = mean - iqr_multiplier * std
            upper_bound = mean + iqr_multiplier * std
        
        outliers = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]
        n_outliers = len(outliers)
        
        logger.info(f"Found {n_outliers} outliers ({n_outliers/initial_count*100:.2f}%)")
        logger.info(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        if action == 'clip':
            df[value_col] = df[value_col].clip(lower=lower_bound, upper=upper_bound)
            logger.info("Outliers clipped to bounds")
        elif action == 'remove':
            df = df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]
            logger.info(f"Removed {initial_count - len(df)} outlier rows")
        elif action == 'flag':
            df['is_outlier'] = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
            logger.info("Outliers flagged in 'is_outlier' column")
        
        return df
    
    def fill_missing_values(
        self,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        max_gap_hours: int = 2,
        method: str = 'forward'
    ) -> pd.DataFrame:
        """
        Fill missing values using forward/backward fill within groups.
        
        Args:
            df: Long format dataframe
            group_cols: Columns to group by (e.g., ['building_id'])
            max_gap_hours: Maximum gap to fill
            method: Fill method ('forward', 'backward', or 'both')
            
        Returns:
            Dataframe with filled values
        """
        logger.info(f"Filling missing values (max gap: {max_gap_hours} hours)...")
        
        df = df.copy()
        initial_missing = df['meter_reading'].isna().sum()
        
        if group_cols:
            if method in ['forward', 'both']:
                df['meter_reading'] = df.groupby(group_cols)['meter_reading'].ffill(limit=max_gap_hours)
            if method in ['backward', 'both']:
                df['meter_reading'] = df.groupby(group_cols)['meter_reading'].bfill(limit=max_gap_hours)
        else:
            if method in ['forward', 'both']:
                df['meter_reading'] = df['meter_reading'].ffill(limit=max_gap_hours)
            if method in ['backward', 'both']:
                df['meter_reading'] = df['meter_reading'].bfill(limit=max_gap_hours)
        
        final_missing = df['meter_reading'].isna().sum()
        filled = initial_missing - final_missing
        
        logger.info(f"Filled {filled:,} missing values ({filled/initial_missing*100:.2f}% of missing)")
        
        return df
    
    def remove_negative_values(
        self,
        df: pd.DataFrame,
        value_col: str = 'meter_reading',
        action: str = 'remove'
    ) -> pd.DataFrame:
        """
        Handle negative consumption values.
        
        Args:
            df: Long format dataframe
            value_col: Name of value column
            action: Action to take ('remove' or 'clip')
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Handling negative values...")
        
        negative_count = (df[value_col] < 0).sum()
        
        if negative_count > 0:
            logger.info(f"Found {negative_count} negative values")
            
            if action == 'remove':
                df = df[df[value_col] >= 0]
                logger.info(f"Removed {negative_count} negative value rows")
            else:  # clip
                df[value_col] = df[value_col].clip(lower=0)
                logger.info(f"Clipped {negative_count} negative values to 0")
        else:
            logger.info("No negative values found")
        
        return df
    
    def remove_zero_values(
        self,
        df: pd.DataFrame,
        value_col: str = 'meter_reading',
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove buildings with excessive zero values.
        
        Args:
            df: Long format dataframe
            value_col: Name of value column
            threshold: Maximum allowed zero percentage
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Analyzing zero values...")
        
        zero_pct_by_building = df.groupby('building_id')[value_col].apply(
            lambda x: (x == 0).sum() / len(x) * 100
        )
        
        high_zero_buildings = zero_pct_by_building[zero_pct_by_building > threshold * 100]
        
        if len(high_zero_buildings) > 0:
            logger.info(f"Removing {len(high_zero_buildings)} buildings with >{threshold*100:.0f}% zeros")
            df = df[~df['building_id'].isin(high_zero_buildings.index)]
        else:
            logger.info("No buildings exceed zero value threshold")
        
        return df
    
    def validate_timestamp_continuity(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        expected_freq: str = '1H'
    ) -> Dict:
        """
        Validate timestamp continuity and detect gaps.
        
        Args:
            df: Dataframe with timestamp column
            timestamp_col: Name of timestamp column
            expected_freq: Expected frequency (e.g., '1H' for hourly)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating timestamp continuity...")
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        
        expected_range = pd.date_range(
            start=df[timestamp_col].min(),
            end=df[timestamp_col].max(),
            freq=expected_freq
        )
        
        actual_timestamps = set(df[timestamp_col].unique())
        expected_timestamps = set(expected_range)
        
        missing_timestamps = expected_timestamps - actual_timestamps
        extra_timestamps = actual_timestamps - expected_timestamps
        
        results = {
            'expected_count': len(expected_range),
            'actual_count': len(actual_timestamps),
            'missing_count': len(missing_timestamps),
            'extra_count': len(extra_timestamps),
            'completeness': len(actual_timestamps) / len(expected_range) * 100
        }
        
        logger.info(f"Timestamp validation:")
        logger.info(f"  Expected: {results['expected_count']}")
        logger.info(f"  Actual: {results['actual_count']}")
        logger.info(f"  Missing: {results['missing_count']}")
        logger.info(f"  Completeness: {results['completeness']:.2f}%")
        
        return results
    
    def clean_pipeline(
        self,
        df: pd.DataFrame,
        building_cols: Optional[List[str]] = None,
        timestamp_col: str = 'timestamp',
        value_col: str = 'meter_reading',
        is_wide_format: bool = True
    ) -> pd.DataFrame:
        """
        Run complete cleaning pipeline.
        
        Args:
            df: Input dataframe (wide or long format)
            building_cols: List of building columns (if wide format)
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            is_wide_format: Whether input is wide format
            
        Returns:
            Cleaned dataframe in long format
        """
        logger.info("="*60)
        logger.info("Starting Enhanced Data Cleaning Pipeline")
        logger.info("="*60)
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Step 1: Convert to long format if needed
        if is_wide_format:
            if building_cols is None:
                building_cols = [c for c in df.columns if c != timestamp_col]
            
            logger.info(f"Converting wide to long format ({len(building_cols)} buildings)...")
            df_long = pd.melt(
                df[[timestamp_col] + building_cols],
                id_vars=[timestamp_col],
                value_vars=building_cols,
                var_name='building_id',
                value_name=value_col
            )
            
            # Remove buildings with high missing data
            df_wide_cleaned, removed = self.remove_buildings_high_missing(
                df, building_cols, threshold=self.config.get('max_missing_pct', 0.90)
            )
            
            # Re-melt after removing buildings
            remaining_buildings = [c for c in building_cols if c not in removed]
            if len(remaining_buildings) < len(building_cols):
                df_long = pd.melt(
                    df_wide_cleaned[[timestamp_col] + remaining_buildings],
                    id_vars=[timestamp_col],
                    value_vars=remaining_buildings,
                    var_name='building_id',
                    value_name=value_col
                )
        else:
            df_long = df.copy()
        
        # Step 2: Remove negative values
        df_long = self.remove_negative_values(
            df_long, 
            value_col=value_col,
            action=self.config.get('negative_action', 'remove')
        )
        
        # Step 3: Fill missing values
        df_long = self.fill_missing_values(
            df_long,
            group_cols=['building_id'],
            max_gap_hours=self.config.get('max_fill_hours', 2),
            method='both'
        )
        
        # Step 4: Handle outliers
        df_long = self.detect_and_handle_outliers(
            df_long,
            value_col=value_col,
            method=self.config.get('outlier_method', 'iqr'),
            iqr_multiplier=self.config.get('iqr_multiplier', 3.0),
            action=self.config.get('outlier_action', 'clip')
        )
        
        # Step 5: Remove excessive zeros
        if self.config.get('remove_high_zeros', True):
            df_long = self.remove_zero_values(
                df_long,
                value_col=value_col,
                threshold=self.config.get('max_zero_pct', 0.95)
            )
        
        # Step 6: Sort and finalize
        df_long = df_long.sort_values([timestamp_col, 'building_id']).reset_index(drop=True)
        
        logger.info("="*60)
        logger.info("Cleaning Pipeline Complete")
        logger.info("="*60)
        logger.info(f"Final shape: {df_long.shape}")
        logger.info(f"Buildings: {df_long['building_id'].nunique()}")
        logger.info(f"Date range: {df_long[timestamp_col].min()} to {df_long[timestamp_col].max()}")
        
        return df_long

