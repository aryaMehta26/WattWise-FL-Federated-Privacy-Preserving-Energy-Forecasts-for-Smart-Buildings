"""Time-series cross-validation splitter for building energy data."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
from datetime import timedelta

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TimeSeriesSplit:
    """
    Time-series cross-validation splitter.
    
    Unlike sklearn's TimeSeriesSplit, this splits by actual dates rather than
    just row indices, which is important for building energy data.
    
    Implements rolling or expanding window strategies.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_months: int = 1,
        gap_months: int = 0,
        method: str = "rolling"
    ):
        """
        Initialize time-series splitter.
        
        Args:
            n_splits: Number of splits
            test_months: Number of months in test set
            gap_months: Number of months between train and test (to avoid leakage)
            method: "rolling" (fixed train size) or "expanding" (growing train size)
        """
        self.n_splits = n_splits
        self.test_months = test_months
        self.gap_months = gap_months
        self.method = method
        
        if method not in ['rolling', 'expanding']:
            raise ValueError(f"method must be 'rolling' or 'expanding', got {method}")
    
    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits.
        
        Args:
            df: Dataframe with timestamp column
            timestamp_col: Name of timestamp column
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        min_date = df[timestamp_col].min()
        max_date = df[timestamp_col].max()
        total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
        
        logger.info(f"Date range: {min_date} to {max_date} ({total_months} months)")
        logger.info(f"Generating {self.n_splits} splits using {self.method} strategy")
        
        # Calculate split points
        months_per_split = (total_months - self.test_months) / self.n_splits
        
        for split_idx in range(self.n_splits):
            # Calculate test period end date
            test_end_month_offset = total_months - (split_idx * months_per_split)
            test_end_date = min_date + pd.DateOffset(months=int(test_end_month_offset))
            
            # Test period start date
            test_start_date = test_end_date - pd.DateOffset(months=self.test_months)
            
            # Gap before test
            train_end_date = test_start_date - pd.DateOffset(months=self.gap_months)
            
            # Train period start date
            if self.method == 'expanding':
                # Use all data before test period
                train_start_date = min_date
            else:
                # Rolling window: use fixed amount of historical data
                train_window_months = int(months_per_split)
                train_start_date = train_end_date - pd.DateOffset(months=train_window_months)
            
            # Get indices
            train_mask = (df[timestamp_col] >= train_start_date) & (df[timestamp_col] < train_end_date)
            test_mask = (df[timestamp_col] >= test_start_date) & (df[timestamp_col] < test_end_date)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                logger.warning(f"Split {split_idx + 1}: Empty train or test set, skipping")
                continue
            
            logger.info(f"Split {split_idx + 1}/{self.n_splits}:")
            logger.info(f"  Train: {train_start_date.date()} to {train_end_date.date()} ({len(train_indices):,} samples)")
            logger.info(f"  Test:  {test_start_date.date()} to {test_end_date.date()} ({len(test_indices):,} samples)")
            
            yield train_indices, test_indices
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def create_time_series_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_months: int = 1,
    gap_months: int = 0,
    method: str = "rolling",
    timestamp_col: str = 'timestamp'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-series CV splits and return as list.
    
    Args:
        df: Dataframe with timestamp column
        n_splits: Number of splits
        test_months: Months in test set
        gap_months: Gap between train and test
        method: "rolling" or "expanding"
        timestamp_col: Name of timestamp column
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_months=test_months,
        gap_months=gap_months,
        method=method
    )
    
    splits = list(splitter.split(df, timestamp_col=timestamp_col))
    
    logger.info(f"Created {len(splits)} time-series splits")
    
    return splits


def split_train_test_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    test_start_date: str = None,
    timestamp_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split by date.
    
    Args:
        df: Dataframe with timestamp column
        train_end_date: Last date for training (exclusive)
        test_start_date: First date for testing. If None, uses train_end_date.
        timestamp_col: Name of timestamp column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    train_end = pd.to_datetime(train_end_date)
    
    if test_start_date is None:
        test_start = train_end
    else:
        test_start = pd.to_datetime(test_start_date)
    
    train_df = df[df[timestamp_col] < train_end].copy()
    test_df = df[df[timestamp_col] >= test_start].copy()
    
    logger.info(f"Train set: {len(train_df):,} samples (up to {train_end.date()})")
    logger.info(f"Test set: {len(test_df):,} samples (from {test_start.date()})")
    
    return train_df, test_df


def split_unseen_sites(
    df: pd.DataFrame,
    test_site_ratio: float = 0.2,
    site_col: str = 'site_id',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by holding out entire sites for testing.
    
    This tests generalization to completely unseen buildings.
    
    Args:
        df: Dataframe with site_id column
        test_site_ratio: Ratio of sites to hold out for testing
        site_col: Name of site column
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    unique_sites = df[site_col].unique()
    n_sites = len(unique_sites)
    n_test_sites = max(1, int(n_sites * test_site_ratio))
    
    np.random.seed(random_state)
    test_sites = np.random.choice(unique_sites, size=n_test_sites, replace=False)
    
    test_mask = df[site_col].isin(test_sites)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Total sites: {n_sites}")
    logger.info(f"Train sites: {n_sites - n_test_sites} ({len(train_df):,} samples)")
    logger.info(f"Test sites: {n_test_sites} ({len(test_df):,} samples)")
    logger.info(f"Test site IDs: {test_sites}")
    
    return train_df, test_df


if __name__ == "__main__":
    """Test time-series splitter with synthetic data."""
    
    # Create synthetic data
    dates = pd.date_range('2016-01-01', '2017-12-31', freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'building_id': np.random.randint(0, 100, len(dates)),
        'site_id': np.random.randint(0, 5, len(dates)),
        'value': np.random.rand(len(dates)) * 100
    })
    
    print(f"Test data: {len(df):,} samples from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Test time-series CV
    print("\n" + "="*60)
    print("Testing Time-Series Cross-Validation")
    print("="*60)
    
    splits = create_time_series_splits(
        df,
        n_splits=3,
        test_months=1,
        gap_months=0,
        method="rolling"
    )
    
    print(f"\nCreated {len(splits)} splits")
    
    # Test unseen site split
    print("\n" + "="*60)
    print("Testing Unseen Site Split")
    print("="*60)
    
    train_df, test_df = split_unseen_sites(df, test_site_ratio=0.2)
    
    print(f"\nTrain buildings: {train_df['building_id'].nunique()}")
    print(f"Test buildings: {test_df['building_id'].nunique()}")

