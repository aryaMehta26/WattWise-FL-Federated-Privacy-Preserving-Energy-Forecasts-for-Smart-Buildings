"""Data preprocessing and cleaning for BDG2 dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

from ..utils.config import load_config, get_paths
from ..utils.logging_utils import get_logger
from ..utils.io import save_parquet, load_csv

logger = get_logger(__name__)


def load_raw_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw BDG2 data files.
    
    Args:
        data_dir: Directory containing raw data files
        
    Returns:
        Tuple of (meter_df, metadata_df, weather_df)
    """
    if data_dir is None:
        config = load_config()
        paths = get_paths(config)
        data_dir = paths['raw_data']
    
    data_dir = Path(data_dir)
    
    logger.info("Loading raw data files...")
    
    # Load metadata
    metadata_path = data_dir / 'metadata.csv'
    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"  Loaded {len(metadata_df)} buildings")
    
    # Load weather
    weather_path = data_dir / 'weather.csv'
    logger.info(f"Loading weather from {weather_path}")
    weather_df = pd.read_csv(weather_path)
    logger.info(f"  Loaded {len(weather_df)} weather records")
    
    # Load meter data (try different filenames)
    meter_df = None
    meter_files = ['train.csv', 'electricity.csv', 'meter_readings.csv']
    
    for filename in meter_files:
        meter_path = data_dir / filename
        if meter_path.exists():
            logger.info(f"Loading meter data from {meter_path}")
            meter_df = pd.read_csv(meter_path)
            logger.info(f"  Loaded {len(meter_df)} meter readings")
            break
    
    if meter_df is None:
        raise FileNotFoundError(
            f"No meter data file found. Tried: {meter_files}\n"
            "Please download meter data (see src/data/download.py for instructions)"
        )
    
    return meter_df, metadata_df, weather_df


def clean_meter_data(
    meter_df: pd.DataFrame,
    remove_negatives: bool = True,
    remove_zeros: bool = False
) -> pd.DataFrame:
    """
    Clean meter reading data.
    
    Args:
        meter_df: Raw meter dataframe
        remove_negatives: Remove negative readings
        remove_zeros: Remove zero readings
        
    Returns:
        Cleaned meter dataframe
    """
    logger.info("Cleaning meter data...")
    initial_rows = len(meter_df)
    
    df = meter_df.copy()
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info("  ✓ Parsed timestamps")
    
    # Check for meter_reading column
    reading_col = None
    for col in ['meter_reading', 'reading', 'value']:
        if col in df.columns:
            reading_col = col
            break
    
    if reading_col is None:
        raise ValueError("Could not find meter reading column")
    
    if reading_col != 'meter_reading':
        df = df.rename(columns={reading_col: 'meter_reading'})
        logger.info(f"  ✓ Renamed '{reading_col}' to 'meter_reading'")
    
    # Remove negative readings
    if remove_negatives:
        negative_mask = df['meter_reading'] < 0
        n_negative = negative_mask.sum()
        if n_negative > 0:
            df = df[~negative_mask].copy()
            logger.info(f"  ✓ Removed {n_negative} negative readings")
    
    # Remove zero readings (optional)
    if remove_zeros:
        zero_mask = df['meter_reading'] == 0
        n_zeros = zero_mask.sum()
        if n_zeros > 0:
            df = df[~zero_mask].copy()
            logger.info(f"  ✓ Removed {n_zeros} zero readings")
    
    # Remove duplicates
    dup_cols = ['building_id', 'timestamp']
    if 'meter' in df.columns:
        dup_cols.append('meter')
    
    n_duplicates = df.duplicated(subset=dup_cols).sum()
    if n_duplicates > 0:
        df = df.drop_duplicates(subset=dup_cols, keep='first')
        logger.info(f"  ✓ Removed {n_duplicates} duplicate readings")
    
    # Sort by building_id and timestamp
    df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
    logger.info("  ✓ Sorted by building_id and timestamp")
    
    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    removed_pct = (removed_rows / initial_rows) * 100
    
    logger.info(f"Cleaning complete: {initial_rows} → {final_rows} rows ({removed_pct:.2f}% removed)")
    
    return df


def clean_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean weather data.
    
    Args:
        weather_df: Raw weather dataframe
        
    Returns:
        Cleaned weather dataframe
    """
    logger.info("Cleaning weather data...")
    
    df = weather_df.copy()
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info("  ✓ Parsed timestamps")
    
    # Forward fill missing values (short gaps only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    max_fill = 2  # Fill up to 2 hours
    
    for col in numeric_cols:
        if col == 'site_id':
            continue
        
        n_missing_before = df[col].isna().sum()
        if n_missing_before > 0:
            df[col] = df.groupby('site_id')[col].fillna(method='ffill', limit=max_fill)
            df[col] = df.groupby('site_id')[col].fillna(method='bfill', limit=max_fill)
            n_missing_after = df[col].isna().sum()
            filled = n_missing_before - n_missing_after
            logger.info(f"  ✓ Filled {filled} missing values in {col}")
    
    # Sort
    df = df.sort_values(['site_id', 'timestamp']).reset_index(drop=True)
    logger.info("  ✓ Sorted by site_id and timestamp")
    
    logger.info("Weather cleaning complete")
    
    return df


def clean_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean building metadata.
    
    Args:
        metadata_df: Raw metadata dataframe
        
    Returns:
        Cleaned metadata dataframe
    """
    logger.info("Cleaning metadata...")
    
    df = metadata_df.copy()
    
    # Add building age
    current_year = datetime.now().year
    if 'year_built' in df.columns:
        df['building_age'] = current_year - df['year_built']
        df['building_age'] = df['building_age'].clip(lower=0)
        logger.info("  ✓ Added building_age column")
    
    # Log transform square_feet
    if 'square_feet' in df.columns:
        df['log_square_feet'] = np.log1p(df['square_feet'])
        logger.info("  ✓ Added log_square_feet column")
    
    # Clean primary_use
    if 'primary_use' in df.columns:
        df['primary_use'] = df['primary_use'].str.strip().str.lower()
        logger.info(f"  ✓ Standardized primary_use ({df['primary_use'].nunique()} unique values)")
    
    logger.info("Metadata cleaning complete")
    
    return df


def merge_all_data(
    meter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge meter, metadata, and weather data.
    
    Args:
        meter_df: Cleaned meter dataframe
        metadata_df: Cleaned metadata dataframe
        weather_df: Cleaned weather dataframe
        
    Returns:
        Merged dataframe
    """
    logger.info("Merging datasets...")
    
    # Merge meter with metadata
    logger.info("  Merging meter + metadata...")
    df = meter_df.merge(metadata_df, on='building_id', how='left')
    logger.info(f"    After merge: {len(df)} rows")
    
    # Merge with weather (on site_id and timestamp)
    logger.info("  Merging with weather...")
    df = df.merge(weather_df, on=['site_id', 'timestamp'], how='left', suffixes=('', '_weather'))
    logger.info(f"    After merge: {len(df)} rows")
    
    # Check for missing values
    missing_pct = (df.isna().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 10]
    
    if len(high_missing) > 0:
        logger.warning("  Columns with >10% missing values:")
        for col, pct in high_missing.items():
            logger.warning(f"    {col}: {pct:.2f}%")
    
    logger.info("Merge complete!")
    
    return df


def preprocess_meter_data(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    meter_type: str = 'electricity',
    start_date: str = '2016-01-01',
    end_date: str = '2017-12-31',
    save_intermediate: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for BDG2 data.
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data
        meter_type: Meter type to process ('electricity', 'all', etc.)
        start_date: Start date for filtering
        end_date: End date for filtering
        save_intermediate: Save intermediate cleaned files
        
    Returns:
        Preprocessed dataframe
    """
    logger.info("="*60)
    logger.info("Starting BDG2 Data Preprocessing Pipeline")
    logger.info("="*60)
    
    # Set up paths
    if data_dir is None or output_dir is None:
        config = load_config()
        paths = get_paths(config)
        if data_dir is None:
            data_dir = paths['raw_data']
        if output_dir is None:
            output_dir = paths['processed_data']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    meter_df, metadata_df, weather_df = load_raw_data(data_dir)
    
    # Clean each dataset
    meter_df = clean_meter_data(meter_df)
    weather_df = clean_weather_data(weather_df)
    metadata_df = clean_metadata(metadata_df)
    
    # Save intermediate files
    if save_intermediate:
        logger.info("Saving intermediate cleaned files...")
        save_parquet(meter_df, output_dir / 'meter_clean.parquet')
        save_parquet(weather_df, output_dir / 'weather_clean.parquet')
        save_parquet(metadata_df, output_dir / 'metadata_clean.parquet')
        logger.info("  ✓ Intermediate files saved")
    
    # Merge all data
    df = merge_all_data(meter_df, metadata_df, weather_df)
    
    # Filter by date range
    logger.info(f"Filtering date range: {start_date} to {end_date}")
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    logger.info(f"  After filtering: {len(df)} rows")
    
    # Filter by meter type if specified
    if meter_type != 'all' and 'meter' in df.columns:
        logger.info(f"Filtering for meter type: {meter_type}")
        # Map meter type to meter code if needed
        meter_type_map = {'electricity': 0, 'chilledwater': 1, 'steam': 2, 'hotwater': 3}
        if meter_type in meter_type_map:
            meter_code = meter_type_map[meter_type]
            df = df[df['meter'] == meter_code].copy()
        logger.info(f"  After filtering: {len(df)} rows")
    
    # Final save
    output_path = output_dir / 'processed_data.parquet'
    logger.info(f"Saving processed data to {output_path}")
    save_parquet(df, output_path)
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("Preprocessing Summary")
    logger.info("="*60)
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Total buildings: {df['building_id'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Mean reading: {df['meter_reading'].mean():.2f}")
    logger.info(f"Median reading: {df['meter_reading'].median():.2f}")
    logger.info("="*60)
    
    return df


if __name__ == "__main__":
    """Run preprocessing as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess BDG2 data")
    parser.add_argument('--data-dir', type=str, help="Raw data directory")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('--meter-type', type=str, default='electricity', help="Meter type")
    
    args = parser.parse_args()
    
    df = preprocess_meter_data(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        meter_type=args.meter_type
    )
    
    logger.info("\nPreprocessing complete!")
    logger.info("Next steps:")
    logger.info("  1. Run validation: python -m src.data.validation")
    logger.info("  2. Explore data: jupyter notebook notebooks/05_preprocessing.ipynb")
    logger.info("  3. Build features: python -m src.features.calendar_features")

