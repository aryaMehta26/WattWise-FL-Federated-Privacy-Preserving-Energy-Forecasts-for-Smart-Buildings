"""Data validation and quality checks for BDG2 dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..utils.config import load_config, get_paths
from ..utils.logging_utils import get_logger
from ..utils.io import save_json, load_parquet

logger = get_logger(__name__)


def check_schema(df: pd.DataFrame, expected_columns: list) -> Dict[str, Any]:
    """
    Check if dataframe has expected columns.
    
    Args:
        df: Dataframe to check
        expected_columns: List of expected column names
        
    Returns:
        Dictionary with validation results
    """
    actual_columns = set(df.columns)
    expected_set = set(expected_columns)
    
    missing = expected_set - actual_columns
    extra = actual_columns - expected_set
    
    passed = len(missing) == 0
    
    return {
        'check': 'schema_validation',
        'passed': passed,
        'missing_columns': list(missing),
        'extra_columns': list(extra),
        'message': 'All required columns present' if passed else f'Missing columns: {missing}'
    }


def check_missing_values(df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
    """
    Check for excessive missing values.
    
    Args:
        df: Dataframe to check
        threshold: Maximum allowed ratio of missing values
        
    Returns:
        Dictionary with validation results
    """
    missing_ratio = df.isna().sum() / len(df)
    high_missing = missing_ratio[missing_ratio > threshold]
    
    passed = len(high_missing) == 0
    
    return {
        'check': 'missing_values',
        'passed': passed,
        'threshold': threshold,
        'high_missing_columns': {col: f"{ratio:.2%}" for col, ratio in high_missing.items()},
        'message': f'No columns exceed {threshold:.0%} missing' if passed else f'{len(high_missing)} columns exceed threshold'
    }


def check_duplicates(df: pd.DataFrame, subset: list = None) -> Dict[str, Any]:
    """
    Check for duplicate rows.
    
    Args:
        df: Dataframe to check
        subset: Columns to use for duplicate detection
        
    Returns:
        Dictionary with validation results
    """
    n_duplicates = df.duplicated(subset=subset).sum()
    duplicate_ratio = n_duplicates / len(df)
    
    passed = n_duplicates == 0
    
    return {
        'check': 'duplicate_rows',
        'passed': passed,
        'n_duplicates': int(n_duplicates),
        'duplicate_ratio': f"{duplicate_ratio:.2%}",
        'message': 'No duplicates found' if passed else f'Found {n_duplicates} duplicates'
    }


def check_timestamp_gaps(df: pd.DataFrame, max_gap_hours: int = 2) -> Dict[str, Any]:
    """
    Check for large gaps in timestamps.
    
    Args:
        df: Dataframe with timestamp column
        max_gap_hours: Maximum allowed gap in hours
        
    Returns:
        Dictionary with validation results
    """
    if 'timestamp' not in df.columns:
        return {
            'check': 'timestamp_gaps',
            'passed': False,
            'message': 'No timestamp column found'
        }
    
    # Sort by building_id and timestamp
    df_sorted = df.sort_values(['building_id', 'timestamp'])
    
    # Calculate time differences within each building
    df_sorted['time_diff'] = df_sorted.groupby('building_id')['timestamp'].diff()
    
    # Convert to hours
    time_diff_hours = df_sorted['time_diff'].dt.total_seconds() / 3600
    
    # Find large gaps
    large_gaps = time_diff_hours[time_diff_hours > max_gap_hours]
    n_large_gaps = len(large_gaps)
    
    passed = n_large_gaps == 0
    
    max_gap = time_diff_hours.max() if len(time_diff_hours) > 0 else 0
    
    return {
        'check': 'timestamp_gaps',
        'passed': passed,
        'max_gap_hours': float(max_gap) if not pd.isna(max_gap) else 0,
        'n_large_gaps': int(n_large_gaps),
        'message': f'No large gaps (>{max_gap_hours}h)' if passed else f'Found {n_large_gaps} large gaps'
    }


def check_negative_readings(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for negative meter readings.
    
    Args:
        df: Dataframe with meter_reading column
        
    Returns:
        Dictionary with validation results
    """
    if 'meter_reading' not in df.columns:
        return {
            'check': 'negative_readings',
            'passed': False,
            'message': 'No meter_reading column found'
        }
    
    n_negative = (df['meter_reading'] < 0).sum()
    negative_ratio = n_negative / len(df)
    
    passed = n_negative == 0
    
    return {
        'check': 'negative_readings',
        'passed': passed,
        'n_negative': int(n_negative),
        'negative_ratio': f"{negative_ratio:.2%}",
        'message': 'No negative readings' if passed else f'Found {n_negative} negative readings'
    }


def check_outliers(df: pd.DataFrame, column: str = 'meter_reading', method: str = 'iqr', multiplier: float = 3.0) -> Dict[str, Any]:
    """
    Check for outliers in meter readings.
    
    Args:
        df: Dataframe to check
        column: Column to check for outliers
        method: Method to use ('iqr' or 'zscore')
        multiplier: Multiplier for outlier threshold
        
    Returns:
        Dictionary with validation results
    """
    if column not in df.columns:
        return {
            'check': 'outliers',
            'passed': False,
            'message': f'Column {column} not found'
        }
    
    values = df[column].dropna()
    
    if method == 'iqr':
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = (values < lower_bound) | (values > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((values - values.mean()) / values.std())
        outliers = z_scores > multiplier
    else:
        raise ValueError(f"Unknown method: {method}")
    
    n_outliers = outliers.sum()
    outlier_ratio = n_outliers / len(values)
    
    return {
        'check': 'outliers',
        'passed': True,  # Outliers are informational, not necessarily a failure
        'method': method,
        'n_outliers': int(n_outliers),
        'outlier_ratio': f"{outlier_ratio:.2%}",
        'message': f'Found {n_outliers} outliers ({outlier_ratio:.2%})'
    }


def validate_data(df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run all validation checks on dataframe.
    
    Args:
        df: Dataframe to validate
        config: Configuration dictionary with validation parameters
        
    Returns:
        Dictionary with all validation results
    """
    if config is None:
        config = load_config()
    
    logger.info("Running data validation checks...")
    
    validation_results = {
        'timestamp': datetime.utcnow().isoformat(),
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'checks': []
    }
    
    # Required columns for merged data
    required_columns = ['building_id', 'timestamp', 'meter_reading']
    
    # Run checks
    checks = [
        check_schema(df, required_columns),
        check_missing_values(df, threshold=config.get('data_quality', {}).get('thresholds', {}).get('max_missing_ratio', 0.3)),
        check_duplicates(df, subset=['building_id', 'timestamp']),
        check_timestamp_gaps(df, max_gap_hours=2),
        check_negative_readings(df),
        check_outliers(df, column='meter_reading', method='iqr', multiplier=3.0),
    ]
    
    # Add results
    for check_result in checks:
        validation_results['checks'].append(check_result)
        
        status = "✓ PASS" if check_result['passed'] else "✗ FAIL"
        logger.info(f"  {status}: {check_result['check']} - {check_result['message']}")
    
    # Overall pass/fail
    all_passed = all(check['passed'] for check in validation_results['checks'])
    validation_results['overall_passed'] = all_passed
    
    logger.info(f"\nOverall validation: {'PASSED' if all_passed else 'FAILED'}")
    
    return validation_results


def generate_data_quality_report(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    
    Args:
        df: Dataframe to analyze
        output_dir: Directory to save report
        
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Generating data quality report...")
    
    if output_dir is None:
        config = load_config()
        paths = get_paths(config)
        output_dir = paths['processed_data']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'summary': {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        },
        'columns': {},
        'time_range': {},
        'building_stats': {},
    }
    
    # Column-level statistics
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'n_missing': int(df[col].isna().sum()),
            'missing_ratio': float(df[col].isna().sum() / len(df)),
            'n_unique': int(df[col].nunique()),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
            })
        
        report['columns'][col] = col_stats
    
    # Time range
    if 'timestamp' in df.columns:
        report['time_range'] = {
            'min': str(df['timestamp'].min()),
            'max': str(df['timestamp'].max()),
            'n_unique_timestamps': int(df['timestamp'].nunique()),
        }
    
    # Building statistics
    if 'building_id' in df.columns:
        readings_per_building = df.groupby('building_id').size()
        report['building_stats'] = {
            'n_buildings': int(df['building_id'].nunique()),
            'min_readings_per_building': int(readings_per_building.min()),
            'max_readings_per_building': int(readings_per_building.max()),
            'mean_readings_per_building': float(readings_per_building.mean()),
            'median_readings_per_building': float(readings_per_building.median()),
        }
    
    # Save report
    report_path = output_dir / 'data_quality_report.json'
    save_json(report, report_path)
    logger.info(f"Data quality report saved to {report_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Data Quality Report Summary")
    logger.info("="*60)
    logger.info(f"Total rows: {report['summary']['n_rows']:,}")
    logger.info(f"Total columns: {report['summary']['n_columns']}")
    logger.info(f"Memory usage: {report['summary']['memory_usage_mb']:.2f} MB")
    if 'building_stats' in report and report['building_stats']:
        logger.info(f"Buildings: {report['building_stats']['n_buildings']}")
    if 'time_range' in report and report['time_range']:
        logger.info(f"Time range: {report['time_range']['min']} to {report['time_range']['max']}")
    logger.info("="*60)
    
    return report


if __name__ == "__main__":
    """Run validation as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate BDG2 data")
    parser.add_argument('--data-file', type=str, help="Path to processed data file")
    parser.add_argument('--output-dir', type=str, help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Load data
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        config = load_config()
        paths = get_paths(config)
        data_path = paths['processed_data'] / 'processed_data.parquet'
    
    logger.info(f"Loading data from {data_path}")
    df = load_parquet(data_path)
    
    # Run validation
    validation_results = validate_data(df)
    
    # Generate quality report
    report = generate_data_quality_report(
        df,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    # Save validation results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        config = load_config()
        paths = get_paths(config)
        output_dir = paths['processed_data']
    
    validation_path = output_dir / 'validation_results.json'
    save_json(validation_results, validation_path)
    logger.info(f"\nValidation results saved to {validation_path}")
    
    logger.info("\nValidation complete!")

