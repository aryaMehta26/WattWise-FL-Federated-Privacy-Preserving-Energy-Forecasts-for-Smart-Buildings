"""
Run Comprehensive EDA Analysis
Executes all three EDA implementations and generates visualizations.
"""

import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.eda_temporal import TemporalEDA
from src.data.eda_building import BuildingEDA
from src.data.eda_meter import MeterEDA
from src.data.eda_cleaning_pipeline import DataCleaningPipeline
from src.visualization.plotting import (
    plot_temporal_patterns,
    plot_building_distributions,
    plot_time_series,
    plot_correlation_heatmap,
    plot_missing_data_analysis,
    plot_outlier_analysis
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return convert_to_serializable(obj.to_dict())
    elif pd.isna(obj):
        return None
    return obj


def main():
    """Run comprehensive EDA analysis."""
    logger.info("="*60)
    logger.info("COMPREHENSIVE EDA ANALYSIS")
    logger.info("="*60)
    
    # Setup paths
    data_dir = Path("data/raw")
    output_dir = Path("results/eda")
    figures_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nLoading electricity data...")
    elec_df = pd.read_csv(data_dir / 'electricity.txt')
    elec_df['timestamp'] = pd.to_datetime(elec_df['timestamp'])
    
    building_cols = [c for c in elec_df.columns if c != 'timestamp']
    logger.info(f"Loaded {len(building_cols)} buildings, {len(elec_df)} timestamps")
    
    # Convert to long format for analysis
    logger.info("\nConverting to long format...")
    sample_buildings = building_cols[:100]  # Sample for speed
    elec_long = pd.melt(
        elec_df[['timestamp'] + sample_buildings],
        id_vars=['timestamp'],
        value_vars=sample_buildings,
        var_name='building_id',
        value_name='meter_reading'
    )
    elec_long = elec_long.dropna()
    logger.info(f"Long format: {elec_long.shape}")
    
    # EDA Implementation 1: Temporal Analysis
    logger.info("\n" + "="*60)
    logger.info("EDA 1: TEMPORAL ANALYSIS")
    logger.info("="*60)
    
    temporal_eda = TemporalEDA(elec_long)
    temporal_report = temporal_eda.generate_report()
    
    # Save temporal report
    with open(output_dir / 'temporal_analysis.json', 'w') as f:
        json.dump(convert_to_serializable(temporal_report), f, indent=2, default=str)
    
    # Create temporal visualizations
    plot_temporal_patterns(
        elec_long,
        output_path=figures_dir / 'temporal_patterns.png'
    )
    
    logger.info("Temporal analysis complete")
    
    # EDA Implementation 2: Building-Level Analysis
    logger.info("\n" + "="*60)
    logger.info("EDA 2: BUILDING-LEVEL ANALYSIS")
    logger.info("="*60)
    
    building_eda = BuildingEDA(elec_long)
    building_report = building_eda.generate_report()
    
    # Save building report
    with open(output_dir / 'building_analysis.json', 'w') as f:
        json.dump(convert_to_serializable(building_report), f, indent=2, default=str)
    
    # Create building visualizations
    meta_df = building_eda.parse_building_metadata()
    stats_df = building_eda.calculate_building_statistics()
    combined_stats = stats_df.merge(meta_df, on='building_id', how='left')
    
    plot_building_distributions(
        combined_stats,
        output_path=figures_dir / 'building_level_analysis.png'
    )
    
    logger.info("Building-level analysis complete")
    
    # EDA Implementation 3: Meter-Level Analysis
    logger.info("\n" + "="*60)
    logger.info("EDA 3: METER-LEVEL ANALYSIS")
    logger.info("="*60)
    
    meter_eda = MeterEDA(data_dir)
    meter_report = meter_eda.generate_report()
    
    # Save meter report
    with open(output_dir / 'meter_analysis.json', 'w') as f:
        json.dump(convert_to_serializable(meter_report), f, indent=2, default=str)
    
    logger.info("Meter-level analysis complete")
    
    # Data Quality Analysis
    logger.info("\n" + "="*60)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("="*60)
    
    plot_missing_data_analysis(
        elec_df[building_cols[:50]],
        output_path=figures_dir / 'missing_data_analysis.png'
    )
    
    plot_outlier_analysis(
        elec_long['meter_reading'],
        output_path=figures_dir / 'outlier_analysis.png'
    )
    
    # Enhanced Cleaning Pipeline
    logger.info("\n" + "="*60)
    logger.info("ENHANCED DATA CLEANING PIPELINE")
    logger.info("="*60)
    
    cleaning_config = {
        'max_missing_pct': 0.90,
        'max_fill_hours': 2,
        'outlier_method': 'iqr',
        'iqr_multiplier': 3.0,
        'outlier_action': 'clip',
        'negative_action': 'remove',
        'remove_high_zeros': True,
        'max_zero_pct': 0.95
    }
    
    cleaning_pipeline = DataCleaningPipeline(cleaning_config)
    cleaned_df = cleaning_pipeline.clean_pipeline(
        elec_df,
        building_cols=sample_buildings,
        is_wide_format=True
    )
    
    logger.info(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Save cleaned data sample
    cleaned_df.head(10000).to_csv(output_dir / 'cleaned_sample.csv', index=False)
    
    # Final Summary
    logger.info("\n" + "="*60)
    logger.info("EDA ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - temporal_analysis.json")
    logger.info("  - building_analysis.json")
    logger.info("  - meter_analysis.json")
    logger.info("  - cleaned_sample.csv")
    logger.info("  - temporal_patterns.png")
    logger.info("  - building_level_analysis.png")
    logger.info("  - missing_data_analysis.png")
    logger.info("  - outlier_analysis.png")
    logger.info("="*60)


if __name__ == '__main__':
    main()

