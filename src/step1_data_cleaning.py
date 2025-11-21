"""
STEP 1: Data Cleaning & Preprocessing
-------------------------------------
This script handles the raw data ingestion, cleaning, and merging.
It takes the raw CSV files (wide format) and converts them into a clean,
long-format dataset merged with weather and building metadata.

Output: data/interim/cleaned_data.pkl
"""

import pandas as pd
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_meter_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("🧹 STEP 1: DATA CLEANING & PREPROCESSING")
    logger.info("="*60)
    
    # Ensure directories exist
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    
    # Check if raw data exists
    raw_path = Path("data/raw/electricity.csv")
    if not raw_path.exists():
        logger.error("❌ Electricity data not found! Please run 'python src/data/download.py' first.")
        return

    logger.info("Loading and preprocessing raw data...")
    logger.info("This handles:")
    logger.info("  1. Loading 'wide' format electricity data")
    logger.info("  2. Melting it into 'long' format (Building, Time, Reading)")
    logger.info("  3. Merging with Building Metadata (Age, Use, Size)")
    logger.info("  4. Merging with Weather Data (Temp, Humidity)")
    
    # We use a 6-month subset for the demo speed, but this can be changed to full range
    df = preprocess_meter_data(
        meter_type='electricity',
        start_date='2016-01-01',
        end_date='2016-06-30', 
        save_intermediate=False # We save manually below
    )
    
    # Print detailed metrics
    logger.info("\n" + "="*60)
    logger.info("📊 DATA METRICS AFTER CLEANING")
    logger.info("="*60)
    logger.info(f"Total Rows: {df.shape[0]:,}")
    logger.info(f"Total Columns: {df.shape[1]}")
    logger.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    logger.info(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Number of Buildings: {df['building_id'].nunique()}")
    logger.info(f"Number of Sites: {df['site_id'].nunique()}")
    logger.info(f"\nColumns: {', '.join(df.columns.tolist())}")
    logger.info("="*60 + "\n")
    
    output_path = "data/interim/cleaned_data.pkl"
    logger.info(f"Saving cleaned data to {output_path}...")
    df.to_pickle(output_path)
    
    logger.info(f"✅ Step 1 Complete! Data shape: {df.shape}")
    logger.info(f"Next: Run 'python src/step2_feature_engineering.py'")

if __name__ == "__main__":
    main()
