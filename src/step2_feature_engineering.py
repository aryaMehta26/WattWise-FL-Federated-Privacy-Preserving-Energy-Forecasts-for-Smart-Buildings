"""
STEP 2: Feature Engineering
---------------------------
This script loads the cleaned data and generates the features needed for Machine Learning.
It creates:
- Calendar Features (Hour, Day, Weekend, etc.)
- Weather Features (Rolling averages, Humidity)
- Building Features (Interactions)
- Lag/Temporal Features (Past consumption values)

Output: data/processed/final_features.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.calendar_features import add_calendar_features
from src.features.weather_features import add_weather_features
from src.features.building_features import add_building_features
from src.features.lag_features import create_all_temporal_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("⚙️ STEP 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    input_path = "data/interim/cleaned_data.pkl"
    if not Path(input_path).exists():
        logger.error(f"❌ {input_path} not found! Please run Step 1 first.")
        return
        
    logger.info(f"Loading cleaned data from {input_path}...")
    df = pd.read_pickle(input_path)
    
    # 1. Calendar Features
    logger.info("--> Adding Calendar Features (Hour, Day, Month, Cyclical)...")
    df = add_calendar_features(df)
    
    # 2. Weather Features
    logger.info("--> Adding Weather Features (Rolling Temps, Humidity)...")
    df = add_weather_features(df)
    
    # 3. Building Features
    logger.info("--> Adding Building Features...")
    df = add_building_features(df)
    
    # 4. Temporal Features (Lags)
    logger.info("--> Creating Lag Features (Past 24h, Rolling Means)...")
    # Config for lags
    temporal_config = {
        'lags': {'enabled': True, 'lag_hours': [1, 24]},
        'rolling': {'enabled': True, 'windows': [{'window': 24, 'functions': ['mean']}]}
    }
    df = create_all_temporal_features(df, config=temporal_config, group_cols=['building_id'])
    
    # Handle Missing Values created by lags
    logger.info("--> Handling Missing Values...")
    initial_rows = len(df)
    
    # Drop rows where we don't have lags (first 24h)
    lag_cols = [c for c in df.columns if 'lag' in c or 'rolling' in c]
    subset_cols = lag_cols + ['meter_reading']
    df = df.dropna(subset=subset_cols)
    
    dropped_rows = initial_rows - len(df)
    logger.info(f"    Dropped {dropped_rows} rows (initial warm-up period for lags).")
    
    output_path = "data/processed/final_features.pkl"
    logger.info(f"Saving feature-rich data to {output_path}...")
    df.to_pickle(output_path)
    
    logger.info(f"✅ Step 2 Complete! Final shape: {df.shape}")
    logger.info(f"Next: Run 'python src/step3_train_models.py'")

if __name__ == "__main__":
    main()
