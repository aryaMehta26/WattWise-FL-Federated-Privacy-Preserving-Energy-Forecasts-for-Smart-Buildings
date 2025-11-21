"""
STEP 3: Model Training & Evaluation
-----------------------------------
This script trains the Machine Learning models on the processed data.
Models:
1. LightGBM (Gradient Boosting) - High Accuracy
2. EBM (Explainable Boosting Machine) - High Interpretability
3. Federated Learning Simulator - Privacy-Preserving Demo

Output: 
- models/lgb_model.pkl
- models/ebm_model.pkl
- data/processed/dashboard_data.csv (for Streamlit)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lightgbm_model import LightGBMModel
from src.models.ebm_model import EBMModel
from src.models.federated import FederatedSimulator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("🧠 STEP 3: MODEL TRAINING & EVALUATION")
    logger.info("="*60)
    
    Path("models").mkdir(parents=True, exist_ok=True)
    
    input_path = "data/processed/final_features.pkl"
    if not Path(input_path).exists():
        logger.error(f"❌ {input_path} not found! Please run Step 2 first.")
        return
        
    logger.info(f"Loading feature data from {input_path}...")
    df = pd.read_pickle(input_path)
    
    # Print dataset metrics
    logger.info("\n" + "="*60)
    logger.info("📊 DATASET METRICS")
    logger.info("="*60)
    logger.info(f"Total Rows: {df.shape[0]:,}")
    logger.info(f"Total Columns: {df.shape[1]}")
    logger.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    logger.info(f"Number of Buildings: {df['building_id'].nunique()}")
    logger.info(f"Number of Sites: {df['site_id'].nunique()}")
    logger.info("="*60 + "\n")
    
    # Train/Test Split
    logger.info("--> Splitting Train/Test sets (Time-based)...")
    dates = df['timestamp'].sort_values().unique()
    split_date = dates[int(len(dates) * 0.8)]
    
    train_df = df[df['timestamp'] < split_date]
    test_df = df[df['timestamp'] >= split_date]
    
    logger.info("\n" + "="*60)
    logger.info("📊 TRAIN/TEST SPLIT METRICS")
    logger.info("="*60)
    logger.info(f"Train: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    logger.info(f"Test:  {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")
    logger.info(f"Train Rows: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Test Rows:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info("="*60 + "\n")
    
    # Feature Selection
    target = 'meter_reading'
    exclude_cols = ['timestamp', 'meter_reading', 'site_id', 'building_id', 'meter', 
                   'primary_use', 'primary_use_grouped', 'timestamp_weather']
    
    features = [c for c in df.columns if c not in exclude_cols]
    # Ensure numeric
    features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    logger.info(f"    Training with {len(features)} features.")
    logger.info(f"    Feature list: {', '.join(features[:10])}...")
    
    # 1. LightGBM
    logger.info("\n[MODEL 1] LightGBM (Gradient Boosting)")
    lgb_model = LightGBMModel({'n_estimators': 100})
    lgb_model.fit(X_train, y_train, X_test, y_test)
    lgb_metrics = lgb_model.evaluate(X_test, y_test)
    lgb_model.save("models/lgb_model.pkl")
    
    # 2. EBM
    logger.info("\n[MODEL 2] EBM (Explainable Boosting Machine)")
    # Sample for speed
    sample_size = min(10000, len(X_train))
    X_train_sample = X_train.sample(sample_size, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    ebm_model = EBMModel()
    ebm_model.fit(X_train_sample, y_train_sample)
    ebm_metrics = ebm_model.evaluate(X_test, y_test)
    ebm_model.save("models/ebm_model.pkl")
    
    # 3. Federated Simulation
    logger.info("\n[MODEL 3] Federated Learning Simulation")
    fed_cols = features + [target, 'site_id']
    fed_df = df[fed_cols].copy()
    fed_df = fed_df.loc[:, ~fed_df.columns.duplicated()]
    
    fed_sim = FederatedSimulator(model_params={'verbose': -1, 'n_estimators': 50}, n_rounds=2)
    fed_sim.fit(fed_df, target_col=target)
    
    # Save Dashboard Data
    logger.info("\n--> Saving results for Dashboard...")
    results_df = X_test.copy()
    results_df['Actual'] = y_test
    results_df['Predicted_LGBM'] = lgb_model.predict(X_test)
    results_df['Predicted_EBM'] = ebm_model.predict(X_test)
    results_df['timestamp'] = test_df['timestamp']
    results_df['building_id'] = test_df['building_id']
    
    # Save sample for one building
    sample_building = results_df['building_id'].iloc[0]
    dashboard_df = results_df[results_df['building_id'] == sample_building].copy()
    dashboard_df.to_csv("data/processed/dashboard_data.csv", index=False)
    
    # Final Report
    logger.info("\n" + "="*60)
    logger.info("✅ FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"LightGBM:")
    logger.info(f"  RMSLE: {lgb_metrics['RMSLE']:.4f}")
    logger.info(f"  R2 Score: {lgb_metrics['R2']:.4f} ({lgb_metrics['R2']*100:.1f}% Accuracy)")
    logger.info(f"  MAE: {lgb_metrics['MAE']:.2f} kWh")
    logger.info(f"\nEBM:")
    logger.info(f"  RMSLE: {ebm_metrics['RMSLE']:.4f}")
    logger.info(f"  R2 Score: {ebm_metrics['R2']:.4f} ({ebm_metrics['R2']*100:.1f}% Accuracy)")
    logger.info(f"  MAE: {ebm_metrics['MAE']:.2f} kWh")
    logger.info(f"\nFederated Sim: Completed")
    logger.info("="*60)

if __name__ == "__main__":
    main()
