"""Federated Learning Simulation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import copy
from tqdm import tqdm

from .lightgbm_model import LightGBMModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class FederatedSimulator:
    def __init__(
        self, 
        model_params: Dict, 
        n_rounds: int = 5,
        fraction_fit: float = 1.0
    ):
        """
        Initialize Federated Learning Simulator.
        
        Args:
            model_params: Parameters for the local models
            n_rounds: Number of FL rounds
            fraction_fit: Fraction of clients to train in each round
        """
        self.model_params = model_params
        self.n_rounds = n_rounds
        self.fraction_fit = fraction_fit
        self.global_model = None
        self.round_metrics = []
        
    def fit(self, df: pd.DataFrame, target_col: str, client_col: str = 'site_id'):
        """
        Run FL simulation.
        
        Args:
            df: Full dataframe containing all clients
            target_col: Target variable name
            client_col: Column identifying clients (e.g., site_id)
        """
        logger.info(f"Starting FL Simulation: {self.n_rounds} rounds, {df[client_col].nunique()} clients")
        
        # 1. Initialize Global Model (conceptually)
        # In this simplified simulation using LightGBM, we can't easily average tree structures.
        # Instead, we will simulate "Ensemble Distillation" or simply train local models 
        # and average their predictions (which is a valid FL strategy for trees).
        # OR, for a true FL simulation with trees, we would need GBDT-specific FL algorithms.
        #
        # APPROACH FOR THIS PROJECT:
        # We will implement "FedAvg" logic where we train local models and then 
        # aggregate their PREDICTIONS on a test set to show the "Global" performance,
        # OR we simply train one model per site and report the average performance.
        #
        # However, to make it "Federated", we should simulate the iterative process.
        # Since averaging LightGBM trees is hard, we will use a simpler proxy:
        # We will train a model on each site's data, and the "Global Model" is the collection of these.
        
        clients = df[client_col].unique()
        client_models = {}
        
        # Split features and target
        feature_cols = [c for c in df.columns if c not in [target_col, 'timestamp', client_col]]
        
        # Simulation Loop
        for round_idx in range(1, self.n_rounds + 1):
            logger.info(f"--- Round {round_idx}/{self.n_rounds} ---")
            
            # Select clients
            n_clients = max(1, int(len(clients) * self.fraction_fit))
            selected_clients = np.random.choice(clients, n_clients, replace=False)
            
            round_losses = []
            
            for client_id in tqdm(selected_clients, desc="Training Clients"):
                # Get client data (Simulating local data access)
                client_data = df[df[client_col] == client_id]
                
                # Train/Test split for this client (simple time split)
                split_idx = int(len(client_data) * 0.8)
                train_data = client_data.iloc[:split_idx]
                val_data = client_data.iloc[split_idx:]
                
                X_train = train_data[feature_cols]
                y_train = train_data[target_col]
                X_val = val_data[feature_cols]
                y_val = val_data[target_col]
                
                # Train Local Model
                model = LightGBMModel(self.model_params)
                model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate Local Model
                metrics = model.evaluate(X_val, y_val)
                round_losses.append(metrics['RMSLE'])
                
                # Store model (Simulating sending update to server)
                client_models[client_id] = model
            
            # Aggregate metrics
            avg_loss = np.mean(round_losses)
            self.round_metrics.append({'round': round_idx, 'avg_rmsle': avg_loss})
            logger.info(f"Round {round_idx} Average RMSLE: {avg_loss:.4f}")
            
        self.global_model = client_models # In this tree-based proxy, the "global" model is the ensemble of locals
        logger.info("FL Simulation Complete.")
        
    def predict(self, X: pd.DataFrame, client_col: str = 'site_id') -> np.ndarray:
        """
        Make predictions using the federated ensemble.
        """
        preds = []
        # For each row, use the corresponding client model
        # This assumes we know which site the data belongs to (Personalized FL)
        
        # Optimization: Group by site to predict in batches
        X = X.copy()
        X['pred'] = np.nan
        
        for client_id, model in self.global_model.items():
            mask = X[client_col] == client_id
            if mask.any():
                client_X = X.loc[mask].drop(columns=[client_col, 'pred'], errors='ignore')
                # Ensure columns match
                model_features = model.model.feature_name()
                client_X = client_X[model_features]
                
                X.loc[mask, 'pred'] = model.predict(client_X)
                
        return X['pred'].values
