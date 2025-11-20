"""Input/Output utilities for saving and loading data."""

import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict
import pandas as pd


def save_pickle(obj: Any, filepath: str):
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save location
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Dict, filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        obj: Dictionary to save
        filepath: Path to save location
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=indent)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_model(model: Any, filepath: str):
    """
    Save ML model using joblib.
    
    Args:
        model: Trained model
        filepath: Path to save location
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load ML model using joblib.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def save_parquet(df: pd.DataFrame, filepath: str):
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        filepath: Path to save location
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(filepath, index=False, engine='pyarrow')


def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from Parquet format.
    
    Args:
        filepath: Path to parquet file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(filepath, engine='pyarrow')


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False):
    """
    Save DataFrame to CSV format.
    
    Args:
        df: DataFrame to save
        filepath: Path to save location
        index: Whether to save index
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=index)


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from CSV format.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath, **kwargs)


if __name__ == "__main__":
    # Test I/O operations
    print("Testing I/O utilities...")
    
    # Test JSON
    test_dict = {"name": "WattWise-FL", "version": "0.1.0"}
    save_json(test_dict, "test_output.json")
    loaded_dict = load_json("test_output.json")
    print(f"JSON test: {loaded_dict}")
    
    # Test pickle
    test_list = [1, 2, 3, 4, 5]
    save_pickle(test_list, "test_output.pkl")
    loaded_list = load_pickle("test_output.pkl")
    print(f"Pickle test: {loaded_list}")
    
    print("\nI/O tests complete!")

