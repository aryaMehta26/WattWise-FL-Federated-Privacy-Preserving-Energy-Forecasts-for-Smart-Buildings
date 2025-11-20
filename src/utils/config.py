"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this file is in src/utils/
    return Path(__file__).parent.parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_paths(config: Dict[str, Any] = None) -> Dict[str, Path]:
    """
    Get all project paths as Path objects.
    
    Args:
        config: Configuration dictionary. If None, loads default config.
        
    Returns:
        Dictionary of path names to Path objects
    """
    if config is None:
        config = load_config()
    
    project_root = get_project_root()
    paths = {}
    
    for key, value in config['paths'].items():
        paths[key] = project_root / value
    
    return paths


def ensure_directories(config: Dict[str, Any] = None):
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary. If None, loads default config.
    """
    paths = get_paths(config)
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Project: {config['project']['name']} v{config['project']['version']}")
    
    paths = get_paths(config)
    print("\nProject paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Ensure directories exist
    ensure_directories(config)
    print("\nAll directories created/verified!")

