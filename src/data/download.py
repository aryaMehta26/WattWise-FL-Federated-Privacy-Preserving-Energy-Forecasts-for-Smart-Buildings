"""Download Building Data Genome 2 (BDG2) dataset."""

import os
import requests
from pathlib import Path
from typing import Optional, List
import time
from tqdm import tqdm
import hashlib
from datetime import datetime
import pandas as pd

from ..utils.config import load_config, get_paths
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# BDG2 file URLs (GitHub raw content)
BDG2_FILES = {
    'metadata': {
        'url': 'https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv',
        'filename': 'metadata.csv'
    },
    'weather': {
        'url': 'https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master/data/weather/weather.csv',
        'filename': 'weather.csv'
    },
}

# Note: Meter data files are large and hosted elsewhere
# We'll provide instructions to download them manually or via Kaggle API


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        filepath: Local path to save file
        chunk_size: Size of chunks to download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def compute_checksum(filepath: Path) -> str:
    """
    Compute SHA256 checksum of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Hexadecimal checksum string
    """
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def log_provenance(
    filepath: Path,
    url: str,
    checksum: str,
    provenance_file: Path
):
    """
    Log data provenance information.
    
    Args:
        filepath: Path to downloaded file
        url: Source URL
        checksum: File checksum
        provenance_file: Path to provenance log
    """
    provenance_entry = {
        'filename': filepath.name,
        'url': url,
        'checksum_sha256': checksum,
        'download_timestamp_utc': datetime.utcnow().isoformat(),
        'file_size_bytes': filepath.stat().st_size,
    }
    
    # Append to provenance log
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    
    if provenance_file.exists():
        with open(provenance_file, 'r') as f:
            import json
            provenance_data = json.load(f)
    else:
        provenance_data = {'downloads': []}
    
    provenance_data['downloads'].append(provenance_entry)
    
    with open(provenance_file, 'w') as f:
        import json
        json.dump(provenance_data, f, indent=2)
    
    logger.info(f"Logged provenance for {filepath.name}")


def download_bdg2_data(
    output_dir: Optional[Path] = None,
    files_to_download: Optional[List[str]] = None,
    force_redownload: bool = False
) -> dict:
    """
    Download BDG2 dataset files.
    
    Args:
        output_dir: Directory to save files. If None, uses config default.
        files_to_download: List of file keys to download. If None, downloads all.
        force_redownload: If True, redownload even if file exists.
        
    Returns:
        Dictionary with download status and file paths
    """
    # Load configuration
    config = load_config()
    
    if output_dir is None:
        paths = get_paths(config)
        output_dir = paths['raw_data']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    provenance_file = output_dir / 'provenance.json'
    
    # Determine which files to download
    if files_to_download is None:
        files_to_download = list(BDG2_FILES.keys())
    
    results = {'successful': [], 'failed': [], 'skipped': []}
    
    logger.info(f"Starting BDG2 data download to {output_dir}")
    logger.info(f"Files to download: {files_to_download}")
    
    for file_key in files_to_download:
        if file_key not in BDG2_FILES:
            logger.warning(f"Unknown file key: {file_key}")
            continue
        
        file_info = BDG2_FILES[file_key]
        url = file_info['url']
        filename = file_info['filename']
        filepath = output_dir / filename
        
        # Check if file already exists
        if filepath.exists() and not force_redownload:
            logger.info(f"File already exists: {filename} (use force_redownload=True to override)")
            results['skipped'].append(filename)
            continue
        
        logger.info(f"Downloading {filename}...")
        success = download_file(url, filepath)
        
        if success:
            # Compute checksum and log provenance
            checksum = compute_checksum(filepath)
            log_provenance(filepath, url, checksum, provenance_file)
            
            logger.info(f"✓ Successfully downloaded {filename}")
            logger.info(f"  Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
            logger.info(f"  SHA256: {checksum[:16]}...")
            
            results['successful'].append(filename)
        else:
            logger.error(f"✗ Failed to download {filename}")
            results['failed'].append(filename)
        
        # Be nice to the server
        time.sleep(0.5)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Download Summary:")
    logger.info(f"  Successful: {len(results['successful'])}")
    logger.info(f"  Failed: {len(results['failed'])}")
    logger.info(f"  Skipped: {len(results['skipped'])}")
    logger.info("="*50)
    
    # Provide instructions for meter data
    logger.info("\n" + "="*50)
    logger.info("IMPORTANT: Meter Data Download Instructions")
    logger.info("="*50)
    logger.info("\nThe meter data files are large and need to be downloaded separately.")
    logger.info("\nOption 1: Download from Kaggle (Recommended)")
    logger.info("  1. Install Kaggle API: pip install kaggle")
    logger.info("  2. Set up Kaggle credentials: https://www.kaggle.com/docs/api")
    logger.info("  3. Download ASHRAE Great Energy Predictor III dataset:")
    logger.info("     kaggle competitions download -c ashrae-energy-prediction")
    logger.info(f"  4. Extract files to: {output_dir}")
    logger.info("\nOption 2: Download from GitHub (if available)")
    logger.info("  Visit: https://github.com/buds-lab/building-data-genome-project-2")
    logger.info("  Download meter CSV files from data/meters/ directory")
    logger.info(f"  Place them in: {output_dir}")
    logger.info("\nRequired meter files:")
    logger.info("  - electricity.csv (or train.csv from Kaggle)")
    logger.info("  - Optional: chilledwater.csv, hotwater.csv, steam.csv")
    
    return results


def verify_downloads(data_dir: Optional[Path] = None) -> bool:
    """
    Verify that required data files exist.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        True if all required files exist, False otherwise
    """
    if data_dir is None:
        config = load_config()
        paths = get_paths(config)
        data_dir = paths['raw_data']
    
    data_dir = Path(data_dir)
    
    required_files = ['metadata.csv', 'weather.csv']
    optional_files = ['train.csv', 'electricity.csv']  # Either Kaggle or BDG2 format
    
    logger.info("Verifying downloads...")
    
    all_required_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            logger.info(f"✓ Found: {filename}")
        else:
            logger.error(f"✗ Missing: {filename}")
            all_required_exist = False
    
    # Check if at least one meter file exists
    meter_file_exists = False
    for filename in optional_files:
        filepath = data_dir / filename
        if filepath.exists():
            logger.info(f"✓ Found meter data: {filename}")
            meter_file_exists = True
            break
    
    if not meter_file_exists:
        logger.warning("⚠ No meter data file found (train.csv or electricity.csv)")
        logger.warning("  Please download meter data manually (see instructions above)")
    
    return all_required_exist and meter_file_exists


if __name__ == "__main__":
    """Run download as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BDG2 dataset")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('--force', action='store_true', help="Force redownload existing files")
    
    args = parser.parse_args()
    
    # Download data
    results = download_bdg2_data(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        force_redownload=args.force
    )
    
    # Verify downloads
    verify_downloads(
        data_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    logger.info("\nDownload complete!")
    logger.info("Next steps:")
    logger.info("  1. Download meter data (see instructions above)")
    logger.info("  2. Run preprocessing: python -m src.data.preprocessing")
    logger.info("  3. Explore data: jupyter notebook notebooks/02_eda_metadata.ipynb")

