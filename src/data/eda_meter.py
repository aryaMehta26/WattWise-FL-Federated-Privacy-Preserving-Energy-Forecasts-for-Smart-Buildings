"""
EDA Implementation 3: Meter-Level Analysis
Cross-meter type comparisons and energy distribution analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MeterEDA:
    """
    Meter-level analysis for energy consumption data.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize meter EDA.
        
        Args:
            data_dir: Directory containing meter data files
        """
        self.data_dir = Path(data_dir)
        self.meter_data = {}
        
    def load_meter_types(self, meter_types: Optional[List[str]] = None) -> Dict:
        """
        Load multiple meter types for comparison.
        
        Args:
            meter_types: List of meter types to load (e.g., ['electricity', 'chilledwater'])
            
        Returns:
            Dictionary with loaded meter dataframes
        """
        if meter_types is None:
            meter_types = ['electricity', 'chilledwater', 'hotwater', 'steam']
        
        logger.info(f"Loading meter types: {meter_types}")
        
        for meter_type in meter_types:
            file_path = self.data_dir / f'{meter_type}.txt'
            if not file_path.exists():
                file_path = self.data_dir / f'{meter_type}.csv'
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, nrows=10000)  # Sample for speed
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        self.meter_data[meter_type] = df
                        logger.info(f"Loaded {meter_type}: {df.shape}")
                except Exception as e:
                    logger.warning(f"Could not load {meter_type}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return self.meter_data
    
    def compare_meter_statistics(self) -> pd.DataFrame:
        """
        Compare statistics across different meter types.
        
        Returns:
            DataFrame with meter comparison statistics
        """
        logger.info("Comparing meter statistics...")
        
        comparison_stats = []
        
        for meter_type, df in self.meter_data.items():
            meter_cols = [c for c in df.columns if c != 'timestamp']
            
            # Calculate statistics
            all_values = df[meter_cols].values.flatten()
            all_values = all_values[~np.isnan(all_values)]
            
            if len(all_values) > 0:
                stats = {
                    'meter_type': meter_type,
                    'n_buildings': len(meter_cols),
                    'n_readings': len(df),
                    'mean': float(np.mean(all_values)),
                    'median': float(np.median(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'q25': float(np.percentile(all_values, 25)),
                    'q75': float(np.percentile(all_values, 75))
                }
                comparison_stats.append(stats)
        
        comparison_df = pd.DataFrame(comparison_stats)
        logger.info(f"Compared {len(comparison_df)} meter types")
        
        return comparison_df
    
    def analyze_meter_distributions(self) -> Dict:
        """
        Analyze distribution characteristics for each meter type.
        
        Returns:
            Dictionary with distribution analysis
        """
        logger.info("Analyzing meter distributions...")
        
        distributions = {}
        
        for meter_type, df in self.meter_data.items():
            meter_cols = [c for c in df.columns if c != 'timestamp']
            all_values = df[meter_cols].values.flatten()
            all_values = all_values[~np.isnan(all_values)]
            
            if len(all_values) > 0:
                distributions[meter_type] = {
                    'skewness': float(pd.Series(all_values).skew()),
                    'kurtosis': float(pd.Series(all_values).kurtosis()),
                    'cv': float(np.std(all_values) / np.mean(all_values)),
                    'zero_pct': float((all_values == 0).sum() / len(all_values) * 100)
                }
        
        return distributions
    
    def identify_correlated_meters(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated meter types (if buildings have multiple meters).
        
        Args:
            threshold: Correlation threshold
            
        Returns:
            List of tuples (meter1, meter2, correlation)
        """
        logger.info("Identifying correlated meters...")
        
        # This would require matching buildings across meter types
        # Simplified version for demonstration
        correlations = []
        
        meter_types = list(self.meter_data.keys())
        for i, meter1 in enumerate(meter_types):
            for meter2 in meter_types[i+1:]:
                # Sample-based correlation (simplified)
                df1 = self.meter_data[meter1]
                df2 = self.meter_data[meter2]
                
                # Find common buildings
                cols1 = set([c for c in df1.columns if c != 'timestamp'])
                cols2 = set([c for c in df2.columns if c != 'timestamp'])
                common_buildings = cols1.intersection(cols2)
                
                if len(common_buildings) > 0:
                    # Calculate average correlation
                    corr_values = []
                    for building in list(common_buildings)[:10]:  # Sample
                        if building in df1.columns and building in df2.columns:
                            merged = df1[['timestamp', building]].merge(
                                df2[['timestamp', building]], 
                                on='timestamp', 
                                suffixes=('_1', '_2')
                            )
                            if len(merged) > 0:
                                corr = merged[f'{building}_1'].corr(merged[f'{building}_2'])
                                if not np.isnan(corr):
                                    corr_values.append(corr)
                    
                    if len(corr_values) > 0:
                        avg_corr = np.mean(corr_values)
                        if avg_corr > threshold:
                            correlations.append((meter1, meter2, float(avg_corr)))
        
        logger.info(f"Found {len(correlations)} highly correlated meter pairs")
        
        return correlations
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive meter-level analysis report.
        
        Returns:
            Dictionary with all meter analysis results
        """
        logger.info("Generating meter-level analysis report...")
        
        if len(self.meter_data) == 0:
            self.load_meter_types()
        
        report = {
            'meter_comparison': self.compare_meter_statistics().to_dict('records'),
            'distributions': self.analyze_meter_distributions(),
            'correlations': self.identify_correlated_meters(),
            'summary': {
                'meter_types_analyzed': list(self.meter_data.keys()),
                'total_buildings': sum([len([c for c in df.columns if c != 'timestamp']) 
                                       for df in self.meter_data.values()])
            }
        }
        
        return report

