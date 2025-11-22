"""
EDA Implementation 2: Building-Level Analysis
Analysis of building characteristics and consumption profiles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BuildingEDA:
    """
    Building-level analysis for energy consumption data.
    """
    
    def __init__(self, df: pd.DataFrame, building_id_col: str = 'building_id',
                 value_col: str = 'meter_reading'):
        """
        Initialize building EDA.
        
        Args:
            df: Long format dataframe
            building_id_col: Name of building ID column
            value_col: Name of value column
        """
        self.df = df.copy()
        self.building_id_col = building_id_col
        self.value_col = value_col
        
    def parse_building_metadata(self) -> pd.DataFrame:
        """
        Parse building metadata from building IDs.
        Assumes format: Site_UseType_BuildingName
        
        Returns:
            DataFrame with parsed metadata
        """
        logger.info("Parsing building metadata...")
        
        building_ids = self.df[self.building_id_col].unique()
        metadata = []
        
        for bid in building_ids:
            parts = str(bid).split('_')
            if len(parts) >= 3:
                metadata.append({
                    'building_id': bid,
                    'site': parts[0],
                    'use_type': parts[1],
                    'building_name': '_'.join(parts[2:])
                })
            else:
                metadata.append({
                    'building_id': bid,
                    'site': 'unknown',
                    'use_type': 'unknown',
                    'building_name': bid
                })
        
        meta_df = pd.DataFrame(metadata)
        logger.info(f"Parsed metadata for {len(meta_df)} buildings")
        logger.info(f"Sites: {meta_df['site'].nunique()}")
        logger.info(f"Use types: {meta_df['use_type'].nunique()}")
        
        return meta_df
    
    def calculate_building_statistics(self) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each building.
        
        Returns:
            DataFrame with building statistics
        """
        logger.info("Calculating building statistics...")
        
        stats = self.df.groupby(self.building_id_col)[self.value_col].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('count', 'count'),
            ('missing_count', lambda x: x.isna().sum())
        ]).reset_index()
        
        stats['missing_pct'] = (stats['missing_count'] / stats['count']) * 100
        stats['cv'] = stats['std'] / stats['mean']  # Coefficient of variation
        stats['range'] = stats['max'] - stats['min']
        stats['iqr'] = stats['q75'] - stats['q25']
        
        logger.info(f"Calculated statistics for {len(stats)} buildings")
        
        return stats
    
    def analyze_by_use_type(self, meta_df: pd.DataFrame, 
                           stats_df: pd.DataFrame) -> Dict:
        """
        Analyze consumption patterns by building use type.
        
        Args:
            meta_df: Building metadata dataframe
            stats_df: Building statistics dataframe
            
        Returns:
            Dictionary with use type analysis
        """
        logger.info("Analyzing by use type...")
        
        combined = stats_df.merge(meta_df, on='building_id', how='left')
        
        use_type_analysis = combined.groupby('use_type').agg({
            'mean': ['mean', 'std', 'count'],
            'median': 'mean',
            'cv': 'mean'
        })
        
        results = {
            'use_type_stats': use_type_analysis.to_dict(),
            'use_type_counts': combined['use_type'].value_counts().to_dict(),
            'highest_consumption_type': combined.groupby('use_type')['mean'].mean().idxmax(),
            'lowest_consumption_type': combined.groupby('use_type')['mean'].mean().idxmin()
        }
        
        logger.info(f"Highest consumption type: {results['highest_consumption_type']}")
        logger.info(f"Lowest consumption type: {results['lowest_consumption_type']}")
        
        return results
    
    def analyze_by_site(self, meta_df: pd.DataFrame,
                       stats_df: pd.DataFrame) -> Dict:
        """
        Analyze consumption patterns by site.
        
        Args:
            meta_df: Building metadata dataframe
            stats_df: Building statistics dataframe
            
        Returns:
            Dictionary with site analysis
        """
        logger.info("Analyzing by site...")
        
        combined = stats_df.merge(meta_df, on='building_id', how='left')
        
        site_analysis = combined.groupby('site').agg({
            'mean': ['mean', 'std', 'count'],
            'median': 'mean'
        })
        
        results = {
            'site_stats': site_analysis.to_dict(),
            'site_counts': combined['site'].value_counts().to_dict(),
            'highest_consumption_site': combined.groupby('site')['mean'].mean().idxmax(),
            'lowest_consumption_site': combined.groupby('site')['mean'].mean().idxmin()
        }
        
        logger.info(f"Highest consumption site: {results['highest_consumption_site']}")
        logger.info(f"Lowest consumption site: {results['lowest_consumption_site']}")
        
        return results
    
    def identify_high_variance_buildings(self, stats_df: pd.DataFrame,
                                        threshold: float = 1.0) -> List[str]:
        """
        Identify buildings with high consumption variance.
        
        Args:
            stats_df: Building statistics dataframe
            threshold: Coefficient of variation threshold
            
        Returns:
            List of building IDs with high variance
        """
        high_variance = stats_df[stats_df['cv'] > threshold]
        building_ids = high_variance[self.building_id_col].tolist()
        
        logger.info(f"Found {len(building_ids)} buildings with CV > {threshold}")
        
        return building_ids
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive building-level analysis report.
        
        Returns:
            Dictionary with all building analysis results
        """
        logger.info("Generating building-level analysis report...")
        
        meta_df = self.parse_building_metadata()
        stats_df = self.calculate_building_statistics()
        
        report = {
            'building_statistics': stats_df.to_dict('records'),
            'use_type_analysis': self.analyze_by_use_type(meta_df, stats_df),
            'site_analysis': self.analyze_by_site(meta_df, stats_df),
            'high_variance_buildings': self.identify_high_variance_buildings(stats_df),
            'summary': {
                'total_buildings': len(stats_df),
                'unique_sites': meta_df['site'].nunique(),
                'unique_use_types': meta_df['use_type'].nunique(),
                'avg_consumption': float(stats_df['mean'].mean()),
                'median_consumption': float(stats_df['median'].median())
            }
        }
        
        return report

