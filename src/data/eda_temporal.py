"""
EDA Implementation 1: Temporal Analysis
Comprehensive time-series analysis of energy consumption patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalEDA:
    """
    Temporal analysis for energy consumption data.
    """
    
    def __init__(self, df: pd.DataFrame, timestamp_col: str = 'timestamp', 
                 value_col: str = 'meter_reading'):
        """
        Initialize temporal EDA.
        
        Args:
            df: Long format dataframe
            timestamp_col: Name of timestamp column
            value_col: Name of value column
        """
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        
        # Parse timestamp
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        
        # Extract temporal features
        self._extract_temporal_features()
        
    def _extract_temporal_features(self):
        """Extract all temporal features from timestamp."""
        self.df['hour'] = self.df[self.timestamp_col].dt.hour
        self.df['day_of_week'] = self.df[self.timestamp_col].dt.dayofweek
        self.df['day_of_month'] = self.df[self.timestamp_col].dt.day
        self.df['month'] = self.df[self.timestamp_col].dt.month
        self.df['quarter'] = self.df[self.timestamp_col].dt.quarter
        self.df['day_of_year'] = self.df[self.timestamp_col].dt.dayofyear
        self.df['week_of_year'] = self.df[self.timestamp_col].dt.isocalendar().week
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        self.df['is_month_start'] = self.df[self.timestamp_col].dt.is_month_start
        self.df['is_month_end'] = self.df[self.timestamp_col].dt.is_month_end
        
    def analyze_hourly_patterns(self) -> Dict:
        """
        Analyze hourly consumption patterns.
        
        Returns:
            Dictionary with hourly statistics
        """
        logger.info("Analyzing hourly patterns...")
        
        hourly_stats = self.df.groupby('hour')[self.value_col].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).to_dict('index')
        
        peak_hour = self.df.groupby('hour')[self.value_col].mean().idxmax()
        low_hour = self.df.groupby('hour')[self.value_col].mean().idxmin()
        
        results = {
            'hourly_stats': hourly_stats,
            'peak_hour': int(peak_hour),
            'low_hour': int(low_hour),
            'peak_consumption': float(self.df.groupby('hour')[self.value_col].mean().max()),
            'low_consumption': float(self.df.groupby('hour')[self.value_col].mean().min()),
            'hourly_variance': float(self.df.groupby('hour')[self.value_col].mean().std())
        }
        
        logger.info(f"Peak hour: {peak_hour}:00")
        logger.info(f"Low hour: {low_hour}:00")
        
        return results
    
    def analyze_daily_patterns(self) -> Dict:
        """
        Analyze daily (day of week) consumption patterns.
        
        Returns:
            Dictionary with daily statistics
        """
        logger.info("Analyzing daily patterns...")
        
        daily_stats = self.df.groupby('day_of_week')[self.value_col].agg([
            'mean', 'median', 'std'
        ]).to_dict('index')
        
        weekday_avg = self.df[~self.df['is_weekend']][self.value_col].mean()
        weekend_avg = self.df[self.df['is_weekend']][self.value_col].mean()
        
        results = {
            'daily_stats': daily_stats,
            'weekday_average': float(weekday_avg),
            'weekend_average': float(weekend_avg),
            'weekend_ratio': float(weekend_avg / weekday_avg) if weekday_avg > 0 else 0
        }
        
        logger.info(f"Weekday average: {weekday_avg:.2f}")
        logger.info(f"Weekend average: {weekend_avg:.2f}")
        
        return results
    
    def analyze_seasonal_patterns(self) -> Dict:
        """
        Analyze seasonal/monthly consumption patterns.
        
        Returns:
            Dictionary with seasonal statistics
        """
        logger.info("Analyzing seasonal patterns...")
        
        monthly_stats = self.df.groupby('month')[self.value_col].agg([
            'mean', 'median', 'std'
        ]).to_dict('index')
        
        quarterly_stats = self.df.groupby('quarter')[self.value_col].agg([
            'mean', 'median', 'std'
        ]).to_dict('index')
        
        peak_month = self.df.groupby('month')[self.value_col].mean().idxmax()
        low_month = self.df.groupby('month')[self.value_col].mean().idxmin()
        
        results = {
            'monthly_stats': monthly_stats,
            'quarterly_stats': quarterly_stats,
            'peak_month': int(peak_month),
            'low_month': int(low_month),
            'seasonal_variance': float(self.df.groupby('month')[self.value_col].mean().std())
        }
        
        logger.info(f"Peak month: {peak_month}")
        logger.info(f"Low month: {low_month}")
        
        return results
    
    def detect_anomalies(self, method: str = 'statistical') -> pd.DataFrame:
        """
        Detect temporal anomalies in consumption.
        
        Args:
            method: Detection method ('statistical' or 'isolation')
            
        Returns:
            Dataframe with anomaly flags
        """
        logger.info(f"Detecting anomalies using {method} method...")
        
        df = self.df.copy()
        
        if method == 'statistical':
            # Z-score method
            hourly_means = df.groupby('hour')[self.value_col].transform('mean')
            hourly_stds = df.groupby('hour')[self.value_col].transform('std')
            z_scores = np.abs((df[self.value_col] - hourly_means) / hourly_stds)
            df['is_anomaly'] = z_scores > 3
        else:
            # Simple threshold based on hourly patterns
            hourly_bounds = df.groupby('hour')[self.value_col].agg(['mean', 'std'])
            hourly_bounds['upper'] = hourly_bounds['mean'] + 3 * hourly_bounds['std']
            hourly_bounds['lower'] = hourly_bounds['mean'] - 3 * hourly_bounds['std']
            
            df = df.merge(hourly_bounds[['upper', 'lower']], left_on='hour', right_index=True)
            df['is_anomaly'] = (df[self.value_col] > df['upper']) | (df[self.value_col] < df['lower'])
            df = df.drop(columns=['upper', 'lower'])
        
        n_anomalies = df['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return df
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive temporal analysis report.
        
        Returns:
            Dictionary with all temporal analysis results
        """
        logger.info("Generating temporal analysis report...")
        
        report = {
            'hourly_analysis': self.analyze_hourly_patterns(),
            'daily_analysis': self.analyze_daily_patterns(),
            'seasonal_analysis': self.analyze_seasonal_patterns(),
            'summary': {
                'total_records': len(self.df),
                'date_range': {
                    'start': str(self.df[self.timestamp_col].min()),
                    'end': str(self.df[self.timestamp_col].max()),
                    'days': (self.df[self.timestamp_col].max() - 
                            self.df[self.timestamp_col].min()).days
                },
                'unique_buildings': self.df['building_id'].nunique() if 'building_id' in self.df.columns else 0
            }
        }
        
        return report

