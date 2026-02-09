from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np

class CMAPSSDataLoader:
    """Load and preprocess CMAPSS turbofan engine degradation dataset"""
    
    def __init__(self, data_path: str = '../data/CMAPSS'):
        self.data_path = Path(data_path)
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        
    def load_dataset(self, subset: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load CMAPSS dataset
        Args:
            subset: 'FD001', 'FD002', 'FD003', or 'FD004'
        Returns:
            train_df, test_df, rul_true
        """
        # Column names
        cols = ['unit_id', 'cycle'] + self.setting_cols + self.sensor_cols
        
        # Load train and test data
        train_file = self.data_path / f'train_{subset}.txt'
        test_file = self.data_path / f'test_{subset}.txt'
        rul_file = self.data_path / f'RUL_{subset}.txt'
        
        train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols)
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols)
        rul_true = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
        
        print(f"Loaded {subset}:")
        print(f"  Training samples: {len(train_df)}, Units: {train_df['unit_id'].nunique()}")
        print(f"  Test samples: {len(test_df)}, Units: {test_df['unit_id'].nunique()}")
        
        return train_df, test_df, rul_true
    
    def add_rul(self, df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
        """Add Remaining Useful Life (RUL) column"""
        df_rul = df.copy()
        # Calculate RUL for each unit
        max_cycles = df_rul.groupby('unit_id')['cycle'].max()
        df_rul = df_rul.merge(max_cycles.to_frame(name='max_cycle'), 
                               left_on='unit_id', right_index=True)
        df_rul['RUL'] = df_rul['max_cycle'] - df_rul['cycle']
        df_rul = df_rul.drop('max_cycle', axis=1)
        
        # Apply piecewise linear RUL (common in literature)
        df_rul['RUL'] = df_rul['RUL'].clip(upper=max_rul)
        
        return df_rul
    
    def remove_constant_features(self, train_df: pd.DataFrame, 
                                  test_df: pd.DataFrame, 
                                  threshold: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Remove features with low variance"""
        feature_cols = self.setting_cols + self.sensor_cols
        
        # Calculate standard deviation
        std_vals = train_df[feature_cols].std()
        
        # Keep features with std > threshold
        features_to_keep = std_vals[std_vals > threshold].index.tolist()
        features_removed = [f for f in feature_cols if f not in features_to_keep]
        
        print(f"Removed {len(features_removed)} constant/low-variance features: {features_removed}")
        print(f"Kept {len(features_to_keep)} features")
        
        # Keep unit_id, cycle, RUL and selected features
        keep_cols = ['unit_id', 'cycle'] + features_to_keep
        if 'RUL' in train_df.columns:
            keep_cols.append('RUL')
        
        return train_df[keep_cols], test_df[[c for c in keep_cols if c in test_df.columns]], features_to_keep
    
    def normalize_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       feature_cols: List[str]) -> Tuple:
        """Normalize sensor data using train statistics"""
        
        # Calculate statistics from training data
        mean = train_df[feature_cols].mean()
        std = train_df[feature_cols].std()
        
        # Normalize
        train_norm = train_df.copy()
        test_norm = test_df.copy()
        
        train_norm[feature_cols] = (train_df[feature_cols] - mean) / (std + 1e-8)
        test_norm[feature_cols] = (test_df[feature_cols] - mean) / (std + 1e-8)
        
        print(f"Normalized {len(feature_cols)} features")
        
        return train_norm, test_norm, mean, std

