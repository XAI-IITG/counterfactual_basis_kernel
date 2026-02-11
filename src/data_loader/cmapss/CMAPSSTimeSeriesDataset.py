from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class CMAPSSTimeSeriesDataset(Dataset):
    """PyTorch Dataset for sequence data"""
    
    def __init__(self, df: pd.DataFrame, sequence_length: int = 50, 
                 feature_cols: List[str] = None):
        self.sequence_length = sequence_length
        
        if feature_cols is None:
            self.feature_cols = [c for c in df.columns if c.startswith(('sensor_', 'setting_'))]
        else:
            self.feature_cols = feature_cols
        
        # Group by unit_id
        self.sequences = []
        self.targets = []
        self.unit_ids = []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
            features = unit_data[self.feature_cols].values
            rul = unit_data['RUL'].values
            
            # Create sliding windows
            for i in range(len(features) - sequence_length + 1):
                self.sequences.append(features[i:i+sequence_length])
                self.targets.append(rul[i+sequence_length-1])
                self.unit_ids.append(unit_id)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.unit_ids = np.array(self.unit_ids, dtype=np.int32)
        
        print(f"Created dataset with {len(self.sequences)} sequences")
        print(f"Sequence shape: {self.sequences.shape}")
        print(f"Feature dimension: {len(self.feature_cols)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])