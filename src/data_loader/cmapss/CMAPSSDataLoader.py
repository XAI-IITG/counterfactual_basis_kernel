from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import json

@dataclass
class PreprocessConfig:
    subset: str = "FD001"
    max_rul: int = 125
    seq_len: int = 50
    low_var_threshold: float = 0.01
    label_mode: str = "scaled"   # "scaled" or "cycles"
    seed: int = 42
    val_ratio: float = 0.2

@dataclass
class PreprocessArtifacts:
    subset: str
    max_rul: int
    seq_len: int
    low_var_threshold: float
    label_mode: str
    feature_cols: list
    removed_features: list
    mean: dict
    std: dict
    train_units: list
    val_units: list

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return PreprocessArtifacts(**d)

class CMAPSSDataLoader:
    """Load and preprocess CMAPSS turbofan engine degradation dataset"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Resolve relative to project root (parent of src/)
            data_path = str(Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'CMAPSS')
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

    @staticmethod
    def make_test_rul_per_row(test_df: pd.DataFrame, rul_true: pd.DataFrame, max_rul: int) -> pd.DataFrame:
        """
        Compute per-row test RUL using:
          EOL(unit) = max_cycle(unit) + RUL_last(unit)
          RUL(row)  = EOL(unit) - cycle(row)
        """
        max_cycles = test_df.groupby("unit_id")["cycle"].max().reset_index()
        max_cycles["RUL_last"] = rul_true["RUL"].values
        max_cycles["EOL"] = max_cycles["cycle"] + max_cycles["RUL_last"]

        out = test_df.merge(max_cycles[["unit_id", "EOL"]], on="unit_id", how="left")
        out["RUL"] = (out["EOL"] - out["cycle"]).clip(upper=max_rul)
        return out.drop(columns=["EOL"])
    
    def fit_preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame, rul_true: pd.DataFrame, cfg: PreprocessConfig):
        # 1) add RUL
        train_df = self.add_rul(train_df, max_rul=cfg.max_rul)
        test_df = self.make_test_rul_per_row(test_df, rul_true, max_rul=cfg.max_rul)

        # 2) split units (NO leakage)
        train_units, val_units = self.split_train_val_units(train_df, cfg.val_ratio, cfg.seed)
        tr = train_df[train_df["unit_id"].isin(train_units)].copy()
        va = train_df[train_df["unit_id"].isin(val_units)].copy()

        # 3) feature cols (all sensors/settings except id/time/label)
        feature_cols = [c for c in tr.columns if c not in ["unit_id", "cycle", "RUL"]]

        # 4) remove constant/low-var based on TRAIN ONLY
        std_vals = tr[feature_cols].std()
        removed = std_vals[std_vals < cfg.low_var_threshold].index.tolist()
        kept = [c for c in feature_cols if c not in removed]

        tr = tr[["unit_id","cycle","RUL"] + kept]
        va = va[["unit_id","cycle","RUL"] + kept]
        te = test_df[["unit_id","cycle","RUL"] + kept]

        # 5) normalise based on TRAIN ONLY
        mean = tr[kept].mean()
        std = tr[kept].std().replace(0.0, 1.0)

        def apply_norm(df):
            out = df.copy()
            out[kept] = (out[kept] - mean) / std
            return out

        trn = apply_norm(tr)
        van = apply_norm(va)
        ten = apply_norm(te)

        artifacts = PreprocessArtifacts(
            subset=cfg.subset,
            max_rul=cfg.max_rul,
            seq_len=cfg.seq_len,
            low_var_threshold=cfg.low_var_threshold,
            label_mode=cfg.label_mode,
            feature_cols=kept,
            removed_features=removed,
            mean=mean.to_dict(),
            std=std.to_dict(),
            train_units=train_units,
            val_units=val_units,
        )
        return trn, van, ten, artifacts
    
    @staticmethod
    def apply_preprocess(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
        kept = artifacts.feature_cols
        mean = pd.Series(artifacts.mean)
        std = pd.Series(artifacts.std).replace(0.0, 1.0)

        out = df[["unit_id","cycle","RUL"] + kept].copy()
        out[kept] = (out[kept] - mean[kept]) / std[kept]
        return out
