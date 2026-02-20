import torch
import numpy as np
from typing import Tuple, List, Set

from torch.utils.data import DataLoader
from src.data_loader.cmapss.CMAPSSDataLoader import CMAPSSDataLoader
from src.data_loader.cmapss.CMAPSSTimeSeriesDataset import CMAPSSTimeSeriesDataset
from src.data_loader.cmapss.CMAPSSDatasetWrapper import CMAPSSDatasetWrapper
from src.data_loader.cmapss.CMAPSSTorchDataloader import CMAPSSTorchDataloader


def split_units(unit_ids: np.ndarray, val_ratio: float = 0.2, seed: int = 42) -> Tuple[Set[int], Set[int]]:
    rng = np.random.default_rng(seed)
    u = np.array(sorted(unit_ids), dtype=int)
    rng.shuffle(u)
    n_val = int(round(len(u) * val_ratio))
    val_units = set(map(int, u[:n_val]))
    train_units = set(map(int, u[n_val:]))
    return train_units, val_units


def add_test_rul_rowwise(test_df, rul_true, max_rul: int):
    # Correct per-row RUL computation (your file already had the correct version here). :contentReference[oaicite:12]{index=12}
    max_cycles = test_df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles["RUL_last"] = rul_true["RUL"].values
    max_cycles["EOL"] = max_cycles["cycle"] + max_cycles["RUL_last"]

    test_df_with_rul = test_df.merge(max_cycles[["unit_id", "EOL"]], on="unit_id", how="left")
    test_df_with_rul["RUL"] = (test_df_with_rul["EOL"] - test_df_with_rul["cycle"]).clip(upper=max_rul)
    return test_df_with_rul.drop(columns=["EOL"])


def remove_low_variance(train_df, val_df, test_df, feature_cols: List[str], threshold: float):
    std_vals = train_df[feature_cols].std()
    kept = std_vals[std_vals > threshold].index.tolist()
    removed = [c for c in feature_cols if c not in kept]
    print(f"Removed {len(removed)} low-variance features: {removed}")
    print(f"Kept {len(kept)} features")
    return kept


def standardise(train_df, val_df, test_df, feature_cols: List[str]):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()

    def norm(df):
        out = df.copy()
        out[feature_cols] = (df[feature_cols] - mean) / (std + 1e-8)
        return out

    return norm(train_df), norm(val_df), norm(test_df), mean, std


def make_last_window_arrays(df_norm, seq_len: int, feature_cols: List[str]):
    Xs, ys = [], []
    for uid in df_norm["unit_id"].unique():
        u = df_norm[df_norm["unit_id"] == uid].sort_values("cycle")
        X = u[feature_cols].values
        y = u["RUL"].values
        if len(X) >= seq_len:
            Xs.append(X[-seq_len:])
            ys.append(y[-1])
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)


def load_and_preprocess_data(
    subset: str = "FD001",
    seq_len: int = 50,
    max_rul: int = 125,
    batch_size: int = 32,
    seed: int = 42,
    val_ratio: float = 0.2,
    var_threshold: float = 0.01,
):
    print(f"\n{'='*80}\nLoading + Preprocessing {subset}\n{'='*80}")

    loader = CMAPSSDataLoader()
    train_df, test_df, rul_true = loader.load_dataset(subset)

    train_df = loader.add_rul(train_df, max_rul=max_rul)
    test_df = add_test_rul_rowwise(test_df, rul_true, max_rul=max_rul)

    train_units, val_units = split_units(train_df["unit_id"].unique(), val_ratio=val_ratio, seed=seed)
    train_split = train_df[train_df["unit_id"].isin(train_units)].copy()
    val_split = train_df[train_df["unit_id"].isin(val_units)].copy()

    # Start from canonical CMAPSS feature set (settings + sensors)
    feature_cols_all = loader.setting_cols + loader.sensor_cols

    # Fit feature selection only on *train_split* to prevent leakage
    kept_features = remove_low_variance(train_split, val_split, test_df, feature_cols_all, threshold=var_threshold)

    keep_cols = ["unit_id", "cycle"] + kept_features + ["RUL"]
    train_split = train_split[keep_cols]
    val_split = val_split[keep_cols]
    test_df = test_df[keep_cols]

    # Standardise using train_split stats only
    train_norm, val_norm, test_norm, mean, std = standardise(train_split, val_split, test_df, kept_features)

    # Train dataset: sliding windows over train_norm
    train_dataset = CMAPSSTimeSeriesDataset(train_norm, sequence_length=seq_len, feature_cols=kept_features, label_mode="cycles", max_rul=max_rul)

    # Val dataset: sliding windows (train data is run-to-failure, so last window always has RUL=0)
    val_dataset = CMAPSSTimeSeriesDataset(val_norm, sequence_length=seq_len, feature_cols=kept_features, label_mode="cycles", max_rul=max_rul)

    # Test: last window per unit (test trajectories are truncated, not run to failure)
    Xt, yt = make_last_window_arrays(test_norm, seq_len, kept_features)
    test_dataset = CMAPSSDatasetWrapper(Xt, yt, label_mode="cycles", max_rul=max_rul)

    # Reproducible shuffling:
    # Use generator/worker init if you later increase num_workers (PyTorch reproducibility guidance). :contentReference[oaicite:14]{index=14}
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    preprocess = {
        "subset": subset,
        "seq_len": int(seq_len),
        "max_rul": int(max_rul),
        "var_threshold": float(var_threshold),
        "feature_cols": kept_features,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "label_mode": "cycles",  # IMPORTANT: model outputs cycles directly
        "seed": int(seed),
        "val_ratio": float(val_ratio),
    }

    print(f"\nFinal feature_dim = {len(kept_features)}")
    print(f"Train batches = {len(train_loader)} | Val batches = {len(val_loader)} | Test batches = {len(test_loader)}")

    return train_loader, val_loader, test_loader, preprocess, train_norm, test_norm, val_norm