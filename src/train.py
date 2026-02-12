# src/train.py (or wherever your training script lives)
from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.cmapss.CMAPSSDataLoader import CMAPSSDataLoader
from src.data_loader.cmapss.CMAPSSTimeSeriesDataset import CMAPSSTimeSeriesDataset
from src.data_loader.cmapss.CMAPSSDatasetWrapper import CMAPSSDatasetWrapper

from src.models.LSTMModel import LSTMModel
from src.models.GRUModel import GRUModel
from src.models.CNNLSTMModel import CNNLSTMModel
from src.models.TransformerModel import TransformerModel

from src.trainer.Trainer import Trainer, TrainingConfig


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    #save the preprocess dict in json
    with open(Path(__file__).parent.parent / "outputs" / "preprocess.json", "w") as f:
        import json
        json.dump(preprocess, f, indent=4)


    return train_loader, val_loader, test_loader, preprocess


def create_model(model_type: str, input_size: int, device: torch.device):
    if model_type == "lstm":
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "gru":
        model = GRUModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "cnn_lstm":
        model = CNNLSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "transformer":
        model = TransformerModel(input_size=input_size, d_model=128, nhead=4, num_layers=2, dropout=0.2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    print(f"{model_type.upper()} params: {sum(p.numel() for p in model.parameters()):,}")
    return model


def train_single_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocess: Dict[str, Any],
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    save_path: str,
):
    input_size = len(preprocess["feature_cols"])
    model = create_model(model_type, input_size, device)

    cfg = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=train_loader.batch_size,
        early_stopping_patience=25,
        early_stopping_start_epoch=80,
        gradient_clip_value=1.0,
        save_path=save_path,
        model_name=f"{model_type}_best",  # produces *_best.pth and *_best.ckpt
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg,
        preprocess=preprocess,
    )

    history = trainer.train()
    return history, trainer


def main():
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu")
    print(f"\nUsing device: {device}")

    train_loader, val_loader, test_loader, preprocess = load_and_preprocess_data(
        subset="FD001",
        seq_len=50,
        max_rul=125,
        batch_size=32,
        seed=42,
        val_ratio=0.2,
        var_threshold=0.01,
    )

    save_path = str(Path(__file__).parent.parent / "outputs" / "saved_models")
    os.makedirs(save_path, exist_ok=True)

    models_config = {
        #"lstm": {"lr": 1e-3, "epochs": 160},
        #"gru": {"lr": 1e-3, "epochs": 160},
        #"cnn_lstm": {"lr": 1e-3, "epochs": 160},
        "transformer": {"lr": 5e-4, "epochs": 160},
    }

    histories = {}
    trainers = {}

    for model_type, mc in models_config.items():
        print(f"\n{'#'*80}\n# Training {model_type.upper()}\n{'#'*80}")
        history, trainer = train_single_model(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,          # ✅ REAL validation set (not test)
            preprocess=preprocess,          # ✅ saved into checkpoint
            device=device,                  # ✅ torch.device, not a string
            num_epochs=mc["epochs"],
            learning_rate=mc["lr"],
            save_path=save_path,
        )
        histories[model_type] = history
        trainers[model_type] = trainer

    print("\nTraining done. Best checkpoints written with preprocess metadata embedded.")


if __name__ == "__main__":
    main()
