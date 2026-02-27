"""
train_ieee_phm.py — Training script for IEEE PHM 2012 Bearing RUL prediction.

Loads preprocessed data produced by the notebook:
    notebooks/ieee_phm/01_IEEE_PHM_Data_Preprocess.ipynb

Expected files in data/processed/ieee_phm/:
    - ieee_phm_sequences.npz   (X_train, y_train, X_val, y_val, X_test, y_test)
    - scaler.pkl               (MinMaxScaler fitted on training features)
    - hyperparams.json          (preprocessing hyperparameters)

Models are saved under outputs/ieee_phm_bearing/saved_models/.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── project root on sys.path ────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ieee_phm.GRUModel import GRURULModel
from src.models.ieee_phm.TransformerModel import TransformerModel
from src.models.ieee_phm.LSTMModel import LSTMModel
from src.models.ieee_phm.CNNLSTMModel import CNNLSTMModel
from src.models.ieee_phm.BearingCNNBiLSTM import BearingCNNBiLSTM
from src.models.ieee_phm.BearingTransformerModel import BearingTransformerModel

from src.trainer.Trainer import Trainer, TrainingConfig

# ── paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = project_root / "data" / "processed" / "ieee_phm"
OUTPUT_DIR = project_root / "outputs" / "ieee_phm_bearing"
SAVE_MODEL_DIR = OUTPUT_DIR / "saved_models"


# ── reproducibility ─────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class RULDataset(Dataset):
    """PyTorch Dataset for RUL prediction sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (N, seq_len, n_features) float32
            y: (N,) float32
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_preprocessed_data(
    batch_size: int = 16,
    num_workers: int = 2,
) -> tuple:
    """
    Load the numpy arrays saved by the preprocessing notebook and
    return DataLoaders, the scaler, and the hyperparams dict.

    Returns:
        train_loader, val_loader, test_loader, scaler, hyperparams
    """
    # ── load hyperparams ─────────────────────────────────────────────────────
    hp_path = PROCESSED_DIR / "hyperparams.json"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"hyperparams.json not found at {hp_path}. "
            "Run the preprocessing notebook first."
        )
    with open(hp_path) as f:
        hyperparams = json.load(f)

    # ── load numpy sequences ─────────────────────────────────────────────────
    npz_path = PROCESSED_DIR / "ieee_phm_sequences.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"ieee_phm_sequences.npz not found at {npz_path}. "
            "Run the preprocessing notebook first."
        )
    data = np.load(npz_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    # ── load scaler ──────────────────────────────────────────────────────────
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # ── create datasets and loaders ──────────────────────────────────────────
    train_dataset = RULDataset(X_train, y_train)
    val_dataset = RULDataset(X_val, y_val)
    test_dataset = RULDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\nLoaded preprocessed IEEE PHM data from {PROCESSED_DIR}")
    print(f"  Train: {X_train.shape} -> y {y_train.shape}  "
          f"[{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  Val:   {X_val.shape} -> y {y_val.shape}  "
          f"[{y_val.min():.4f}, {y_val.max():.4f}]")
    print(f"  Test:  {X_test.shape} -> y {y_test.shape}  "
          f"[{y_test.min():.4f}, {y_test.max():.4f}]")

    return train_loader, val_loader, test_loader, scaler, hyperparams


# ══════════════════════════════════════════════════════════════════════════════
# Preprocess dict (embedded into checkpoints by the Trainer)
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocess_dict(hyperparams: dict, scaler: Any = None) -> Dict[str, Any]:
    """
    Assemble a `preprocess` dict compatible with the Trainer checkpoint format
    and downstream evaluation / counterfactual generation code.
    """
    preprocess: Dict[str, Any] = {
        "dataset": "ieee_phm",
        "feature_cols": hyperparams["FEATURE_NAMES"],
        "n_features": hyperparams["N_FEATURES"],
        "sequence_length": hyperparams["GRU_SEQ_LEN"],
        "raw_window_size": hyperparams["RAW_WINDOW_SIZE"],
        "raw_stride": hyperparams["RAW_STRIDE"],
        "train_bearings": hyperparams["TRAIN_BEARINGS"],
        "val_bearing": hyperparams["VAL_BEARING"],
        "test_bearings": hyperparams["TEST_BEARINGS"],
        "n_train": hyperparams["train_samples"],
        "n_val": hyperparams["val_samples"],
        "n_test": hyperparams["test_samples"],
        # RUL is normalized linearly to [0, 1] (1 = healthy, 0 = failure)
        "rul_range": [0.0, 1.0],
        "scaler_type": "minmax",
    }

    # Embed scaler parameters if available
    if scaler is not None and hasattr(scaler, "data_min_"):
        preprocess["scaler_params"] = {
            "feature_min": scaler.data_min_.tolist(),
            "feature_max": scaler.data_max_.tolist(),
            "feature_range": scaler.feature_range,
        }

    return preprocess


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def create_model(
    model_type: str,
    n_features: int,
    device: torch.device,
    seq_len: int = 128,
) -> torch.nn.Module:
    """Instantiate a model by name and move it to *device*."""

    model_type_lower = model_type.lower()

    if model_type_lower == "gru":
        model = GRURULModel(
            n_features=n_features,
            hidden1=128,
            hidden2=64,
            fc_dim=64,
            dropout=0.3,
            bidirectional=False,
        )
    elif model_type_lower == "gru_bi":
        model = GRURULModel(
            n_features=n_features,
            hidden1=128,
            hidden2=64,
            fc_dim=64,
            dropout=0.3,
            bidirectional=True,
        )
    elif model_type_lower == "lstm":
        model = LSTMModel(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )
    elif model_type_lower == "cnn_lstm":
        model = CNNLSTMModel(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )
    elif model_type_lower == "cnn_bilstm":
        model = BearingCNNBiLSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )
    elif model_type_lower == "transformer":
        model = TransformerModel(
            input_size=n_features,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.2,
        )
    elif model_type_lower == "bearing_transformer":
        model = BearingTransformerModel(
            input_size=n_features,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_ff=256,
            dropout=0.3,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from: gru, gru_bi, lstm, cnn_lstm, cnn_bilstm, "
            f"transformer, bearing_transformer"
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_type.upper()} — {n_params:,} parameters")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Single-model training
# ══════════════════════════════════════════════════════════════════════════════

def train_single_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocess: Dict[str, Any],
    device: torch.device,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    save_path: str = "",
):
    """Train one model and return the history dict + Trainer instance."""

    n_features = preprocess["n_features"]
    seq_len = preprocess["sequence_length"]
    model = create_model(model_type, n_features, device, seq_len=seq_len)

    cfg = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=train_loader.batch_size,
        early_stopping_patience=25,
        early_stopping_start_epoch=30,
        gradient_clip_value=1.0,
        save_path=save_path,
        dataset_name="ieee_phm",
        model_name=f"{model_type}_best",
        scheduler_patience=15,
        scheduler_factor=0.5,
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    seed_everything(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 7
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # ── load data ────────────────────────────────────────────────────────────
    BATCH_SIZE = 16  # matches the notebook's preprocessing hyperparams

    train_loader, val_loader, test_loader, scaler, hyperparams = (
        load_preprocessed_data(batch_size=BATCH_SIZE, num_workers=2)
    )

    # ── build preprocess dict (embedded in checkpoints) ──────────────────────
    preprocess = build_preprocess_dict(hyperparams, scaler)

    # ── persist preprocess config ────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_path = OUTPUT_DIR / "preprocess.json"
    with open(preprocess_path, "w") as f:
        json.dump(preprocess, f, indent=2, default=str)
    print(f"\nPreprocess config saved to {preprocess_path}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IEEE PHM 2012 BEARING — TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Features:        {hyperparams['N_FEATURES']}  "
          f"{hyperparams['FEATURE_NAMES']}")
    print(f"  Sequence length: {hyperparams['GRU_SEQ_LEN']}")
    print(f"  Train bearings:  {hyperparams['TRAIN_BEARINGS']}")
    print(f"  Val bearing:     {hyperparams['VAL_BEARING']}")
    print(f"  Test bearings:   {hyperparams['TEST_BEARINGS']}")
    print(f"  Train samples:   {hyperparams['train_samples']:,}")
    print(f"  Val samples:     {hyperparams['val_samples']:,}")
    print(f"  Test samples:    {hyperparams['test_samples']:,}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
    print(f"  Test batches:    {len(test_loader)}")
    print("=" * 70)

    # ── model configurations to train ────────────────────────────────────────
    # Uncomment models you want to train.
    models_config = {
        # "gru":         {"lr": 1e-3, "epochs": 50},
        "gru_bi":      {"lr": 1e-3, "epochs": 150},
        # "lstm":        {"lr": 1e-3, "epochs": 50},
        # "cnn_lstm":    {"lr": 1e-3, "epochs": 50},
        # "cnn_bilstm":  {"lr": 1e-3, "epochs": 50},
        "transformer": {"lr": 5e-4, "epochs": 160},
        # "bearing_transformer": {"lr": 5e-4, "epochs": 60},
    }

    # ── training loop ────────────────────────────────────────────────────────
    save_path = str(SAVE_MODEL_DIR)
    os.makedirs(save_path, exist_ok=True)

    histories: Dict[str, Any] = {}
    trainers: Dict[str, Trainer] = {}

    for model_type, mc in models_config.items():
        print(f"\n{'#' * 80}")
        print(f"# Training {model_type.upper()}")
        print(f"{'#' * 80}")

        history, trainer = train_single_model(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            preprocess=preprocess,
            device=device,
            num_epochs=mc["epochs"],
            learning_rate=mc["lr"],
            save_path=save_path,
        )
        histories[model_type] = history
        trainers[model_type] = trainer

    # ── done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training complete. Best checkpoints saved to:")
    print(f"  {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
