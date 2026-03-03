# src/train_cmapss_v2.py
# Training script using the v2 CMAPSS data loader (CiRNN-style pipeline).
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

from src.data_loader.cmapss.v2.CMAPSSTorchDataset import CMAPSSTorchDataset, CMAPSSPreprocessInfo
from src.models.cmapss.LSTMModel import LSTMModel
from src.models.cmapss.GRUModel import GRUModel
from src.models.cmapss.CNNLSTMModel import CNNLSTMModel
from src.models.cmapss.TransformerModel import TransformerModel

from src.trainer.Trainer import Trainer, TrainingConfig, NASAScore

# ─────────────────────────── paths ──────────────────────────────────
DATA_DIR = str(Path(__file__).parent.parent / "data" / "processed" / "CMAPSS" / "data")
OUTPUT_ROOT = str(Path(__file__).parent.parent / "outputs" / "cmapss" / "v2")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def info_to_preprocess_dict(info: CMAPSSPreprocessInfo) -> Dict[str, Any]:
    """Convert the CMAPSSPreprocessInfo dataclass into a JSON-safe dict
    that the Trainer can embed in checkpoint metadata."""
    return {
        "subset": info.subset,
        "seq_len": info.seq_len,
        "max_rul": float(info.rul_range),  # range == max since min is 0
        "feature_list": info.feature_list,
        "feature_cols": info.feature_list[2:-1],  # settings + sensors (no unit/time/RUL)
        "n_features": info.n_features,
        "n_context": info.n_context,
        "batch_size": info.batch_size,
        "smoothing_window": info.smoothing_window,
        "use_clustering": info.use_clustering,
        "split_context": info.split_context,
        "global_min": info.global_min.tolist(),
        "global_range": info.global_range.tolist(),
        "label_mode": "cycles",
    }


def create_model(model_type: str, input_size: int, device: torch.device, seq_len: int = 50, max_rul: float = 125.0):
    if model_type == "lstm":
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "gru":
        model = GRUModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "cnn_lstm":
        model = CNNLSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "transformer":
        model = TransformerModel(input_size=input_size, d_model=128, nhead=4, num_layers=2, dropout=0.2)
    elif model_type == "star":
        from src.models.cmapss.STARModel import STARModel
        model = STARModel(input_dim=input_size, seq_len=seq_len, patch_len=5, num_scales=3, d_model=128, nhead=4, ff_dim=256, dropout=0.2, max_rul=max_rul)
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
    input_size = preprocess["n_features"]
    seq_len = preprocess["seq_len"]
    max_rul = float(preprocess["max_rul"])
    subset = preprocess["subset"]
    model = create_model(model_type, input_size, device, seq_len=seq_len, max_rul=max_rul)

    cfg = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=train_loader.batch_size,
        early_stopping_patience=25,
        early_stopping_start_epoch=120,
        gradient_clip_value=1.0,
        save_path=save_path,
        dataset_name="cmapss",
        model_name=f"{model_type}_{subset}_best",
        custom_metrics={"nasa_score": NASAScore.compute},
        progress_bar_extra_metric="nasa_score",
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

    # ---- Subsets to train on ----
    subsets = ["FD001", "FD002", "FD003", "FD004"]

    models_config = {
        "lstm": {"lr": 1e-3, "epochs": 260},
        #"gru": {"lr": 1e-3, "epochs": 160},
        "cnn_lstm": {"lr": 1e-3, "epochs": 260},
        "transformer": {"lr": 5e-4, "epochs": 260},
        "star": {"lr": 5e-4, "epochs": 260},
    }

    all_histories = {}

    ds = CMAPSSTorchDataset(data_dir=DATA_DIR)

    for subset in subsets:
        print(f"\n{'='*80}")
        print(f"  CMAPSS SUBSET: {subset}")
        print(f"{'='*80}")

        train_loader, val_loader, test_loader, info = ds.load(
            subset=subset,
            seq_len=50,
            batch_size=32,
            smoothing_window=3,
            split_context=False,
            seed=42,
        )

        # Build a preprocess dict from the info dataclass (replaces preprocess.json)
        preprocess = info_to_preprocess_dict(info)

        save_path = os.path.join(OUTPUT_ROOT, subset, "saved_models")
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path , "preprocess.json"), "w") as f:
            import json
            json.dump(preprocess, f, indent=4)

        subset_histories = {}

        for model_type, mc in models_config.items():
            print(f"\n{'#'*80}\n# Training {model_type.upper()} on {subset}\n{'#'*80}")
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
            subset_histories[model_type] = history

        all_histories[subset] = subset_histories

    print("\nTraining done for all subsets. Best checkpoints written with preprocess metadata embedded.")


if __name__ == "__main__":
    main()