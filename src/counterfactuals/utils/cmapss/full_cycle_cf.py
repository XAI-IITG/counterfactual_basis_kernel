"""
Utilities to generate and stitch counterfactuals across the FULL engine cycle,
not just a single sequence window.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from src.counterfactuals.core import (
    BasisGenerator,
    TSFeatureSchema,
    TargetSpec,
    LossWeights,
)
from src.counterfactuals.utils.cmapss.cf_utils import predict_rul, get_valid_target_rul


def get_full_unit_data(
    df,
    unit_id: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the FULL time-series for a given unit.

    Returns:
        features: (total_cycles, D)
        ruls:     (total_cycles,)
        cycles:   (total_cycles,)
    """
    unit_df = df[df['unit_id'] == unit_id].sort_values('cycle').reset_index(drop=True)
    features = unit_df[feature_cols].values.astype(np.float32)
    ruls = unit_df['rul'].values.astype(np.float32) if 'rul' in unit_df.columns else np.zeros(len(unit_df))
    cycles = unit_df['cycle'].values.astype(int)
    return features, ruls, cycles


def sliding_window_indices(
    total_length: int,
    window_size: int,
    stride: int = 1,
) -> List[Tuple[int, int]]:
    """
    Return (start, end) indices for each sliding window.
    Only produces windows of exactly `window_size`.
    """
    indices = []
    for start in range(0, total_length - window_size + 1, stride):
        indices.append((start, start + window_size))
    return indices



def generate_full_cycle_cf(
    gen: BasisGenerator,
    model: torch.nn.Module,
    full_features: np.ndarray,       # (L, D)
    full_ruls: np.ndarray,           # (L,)
    schema: TSFeatureSchema,
    loss_weights: LossWeights,
    sequence_length: int = 50,
    stride: int = 1,
    rul_increase_range: Tuple[int, int] = (10, 30),
    max_rul: float = 125.0,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate counterfactuals for EVERY sliding window across the full cycle
    and stitch them together by averaging overlapping regions.

    If the unit is shorter than sequence_length, it is left-padded (by repeating
    the first timestep) so the model can still process it.  The padded prefix is
    stripped from all outputs so the returned arrays match the original length.

    Returns dict with:
        'original':        (L, D) full original features
        'cf':              (L, D) stitched counterfactual
        'original_preds':  (num_windows,) model prediction per window on original
        'cf_preds':        (num_windows,) model prediction per window on CF
        'target_ruls':     (num_windows,) target RUL per window
        'window_starts':   (num_windows,) start index of each window (in original coords)
        'window_count':    (L, D) how many windows contributed to each position
        'ruls':            (L,) true RUL per timestep
    """
    L_orig, D = full_features.shape
    padded = False
    pad_len = 0

    if L_orig < sequence_length:
        # Left-pad by repeating the first timestep
        pad_len = sequence_length - L_orig
        pad_features = np.tile(full_features[0:1, :], (pad_len, 1))  # (pad_len, D)
        full_features = np.concatenate([pad_features, full_features], axis=0)
        # Pad RULs: use max_rul for the padded prefix (earliest = highest RUL)
        pad_ruls = np.full(pad_len, full_ruls[0] + pad_len if full_ruls is not None else max_rul)
        full_ruls = np.concatenate([pad_ruls, full_ruls], axis=0)
        padded = True
        if verbose:
            print(f"  ℹ️ Unit has {L_orig} cycles < sequence_length={sequence_length}. "
                  f"Left-padded by {pad_len} to {full_features.shape[0]}.")

    L, D = full_features.shape

    windows = sliding_window_indices(L, sequence_length, stride)
    num_windows = len(windows)

    if num_windows == 0:
        raise ValueError(
            f"No valid windows after padding. L={L}, sequence_length={sequence_length}."
        )

    # Accumulators for stitching (weighted average of overlapping CFs)
    cf_accum = np.zeros((L, D), dtype=np.float64)
    cf_count = np.zeros((L, D), dtype=np.float64)

    original_preds = []
    cf_preds = []
    target_ruls = []
    window_starts = []

    for i, (ws, we) in enumerate(windows):
        query_seq = full_features[ws:we]  # (T, D)

        # Get original prediction
        orig_pred = predict_rul(model, query_seq, device)

        # Determine target
        target_rul = get_valid_target_rul(orig_pred, increase_range=rul_increase_range, max_rul=max_rul)
        target = TargetSpec(task_type="regression", target_value=target_rul)

        query_tensor = torch.tensor(query_seq, dtype=torch.float32).to(device)

        try:
            cfs, info = gen.generate(
                query_instance=query_tensor,
                target=target,
                schema=schema,
                num_cfs=1,
                loss_weights=loss_weights,
                verbose=False,
            )
            cf_window = cfs[0].detach().cpu().numpy()  # (T, D)
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Window [{ws}:{we}] failed: {e}, using original")
            cf_window = query_seq.copy()

        cf_pred = predict_rul(model, cf_window, device)

        # Accumulate
        cf_accum[ws:we] += cf_window
        cf_count[ws:we] += 1.0

        original_preds.append(orig_pred)
        cf_preds.append(cf_pred)
        target_ruls.append(target_rul)
        window_starts.append(ws)

        if verbose and (i % max(1, num_windows // 10) == 0 or i == num_windows - 1):
            print(
                f"  Window {i+1}/{num_windows} [{ws}:{we}] | "
                f"orig={orig_pred:.1f} → target={target_rul:.1f} → cf={cf_pred:.1f}"
            )

    # Average overlapping regions
    cf_count = np.maximum(cf_count, 1.0)  # avoid division by zero
    cf_stitched = (cf_accum / cf_count).astype(np.float32)

    # Strip padding from spatial arrays, adjust window_starts for original coordinates
    if padded:
        full_features = full_features[pad_len:]
        cf_stitched = cf_stitched[pad_len:]
        cf_count_out = cf_count[pad_len:]
        full_ruls = full_ruls[pad_len:]
        window_starts = [max(0, ws - pad_len) for ws in window_starts]
    else:
        cf_count_out = cf_count

    return {
        'original': full_features,
        'cf': cf_stitched,
        'original_preds': np.array(original_preds),
        'cf_preds': np.array(cf_preds),
        'target_ruls': np.array(target_ruls),
        'window_starts': np.array(window_starts),
        'window_count': cf_count_out.astype(np.float32),
        'ruls': full_ruls,
    }



def predict_full_cycle_rul(
    model: torch.nn.Module,
    full_features: np.ndarray,  # (L, D)
    sequence_length: int,
    device: str = "cpu",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get per-window RUL predictions across the full cycle.

    Returns:
        window_centers: (num_windows,) center timestep of each window
        predictions:    (num_windows,) RUL prediction
    """
    L = full_features.shape[0]
    windows = sliding_window_indices(L, sequence_length, stride)

    centers = []
    preds = []
    for ws, we in windows:
        seq = full_features[ws:we]
        pred = predict_rul(model, seq, device)
        centers.append((ws + we) / 2.0)
        preds.append(pred)

    return np.array(centers), np.array(preds)