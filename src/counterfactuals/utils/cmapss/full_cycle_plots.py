"""
Plotting utilities for full-cycle counterfactuals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional


def plot_full_cycle_features(
    result: Dict[str, np.ndarray],
    feature_cols: List[str],
    unit_id: int,
    features_to_plot: Optional[List[int]] = None,
    max_features: int = 6,
    figsize_per_row: float = 2.5,
):
    """
    Plot original vs counterfactual for selected features across the FULL engine cycle.
    """
    original = result['original']  # (L, D)
    cf = result['cf']              # (L, D)
    L, D = original.shape

    if features_to_plot is None:
        # Pick features with largest total change
        total_change = np.mean(np.abs(cf - original), axis=0)  # (D,)
        features_to_plot = np.argsort(total_change)[-max_features:][::-1].tolist()

    n_feats = len(features_to_plot)
    fig, axes = plt.subplots(n_feats, 1, figsize=(14, figsize_per_row * n_feats), sharex=True)
    if n_feats == 1:
        axes = [axes]

    timesteps = np.arange(L)

    for ax, fidx in zip(axes, features_to_plot):
        fname = feature_cols[fidx] if fidx < len(feature_cols) else f"Feature {fidx}"

        ax.plot(timesteps, original[:, fidx], label='Original', color='#1f77b4', linewidth=1.2)
        ax.plot(timesteps, cf[:, fidx], label='Counterfactual', color='#d62728', linewidth=1.2, alpha=0.85)

        # Shade the difference
        ax.fill_between(
            timesteps,
            original[:, fidx],
            cf[:, fidx],
            alpha=0.15,
            color='#d62728',
        )

        ax.set_ylabel(fname, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Cycle (timestep)', fontsize=11)
    fig.suptitle(f'Full-Cycle Counterfactual — Unit {unit_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_full_cycle_rul_trajectory(
    result: Dict[str, np.ndarray],
    unit_id: int,
    sequence_length: int = 50,
    figsize: tuple = (14, 5),
):
    """
    Plot RUL predictions along the full cycle for original and CF windows.
    Also overlay the true RUL if available.
    """
    orig_preds = result['original_preds']
    cf_preds = result['cf_preds']
    target_ruls = result['target_ruls']
    window_starts = result['window_starts']
    ruls = result['ruls']
    L = result['original'].shape[0]

    # Window centers for x-axis alignment
    window_centers = window_starts + sequence_length / 2.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # True RUL curve
    if ruls is not None and np.any(ruls > 0):
        ax.plot(np.arange(L), ruls, label='True RUL', color='green', linewidth=2, linestyle='--')

    ax.plot(window_centers, orig_preds, label='Original Pred RUL', color='#1f77b4', linewidth=1.5)
    ax.plot(window_centers, cf_preds, label='CF Pred RUL', color='#d62728', linewidth=1.5, alpha=0.85)
    ax.plot(window_centers, target_ruls, label='Target RUL', color='#ff7f0e', linewidth=1, linestyle=':', alpha=0.7)

    ax.fill_between(window_centers, orig_preds, cf_preds, alpha=0.12, color='#d62728')

    ax.set_xlabel('Cycle (timestep)', fontsize=12)
    ax.set_ylabel('RUL', fontsize=12)
    ax.set_title(f'Full-Cycle RUL Trajectory — Unit {unit_id}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_full_cycle_delta_heatmap(
    result: Dict[str, np.ndarray],
    feature_cols: List[str],
    unit_id: int,
    figsize: tuple = (16, 6),
    cmap: str = 'RdBu_r',
):
    """
    Heatmap of |CF - Original| across all cycles and features.
    """
    original = result['original']
    cf = result['cf']
    delta = cf - original  # signed difference

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Signed delta
    vmax = np.max(np.abs(delta))
    im0 = axes[0].imshow(
        delta.T, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation='nearest',
    )
    axes[0].set_title('Signed Change (CF − Original)', fontsize=12)
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Feature')
    axes[0].set_yticks(range(len(feature_cols)))
    axes[0].set_yticklabels(feature_cols, fontsize=7)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Absolute delta
    im1 = axes[1].imshow(
        np.abs(delta).T, aspect='auto', cmap='hot_r',
        interpolation='nearest',
    )
    axes[1].set_title('|CF − Original|', fontsize=12)
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Feature')
    axes[1].set_yticks(range(len(feature_cols)))
    axes[1].set_yticklabels(feature_cols, fontsize=7)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f'Full-Cycle Perturbation Heatmap — Unit {unit_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_full_cycle_overlap_coverage(
    result: Dict[str, np.ndarray],
    unit_id: int,
    figsize: tuple = (14, 3),
):
    """
    Show how many overlapping windows contributed to each timestep.
    """
    window_count = result['window_count']  # (L, D)
    mean_count = window_count.mean(axis=1)  # (L,)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.fill_between(np.arange(len(mean_count)), 0, mean_count, alpha=0.5, color='steelblue')
    ax.plot(np.arange(len(mean_count)), mean_count, color='steelblue', linewidth=1)
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('Window overlap count', fontsize=11)
    ax.set_title(f'Window Overlap Coverage — Unit {unit_id}', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_full_cycle_summary(
    result: Dict[str, np.ndarray],
    feature_cols: List[str],
    unit_id: int,
    sequence_length: int = 50,
    features_to_plot: Optional[List[int]] = None,
    max_features: int = 4,
):
    """
    Combined summary: RUL trajectory + top-changed features + delta heatmap.
    """
    plot_full_cycle_rul_trajectory(result, unit_id, sequence_length)
    plot_full_cycle_features(result, feature_cols, unit_id, features_to_plot, max_features)
    plot_full_cycle_delta_heatmap(result, feature_cols, unit_id)
    plot_full_cycle_overlap_coverage(result, unit_id)