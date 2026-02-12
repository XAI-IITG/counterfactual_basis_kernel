# Custom loss functions (DPP, Sparsity, Validity)
import torch

def validity_loss(y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """
    MSE Loss ensuring prediction moves towards target.
    """
    return torch.mean((y_pred - y_target) ** 2)

def proximity_loss(Delta: torch.Tensor) -> torch.Tensor:
    """
    L2 norm of the perturbation in the time domain.
    Delta shape: (N, T, D)
    ✅ ENHANCED: Penalize both average and maximum changes
    """
    # Average L2 norm across all dimensions
    l2_norm = torch.mean(torch.sum(Delta ** 2, dim=(1, 2)))
    
    # ✅ FIX: Use reshape instead of view for non-contiguous tensors
    max_change = torch.mean(torch.max(torch.abs(Delta).reshape(Delta.shape[0], -1), dim=1)[0])
    
    return l2_norm + 0.5 * max_change

def sparsity_loss(W: torch.Tensor) -> torch.Tensor:
    """
    L1 norm of the weights to encourage using fewer basis functions.
    W shape: (N, K, D)
    """
    return torch.mean(torch.sum(torch.abs(W), dim=(1, 2)))

def dpp_diversity_loss(W: torch.Tensor) -> torch.Tensor:
    """
    Determinantal Point Process (DPP) loss to enforce diversity in WEIGHT space.
    ✅ FIXED: Added numerical stability and better scaling
    """
    N = W.shape[0]
    
    # Early exit for single sample
    if N <= 1:
        return torch.tensor(0.0, device=W.device)
    
    # ✅ FIX: Use reshape instead of view
    w_flat = W.reshape(N, -1)
    
    # ✅ FIX: Add small noise to prevent identical samples
    w_flat = w_flat + torch.randn_like(w_flat) * 1e-6
    
    # Normalize rows
    w_norm = torch.nn.functional.normalize(w_flat, p=2, dim=1, eps=1e-8)
    
    # Compute Similarity Kernel (Cosine Similarity)
    S = torch.mm(w_norm, w_norm.t()) # (N, N)
    
    # ✅ FIX: Add larger identity noise for stability
    S = S + torch.eye(N, device=W.device) * 0.01
    
    # ✅ FIX: Use Cholesky decomposition for better numerical stability
    try:
        L = torch.linalg.cholesky(S)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        return -log_det  # Minimize negative log-det to maximize diversity
    except RuntimeError:
        # If Cholesky fails, use SVD-based determinant
        _, s, _ = torch.svd(S)
        # Add small epsilon to prevent log(0)
        log_det = torch.sum(torch.log(s + 1e-8))
        return -log_det