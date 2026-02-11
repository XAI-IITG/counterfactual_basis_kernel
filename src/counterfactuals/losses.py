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
    """
    # Sum over time and features, mean over batch (N)
    return torch.mean(torch.sum(Delta ** 2, dim=(1, 2)))

def sparsity_loss(W: torch.Tensor) -> torch.Tensor:
    """
    L1 norm of the weights to encourage using fewer basis functions.
    W shape: (N, K, D)
    """
    return torch.mean(torch.sum(torch.abs(W), dim=(1, 2)))

def dpp_diversity_loss(W: torch.Tensor) -> torch.Tensor:
    """
    Determinantal Point Process (DPP) loss to enforce diversity in WEIGHT space.
    If W vectors are orthogonal, determinant is max. If parallel, det is 0.
    """
    # Flatten W to (N, K*D)
    N = W.shape[0]
    w_flat = W.view(N, -1)
    
    # Normalize rows
    w_norm = torch.nn.functional.normalize(w_flat, p=2, dim=1)
    
    # Compute Similarity Kernel (Cosine Similarity)
    # S_ij = <w_i, w_j>
    S = torch.mm(w_norm, w_norm.t()) # (N, N)
    
    # DPP Loss = -log(det(S))
    # We want to maximize determinant (diversity), so minimize -log(det)
    # Add identity noise for stability
    S = S + torch.eye(N, device=W.device) * 1e-4
    
    return -torch.logdet(S)