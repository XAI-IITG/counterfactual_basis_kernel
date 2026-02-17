import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Dict

# Import the basis classes we defined earlier
from .basis import BSplineBasis, FourierBasis, RBFBasis, WaveletBasis, PolynomialBasis
# Import loss functions
from .losses import validity_loss, proximity_loss, sparsity_loss, dpp_diversity_loss, smoothness_loss

class BasisGenerator:
    """
    The core engine for generating Time-Basis Counterfactuals.
    """
    def __init__(
        self, 
        model: nn.Module, 
        sequence_length: int, 
        feature_dim: int,
        basis_type: str = 'bspline', 
        num_basis: int = 5,
        device: str = 'cpu',
        normalization_stats: Optional[Dict[str, np.ndarray]] = None
    ):
        self.model = model.to(device)
        self.model.eval() # <--- CRITICAL: Must be in eval mode to freeze BatchNorm/Dropout
        
        # Freeze model parameters to save memory/compute
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.T = sequence_length
        self.D = feature_dim
        self.K = num_basis
        self.device = device
        
        # Initialize the fixed Basis Matrix Phi (T, K)
        self.basis_fn = self._get_basis(basis_type).to(device)

        # Store normalization stats if provided
        if normalization_stats is not None:
            self.mean = torch.tensor(normalization_stats['mean'], dtype=torch.float32, device=device)
            self.std = torch.tensor(normalization_stats['std'], dtype=torch.float32, device=device)
            self.has_norm_stats = True
        else:
            self.has_norm_stats = False

        # MAD inverse for feature-normalised proximity (set via set_mad)
        self.mad_inv: Optional[torch.Tensor] = None

        # Last optimised weights (exposed for diagnostics / error analysis)
        self.last_weights_: Optional[torch.Tensor] = None
            
        self.basis_fn = self._get_basis(basis_type).to(device)
        with torch.no_grad():
            self.Phi = self.basis_fn()
        
    # ------------------------------------------------------------------
    # Optional: compute MAD from training data for proximity scaling
    # ------------------------------------------------------------------
    def set_mad_from_data(self, train_data: np.ndarray, eps: float = 1e-8):
        """
        Compute per-feature Median Absolute Deviation from training data
        and store its inverse for use in proximity_loss.

        train_data : np.ndarray of shape (num_samples, T, D) or (num_rows, D)
        """
        if train_data.ndim == 3:
            # Flatten (samples, T, D) -> (samples*T, D)
            flat = train_data.reshape(-1, train_data.shape[-1])
        else:
            flat = train_data

        median = np.median(flat, axis=0)
        mad = np.median(np.abs(flat - median), axis=0)
        mad_inv = 1.0 / (mad + eps)
        self.mad_inv = torch.tensor(mad_inv, dtype=torch.float32, device=self.device)

    def _get_basis(self, name: str):
        if name == 'bspline':
            return BSplineBasis(self.T, self.K)
        elif name == 'fourier':
            return FourierBasis(self.T, self.K)
        elif name == 'rbf':
            return RBFBasis(self.T, self.K)
        elif name == 'wavelet':
            return WaveletBasis(self.T, self.K)
        elif name == 'polynomial':
            return PolynomialBasis(self.T, self.K)
        else:
            raise ValueError(f"Unknown basis type: {name}")

    def generate(
        self, 
        query_instance: torch.Tensor, 
        target_rul: float, 
        num_cfs: int = 1,
        lr: float = 0.1,
        max_iter: int = 3000,
        lambdas: dict = None,
        verbose: bool = True
    ) -> torch.Tensor:
        
        # Default Lambdas
        if lambdas is None:
            lambdas = {
                'validity': 10.0,
                'prox': 5.0,
                'sparsity': 0.5,
                'diversity': 0.05,
                'smoothness': 2.0,
            }

        # 1. Setup Data
        X_original = query_instance.to(self.device).unsqueeze(0).repeat(num_cfs, 1, 1)
        y_target = torch.tensor([target_rul] * num_cfs, device=self.device).float()
        
        # 2. Initialize Weights W (Learnable)
        # Initialize close to zero so we start near the original trajectory
        W = torch.randn(num_cfs, self.K, self.D, device=self.device) * 0.01
        W.requires_grad = True
        
        # 3. Optimizer + LR scheduler
        optimizer = optim.Adam([W], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iter, eta_min=1e-4
        )
        
        best_loss = float('inf')
        best_cfs = X_original.clone()
        
        # 4. Optimization Loop
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # A. Project Weights: Delta = Phi * W
            Delta = torch.einsum('tk, nkd -> ntd', self.Phi, W)
            
            # B. Apply Perturbation + clamp to feasible normalised range
            X_cf = X_original + Delta
            X_cf = torch.clamp(X_cf, min=-3.0, max=3.0)
            
            # C. Predict RUL
            y_pred = self.model(X_cf)
            if y_pred.dim() > 1: y_pred = y_pred.squeeze()
            if y_pred.dim() == 0: y_pred = y_pred.unsqueeze(0) # Handle batch size 1
            
            # D. Loss Calculation
            l_valid = validity_loss(y_pred, y_target)
            l_prox = proximity_loss(Delta, mad_inv=self.mad_inv)
            l_sparse = sparsity_loss(W)
            l_smooth = smoothness_loss(Delta)
            
            total_loss = (lambdas['validity'] * l_valid + 
                          lambdas['prox'] * l_prox + 
                          lambdas['sparsity'] * l_sparse +
                          lambdas.get('smoothness', 0.0) * l_smooth)
            
            if num_cfs > 1:
                l_div = dpp_diversity_loss(W)
                total_loss += lambdas.get('diversity', 0.0) * l_div
            
            # E. Backward Pass + gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # F. Tracking
            current_mae = torch.mean(torch.abs(y_pred - y_target)).item()
            
            # Save best result based on VALIDITY (hitting the target is priority)
            if l_valid.item() < best_loss:
                best_loss = l_valid.item()
                best_cfs = X_cf.detach().clone()
                self.last_weights_ = W.detach().clone()
            
            if verbose and i % 100 == 0:
                print(f"Iter {i}: Loss={total_loss.item():.4f} | "
                      f"Valid_MSE={l_valid.item():.4f} | "
                      f"Mean_Pred={y_pred.mean().item():.1f} | "
                      f"Target={target_rul:.1f}")
                
            # Early stopping if target is reached very closely
            if current_mae < 1.0:
                if verbose: print(f"Target reached at iter {i}!")
                best_cfs = X_cf.detach().clone()
                self.last_weights_ = W.detach().clone()
                break

        return best_cfs