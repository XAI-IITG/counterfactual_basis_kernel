import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Dict

# Import the basis classes we defined earlier
from .basis import BSplineBasis, FourierBasis, RBFBasis, WaveletBasis, PolynomialBasis
# Import loss functions
from .losses import validity_loss, proximity_loss, sparsity_loss, dpp_diversity_loss

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
            
        self.basis_fn = self._get_basis(basis_type).to(device)
        with torch.no_grad():
            self.Phi = self.basis_fn()
        
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
        
        # Default Lambdas (Validity needs to be high to force RUL change)
        if lambdas is None:
            lambdas = {'validity': 10.0, 'prox': 0.01, 'sparsity': 0.01, 'diversity': 0.1}

        # 1. Setup Data
        X_original = query_instance.to(self.device).unsqueeze(0).repeat(num_cfs, 1, 1)
        y_target = torch.tensor([target_rul] * num_cfs, device=self.device).float()
        
        # 2. Initialize Weights W (Learnable)
        # Initialize close to zero so we start near the original trajectory
        W = torch.randn(num_cfs, self.K, self.D, device=self.device) * 0.01
        W.requires_grad = True
        
        # 3. Optimizer
        optimizer = optim.Adam([W], lr=lr)
        
        best_loss = float('inf')
        best_cfs = X_original.clone()
        
        # 4. Optimization Loop
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # A. Project Weights: Delta = Phi * W
            Delta = torch.einsum('tk, nkd -> ntd', self.Phi, W)
            
            # B. Apply Perturbation
            X_cf = X_original + Delta
            
            # C. Predict RUL
            y_pred = self.model(X_cf)
            if y_pred.dim() > 1: y_pred = y_pred.squeeze()
            if y_pred.dim() == 0: y_pred = y_pred.unsqueeze(0) # Handle batch size 1
            
            # D. Loss Calculation
            l_valid = validity_loss(y_pred, y_target)
            l_prox = proximity_loss(Delta)
            l_sparse = sparsity_loss(W)
            
            total_loss = (lambdas['validity'] * l_valid + 
                          lambdas['prox'] * l_prox + 
                          lambdas['sparsity'] * l_sparse)
            
            if num_cfs > 1:
                l_div = dpp_diversity_loss(W)
                total_loss += lambdas['diversity'] * l_div
            
            # E. Backward Pass
            total_loss.backward()
            optimizer.step()
            
            # F. Tracking
            current_mae = torch.mean(torch.abs(y_pred - y_target)).item()
            
            # Save best result based on VALIDITY (hitting the target is priority)
            # We accept a bit more distance if it means we actually reach the RUL
            if l_valid.item() < best_loss:
                best_loss = l_valid.item()
                best_cfs = X_cf.detach().clone()
            
            if verbose and i % 100 == 0:
                print(f"Iter {i}: Loss={total_loss.item():.4f} | "
                      f"Valid_MSE={l_valid.item():.4f} | "
                      f"Mean_Pred={y_pred.mean().item():.1f} | "
                      f"Target={target_rul:.1f}")
                
            # Early stopping if target is reached very closely
            if current_mae < 1.0:
                if verbose: print(f"Target reached at iter {i}!")
                best_cfs = X_cf.detach().clone()
                break

        return best_cfs