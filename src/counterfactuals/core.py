# # The optimization loop
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List

# Import the basis classes we defined earlier
from .basis import BSplineBasis, FourierBasis, RBFBasis, WaveletBasis, PolynomialBasis
# Import loss functions (defined below for completeness, or move to losses.py)
from .losses import validity_loss, proximity_loss, sparsity_loss, dpp_diversity_loss

class BasisGenerator:
    """
    The core engine for generating Time-Basis Counterfactuals.
    
    It replaces the pixel-wise optimization of standard DiCE with 
    optimization in the coefficient space of a Temporal Basis.
    """
    def __init__(
        self, 
        model: nn.Module, 
        sequence_length: int, 
        feature_dim: int,
        basis_type: str = 'bspline', 
        num_basis: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: The trained RUL predictor (f).
            sequence_length: T (e.g., 50 cycles).
            feature_dim: D (e.g., 14 sensors).
            basis_type: One of ['bspline', 'fourier', 'rbf', 'wavelet', 'polynomial'].
            num_basis: K (Number of control points/basis functions).
        """
        self.model = model.to(device)
        self.model.eval() # Freezes model batchnorm/dropout
        self.T = sequence_length
        self.D = feature_dim
        self.K = num_basis
        self.device = device
        
        # 1. Initialize the fixed Basis Matrix Phi (T, K)
        self.basis_fn = self._get_basis(basis_type).to(device)
        
        # Pre-compute Phi once since it's constant for all iterations
        with torch.no_grad():
            self.Phi = self.basis_fn() # Shape: (T, K)

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
        max_iter: int = 500,
        lambdas: dict = {'validity': 1.0, 'prox': 0.1, 'sparsity': 0.05, 'diversity': 0.2},
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generates counterfactuals for a single query instance.
        
        Args:
            query_instance: Tensor of shape (T, D)
            target_rul: The desired RUL value (y_target)
            num_cfs: Number of diverse counterfactuals to generate (N)
            
        Returns:
            best_cfs: Tensor of shape (N, T, D)
        """
        # 1. Setup Data
        # X: (1, T, D) -> (N, T, D) (Replicate query for N diversities)
        X_original = query_instance.to(self.device).unsqueeze(0).repeat(num_cfs, 1, 1)
        y_target = torch.tensor([target_rul], device=self.device).float()
        
        # 2. Initialize Learnable Weights W
        # Shape: (N, K, D) - N counterfactuals, K basis weights per feature
        # Init with small noise to break symmetry
        W = torch.randn(num_cfs, self.K, self.D, device=self.device, requires_grad=True)
        W.data = W.data * 0.01 
        
        # 3. Optimizer
        optimizer = optim.Adam([W], lr=lr)
        
        best_loss = float('inf')
        best_cfs = X_original.clone()
        
        # 4. Optimization Loop
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # A. Project Weights to Time Domain: Delta = Phi * W
            # Phi: (T, K), W: (N, K, D) -> Delta: (N, T, D)
            # Einstein Summation: n=batch, t=time, k=basis, d=feature
            Delta = torch.einsum('tk, nkd -> ntd', self.Phi, W)
            
            # B. Apply Perturbation
            X_cf = X_original + Delta
            
            # C. Predict RUL
            # Assumes model takes (Batch, T, D)
            y_pred = self.model(X_cf) # Shape (N, 1) or (N,)
            
            # --- Loss Calculation ---
            l_valid = validity_loss(y_pred, y_target)
            l_prox = proximity_loss(Delta)    # Minimize L2 norm of perturbation
            l_sparse = sparsity_loss(W)       # Minimize L1 norm of weights (simplicity)
            l_div = dpp_diversity_loss(W) if num_cfs > 1 else 0.0
            
            total_loss = (lambdas['validity'] * l_valid + 
                          lambdas['prox'] * l_prox + 
                          lambdas['sparsity'] * l_sparse + 
                          lambdas['diversity'] * l_div)
            
            # --- Backward Pass ---
            total_loss.backward()
            optimizer.step()
            
            # --- Logging & Tracking Best ---
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_cfs = X_cf.detach().clone()
            
            if verbose and i % 100 == 0:
                print(f"Iter {i}: Loss={total_loss.item():.4f} | Pred={y_pred.mean().item():.1f} | Target={target_rul}")

        return best_cfs