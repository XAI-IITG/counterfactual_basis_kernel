# # The optimization loop
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Dict

from .basis import BSplineBasis, FourierBasis, RBFBasis, WaveletBasis, PolynomialBasis
from .losses import validity_loss, proximity_loss, sparsity_loss, dpp_diversity_loss

class BasisGenerator:
    """
    Multi-Stage Counterfactual Generator with proper normalization handling
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
        """
        Args:
            normalization_stats: Dict with 'mean' and 'std' arrays for denormalization
        """
        self.model = model.to(device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.T = sequence_length
        self.D = feature_dim
        self.K = num_basis
        self.device = device
        
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

    def _get_data_statistics(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate data statistics for adaptive perturbation scaling"""
        # Calculate feature-wise statistics
        feature_std = X.std(dim=0)  # Shape: (D,)
        feature_range = X.max(dim=0)[0] - X.min(dim=0)[0]  # Shape: (D,)
        
        return {
            'std': feature_std,
            'range': feature_range,
            'mean': X.mean(dim=0)
        }

    def _random_search(
        self, 
        X_original: torch.Tensor,
        target_rul: float,
        num_candidates: int = 200,
        adaptive_scale: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Stage 1: Random search with adaptive scaling"""
        best_cf = X_original.clone()
        best_error = float('inf')
        
        # ✅ FIX: X_original should be (T, D), not (1, T, D)
        if X_original.dim() == 3:
            X_original = X_original.squeeze(0)
        
        # Get data statistics for adaptive scaling
        if adaptive_scale:
            stats = self._get_data_statistics(X_original)
            scale_factor = stats['std'].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        else:
            scale_factor = 0.1
        
        for _ in range(num_candidates):
            # Random perturbation scaled by feature std
            W_random = torch.randn(1, self.K, self.D, device=self.device) * 0.05
            Delta = torch.einsum('tk, nkd -> ntd', self.Phi, W_random)
            
            if adaptive_scale:
                Delta = Delta * scale_factor
            
            # Apply more conservative clamping based on data range
            max_delta = 0.5  # Max 0.5 std deviation change
            Delta = torch.clamp(Delta, min=-max_delta, max=max_delta)
            
            X_cf = X_original.unsqueeze(0) + Delta  # Add batch dim for computation
            
            # Clip to reasonable normalized range (typically -3 to 3 for normalized data)
            X_cf = torch.clamp(X_cf, min=-3.0, max=3.0)
            
            with torch.no_grad():
                pred = self.model(X_cf).squeeze()
                if pred.dim() > 0:
                    pred = pred[0].item() if len(pred) == 1 else pred.mean().item()
                else:
                    pred = pred.item()
                    
                error = abs(pred - target_rul)
                
                if error < best_error:
                    best_error = error
                    best_cf = X_cf.squeeze(0).clone()  # Remove batch dim for return
        
        return best_cf.unsqueeze(0), best_error  # Add back batch dim for consistency

    def _direction_search(
        self,
        X_original: torch.Tensor,
        target_rul: float,
        num_directions: int = 100
    ) -> Tuple[torch.Tensor, float]:
        """Stage 2: Search along specific feature directions with adaptive scaling"""
        best_cf = X_original.clone()
        best_error = float('inf')
        
        # Get current prediction
        with torch.no_grad():
            pred_tensor = self.model(X_original.unsqueeze(0)).squeeze()
            if pred_tensor.dim() > 0:
                current_pred = pred_tensor[0].item() if len(pred_tensor) == 1 else pred_tensor.mean().item()
            else:
                current_pred = pred_tensor.item()
        
        direction = 1 if target_rul > current_pred else -1
        
        # Get feature statistics for adaptive scaling
        stats = self._get_data_statistics(X_original)
        
        # Try modifying each feature independently
        for feature_idx in range(self.D):
            # Scale by feature's std deviation
            feature_scale = stats['std'][feature_idx].item()
            
            # More granular search with smaller steps
            for scale in np.linspace(0.05, 0.3, num_directions // self.D):
                X_cf = X_original.clone()
                # Modify this feature across time, scaled by its std
                X_cf[:, feature_idx] += direction * scale * feature_scale
                X_cf = torch.clamp(X_cf, min=-3.0, max=3.0)
                
                with torch.no_grad():
                    pred_tensor = self.model(X_cf.unsqueeze(0)).squeeze()
                    if pred_tensor.dim() > 0:
                        pred = pred_tensor[0].item() if len(pred_tensor) == 1 else pred_tensor.mean().item()
                    else:
                        pred = pred_tensor.item()
                        
                    error = abs(pred - target_rul)
                    
                    # Check proximity constraint in normalized space
                    delta_magnitude = torch.norm(X_cf - X_original).item()
                    
                    # More lenient proximity (normalized data typically has unit std)
                    if error < best_error and delta_magnitude < 3.0:
                        best_error = error
                        best_cf = X_cf.clone()
        
        return best_cf, best_error

    def _gradient_refinement(
        self,
        X_init: torch.Tensor,
        X_original: torch.Tensor,
        target_rul: float,
        max_iter: int = 300,
        lr: float = 0.01
    ) -> torch.Tensor:
        """Stage 3: Gradient-based refinement with proximity constraint"""
        X_cf = X_init.clone().unsqueeze(0).requires_grad_(True)
        optimizer = optim.Adam([X_cf], lr=lr)
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            pred = self.model(X_cf).squeeze()
            if pred.dim() > 0:
                pred = pred[0] if len(pred) == 1 else pred.mean()
                
            target_tensor = torch.tensor([target_rul], device=self.device)
            
            # Combined loss: validity + proximity
            validity_loss = torch.nn.functional.mse_loss(pred.unsqueeze(0), target_tensor)
            proximity_loss = 0.1 * torch.nn.functional.mse_loss(X_cf.squeeze(), X_original.unsqueeze(0))
            loss = validity_loss + proximity_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                break
            
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range
            with torch.no_grad():
                X_cf.data = torch.clamp(X_cf.data, min=-3.0, max=3.0)
            
            # Early stop if close enough
            if abs(pred.item() - target_rul) < 2.0 and iteration > 50:
                break
        
        return X_cf.detach().squeeze()

    def generate(
        self, 
        query_instance: torch.Tensor, 
        target_rul: float, 
        num_cfs: int = 1,
        lr: float = 0.05,
        max_iter: int = 500,
        lambdas: dict = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Multi-stage counterfactual generation with proper normalization handling
        """
        if verbose:
            print("\n🔍 Stage 1: Random Search...")
        
        best_cfs = []
        
        for cf_idx in range(num_cfs):
            # Stage 1: Random search with adaptive scaling
            # ✅ FIX: Pass query_instance directly (T, D) not with batch dim
            cf_random, error_random = self._random_search(
                query_instance,  # Changed from query_instance.unsqueeze(0)
                target_rul,
                num_candidates=300,
                adaptive_scale=True
            )
            
            if verbose:
                with torch.no_grad():
                    pred_tensor = self.model(cf_random).squeeze()
                    if pred_tensor.dim() > 0:
                        pred = pred_tensor[0].item() if len(pred_tensor) == 1 else pred_tensor.mean().item()
                    else:
                        pred = pred_tensor.item()
                print(f"  CF {cf_idx+1} - Random search: Pred={pred:.1f}, Target={target_rul:.1f}, Error={error_random:.1f}")
            
            # Stage 2: Direction search
            if verbose:
                print(f"\n🎯 Stage 2: Directional Search...")
            
            cf_direction, error_direction = self._direction_search(
                query_instance,
                target_rul,
                num_directions=150
            )
            
            if verbose:
                with torch.no_grad():
                    pred_tensor = self.model(cf_direction.unsqueeze(0)).squeeze()
                    if pred_tensor.dim() > 0:
                        pred = pred_tensor[0].item() if len(pred_tensor) == 1 else pred_tensor.mean().item()
                    else:
                        pred = pred_tensor.item()
                print(f"  CF {cf_idx+1} - Direction search: Pred={pred:.1f}, Target={target_rul:.1f}, Error={error_direction:.1f}")
            
            # Use better of the two
            if error_direction < error_random:
                best_init = cf_direction
                best_error = error_direction
                method = "direction"
            else:
                best_init = cf_random.squeeze(0) if cf_random.dim() == 3 else cf_random  # ✅ FIX
                best_error = error_random
                method = "random"
            
            # Stage 3: Gradient refinement
            if verbose:
                print(f"\n⚡ Stage 3: Gradient Refinement (from {method})...")
            
            cf_final = self._gradient_refinement(
                best_init,
                query_instance,
                target_rul,
                max_iter=400,
                lr=0.005  # Lower learning rate for fine-tuning
            )
            
            # Verify final result
            with torch.no_grad():
                final_pred_tensor = self.model(cf_final.unsqueeze(0)).squeeze()
                if final_pred_tensor.dim() > 0:
                    final_pred = final_pred_tensor[0].item() if len(final_pred_tensor) == 1 else final_pred_tensor.mean().item()
                else:
                    final_pred = final_pred_tensor.item()
            
            if verbose:
                final_error = abs(final_pred - target_rul)
                delta_norm = torch.norm(cf_final - query_instance).item()
                max_change = torch.max(torch.abs(cf_final - query_instance)).item()
                print(f"  CF {cf_idx+1} - Final: Pred={final_pred:.1f}, Error={final_error:.1f}, ||Δ||={delta_norm:.3f}, max|Δ|={max_change:.3f}")
            
            best_cfs.append(cf_final)
        
        # Stack all counterfactuals
        result = torch.stack(best_cfs)
        
        if verbose:
            print(f"\n✅ Generated {num_cfs} counterfactual(s)")
        
        return result