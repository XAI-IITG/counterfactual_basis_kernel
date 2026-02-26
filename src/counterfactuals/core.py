# core.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

# Basis classes (keep your basis.py as-is)
from .basis import BSplineBasis, FourierBasis, RBFBasis, WaveletBasis, PolynomialBasis

# Generic losses
from .losses import (
    proximity_loss,
    sparsity_loss,
    smoothness_loss,
    dpp_diversity_loss,
    group_channel_sparsity_loss,
    validity_loss_regression,
    validity_loss_binary,
    validity_loss_multiclass,
)

FeatureRole = Literal["immutable", "action", "state", "context"]
TaskType = Literal["regression", "binary", "multiclass"]


# ------------------------------- Config dataclasses -------------------------------

@dataclass
class TSFeatureSchema:
    """
    Describes feature semantics and constraints for a multivariate time-series dataset.
    All bounds/steps are assumed to be in the SAME space as model input (e.g., normalized).
    """
    feature_names: List[str]
    roles: List[FeatureRole]  # len D

    # Optional direct mutability override (shape D). If None, derived from editable roles.
    mutable_mask: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None

    # Bounds: shape (D,) or (T,D), optional
    min_vals: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    max_vals: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None

    # Optional robust scaling and cost
    mad_inv: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None   # (D,)
    change_cost: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None  # (D,)

    # Optional temporal edit mask: (T,), (T,1), (1,D), or (T,D)
    time_mutable_mask: Optional[Union[np.ndarray, torch.Tensor]] = None

    # Features repeated across time but conceptually static (e.g., age, sex one-hot repeated in each timestep)
    static_mask: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None  # (D,)

    # Optional discretisation constraints
    step_size: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None  # (D,)
    integer_mask: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None  # (D,)

    # Action channels/groups for group penalties
    action_groups: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.feature_names) != len(self.roles):
            raise ValueError("feature_names and roles must have the same length.")

    @property
    def D(self) -> int:
        return len(self.feature_names)


@dataclass
class TargetSpec:
    task_type: TaskType

    # Regression
    target_value: Optional[float] = None
    target_range: Optional[Tuple[float, float]] = None

    # Classification
    target_class: Optional[int] = None

    # Optional confidence margin (classification)
    margin: float = 0.0


@dataclass
class GeneratorConfig:
    basis_type: str = "bspline"
    num_basis: int = 5
    device: str = "cpu"

    # Which roles are directly editable by default
    editable_roles: Tuple[FeatureRole, ...] = ("action",)
    allow_state_edits: bool = False

    init_std: float = 1e-2
    lr: float = 0.1
    max_iter: int = 3000
    eta_min: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stop_tol_reg: float = 1.0
    early_stop_patience: int = 300
    clamp_during_optim: bool = True


@dataclass
class LossWeights:
    validity: float = 10.0
    proximity: float = 5.0
    sparsity: float = 0.5
    diversity: float = 0.05
    smoothness: float = 2.0
    channel_sparsity: float = 0.0
    state_lock: float = 0.0  # soft penalty for direct edits on state channels


# ------------------------------- Utility helpers -------------------------------

def _to_tensor(x, device, dtype=torch.float32):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class DefaultOutputAdapter:
    """
    Adapts arbitrary model output to tensor usable by validity loss.
    Supports:
      - tensor
      - tuple/list -> first element
      - dict with 'logits' or 'pred'
    """
    def __call__(self, y_raw: Any) -> torch.Tensor:
        if isinstance(y_raw, torch.Tensor):
            return y_raw
        if isinstance(y_raw, (tuple, list)):
            if len(y_raw) == 0:
                raise ValueError("Model returned empty tuple/list.")
            if not isinstance(y_raw[0], torch.Tensor):
                raise ValueError("First element of model output tuple/list is not a Tensor.")
            return y_raw[0]
        if isinstance(y_raw, dict):
            if "logits" in y_raw and isinstance(y_raw["logits"], torch.Tensor):
                return y_raw["logits"]
            if "pred" in y_raw and isinstance(y_raw["pred"], torch.Tensor):
                return y_raw["pred"]
            raise ValueError("Dict model output unsupported. Provide a custom output_adapter.")
        raise ValueError("Unsupported model output type. Provide output_adapter.")


# ------------------------------- Generic BasisGenerator -------------------------------

class BasisGenerator:
    """
    Generic Basis-guided counterfactual generator for multivariate time series.

    Supports:
      - regression / binary / multiclass
      - immutable/action/state/context roles
      - tensor input OR dict input (set sequence_key for dict)
    """
    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        feature_dim: int,
        basis_type: str = "bspline",
        num_basis: int = 5,
        device: str = "cpu",
        output_adapter: Optional[Callable[[Any], torch.Tensor]] = None,
        sequence_key: Optional[str] = None,  # if query_instance is dict, e.g., "x_ts"
        config: Optional[GeneratorConfig] = None,
    ):
        self.config = config or GeneratorConfig(
            basis_type=basis_type,
            num_basis=num_basis,
            device=device,
        )
        self.config.basis_type = basis_type
        self.config.num_basis = num_basis
        self.config.device = device

        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.model.eval()  # freeze Dropout/BatchNorm behavior
        for p in self.model.parameters():
            p.requires_grad = False

        self.T = sequence_length
        self.D = feature_dim
        self.K = num_basis
        self.sequence_key = sequence_key
        self.output_adapter = output_adapter or DefaultOutputAdapter()

        self.basis_fn = self._get_basis(self.config.basis_type).to(self.device)
        with torch.no_grad():
            self.Phi = self.basis_fn()  # (T, K)

        self.last_weights_: Optional[torch.Tensor] = None
        self.last_info_: Dict[str, Any] = {}

    def _get_basis(self, name: str):
        if name == "bspline":
            return BSplineBasis(self.T, self.K)
        elif name == "fourier":
            return FourierBasis(self.T, self.K)
        elif name == "rbf":
            return RBFBasis(self.T, self.K)
        elif name == "wavelet":
            return WaveletBasis(self.T, self.K)
        elif name == "polynomial":
            return PolynomialBasis(self.T, self.K)
        else:
            raise ValueError(f"Unknown basis type: {name}")

    # ------------------------ Input handling ------------------------

    def _extract_sequence(self, query_instance: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(query_instance, torch.Tensor):
            x = query_instance
        elif isinstance(query_instance, dict):
            if self.sequence_key is None:
                raise ValueError("query_instance is dict; set sequence_key (e.g., 'x_ts').")
            x = query_instance[self.sequence_key]
        else:
            raise TypeError("query_instance must be Tensor or Dict[str, Tensor].")

        if x.ndim != 2:
            raise ValueError(f"Expected sequence tensor shape (T, D), got {tuple(x.shape)}.")
        if x.shape[0] != self.T or x.shape[1] != self.D:
            raise ValueError(f"Expected (T, D)=({self.T}, {self.D}), got {tuple(x.shape)}.")
        return x.to(self.device)

    def _inject_sequence(self, query_instance, x_cf_batch: torch.Tensor):
        """
        x_cf_batch: (N, T, D)
        Returns model input of same 'type family' as original query_instance.
        """
        if isinstance(query_instance, torch.Tensor):
            return x_cf_batch  # assume model accepts batched tensor (N,T,D)

        payload = {}
        for k, v in query_instance.items():
            payload[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        payload[self.sequence_key] = x_cf_batch
        return payload

    # ------------------------ Schema handling ------------------------

    def _schema_to_tensors(self, schema: TSFeatureSchema) -> Dict[str, torch.Tensor]:
        if schema.D != self.D:
            raise ValueError(f"Schema D={schema.D}, but generator feature_dim={self.D}.")

        # Role-derived mutability (default)
        role_mutable = torch.tensor(
            [
                (r in self.config.editable_roles) or (self.config.allow_state_edits and r == "state")
                for r in schema.roles
            ],
            device=self.device,
            dtype=torch.float32,
        )

        if schema.mutable_mask is not None:
            mutable = _to_tensor(schema.mutable_mask, self.device).view(-1).float()
        else:
            mutable = role_mutable

        if mutable.numel() != self.D:
            raise ValueError("mutable_mask must have shape (D,)")

        immutable = torch.tensor([1.0 if r == "immutable" else 0.0 for r in schema.roles], device=self.device)
        action = torch.tensor([1.0 if r == "action" else 0.0 for r in schema.roles], device=self.device)
        state = torch.tensor([1.0 if r == "state" else 0.0 for r in schema.roles], device=self.device)

        static = _to_tensor(schema.static_mask, self.device) if schema.static_mask is not None else torch.zeros(self.D, device=self.device)
        static = static.view(-1).float()

        # Base edit mask (T, D)
        edit_mask_td = mutable.view(1, self.D).expand(self.T, self.D)

        if schema.time_mutable_mask is not None:
            tm = _to_tensor(schema.time_mutable_mask, self.device).float()
            if tm.ndim == 1 and tm.numel() == self.T:
                tm = tm.view(self.T, 1)
            if tm.shape == (self.T, 1):
                tm = tm.expand(self.T, self.D)
            elif tm.shape == (1, self.D):
                tm = tm.expand(self.T, self.D)
            elif tm.shape != (self.T, self.D):
                raise ValueError("time_mutable_mask must be (T,), (T,1), (1,D), or (T,D)")
            edit_mask_td = edit_mask_td * tm

        def _expand_bound(x):
            if x is None:
                return None
            t = _to_tensor(x, self.device).float()
            if t.ndim == 1:
                if t.numel() != self.D:
                    raise ValueError("Bound vector must have shape (D,)")
                t = t.view(1, self.D).expand(self.T, self.D)
            elif t.shape != (self.T, self.D):
                raise ValueError("Bound must be shape (D,) or (T,D)")
            return t

        min_td = _expand_bound(schema.min_vals)
        max_td = _expand_bound(schema.max_vals)

        mad_inv = _to_tensor(schema.mad_inv, self.device)
        if mad_inv is not None:
            mad_inv = mad_inv.view(-1)

        change_cost = _to_tensor(schema.change_cost, self.device)
        if change_cost is not None:
            change_cost = change_cost.view(-1)

        step_size = _to_tensor(schema.step_size, self.device)
        if step_size is not None:
            step_size = step_size.view(-1)

        integer_mask = _to_tensor(schema.integer_mask, self.device)
        if integer_mask is not None:
            integer_mask = integer_mask.view(-1)

        return {
            "mutable": mutable,
            "immutable": immutable,
            "action": action,
            "state": state,
            "static": static,
            "edit_mask_td": edit_mask_td,
            "min_td": min_td,
            "max_td": max_td,
            "mad_inv": mad_inv,
            "change_cost": change_cost,
            "step_size": step_size,
            "integer_mask": integer_mask,
        }

    # ------------------------ Constraints / projection ------------------------

    def _apply_constraints(
        self,
        X_orig: torch.Tensor,   # (N, T, D)
        Delta: torch.Tensor,    # (N, T, D)
        st: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edit_mask = st["edit_mask_td"].unsqueeze(0)  # (1, T, D)

        # 1) Hard no-edit for non-mutable features/channels/time steps
        Delta = Delta * edit_mask

        # 2) Static repeated features: enforce same delta across time
        static = st["static"].view(1, 1, self.D)
        if torch.any(static > 0):
            mean_delta = Delta.mean(dim=1, keepdim=True)
            Delta = Delta * (1.0 - static) + mean_delta * static

        X_cf = X_orig + Delta

        # 3) Bounds on editable dimensions only
        if self.config.clamp_during_optim and (st["min_td"] is not None or st["max_td"] is not None):
            X_tmp = X_cf
            if st["min_td"] is not None:
                X_tmp = torch.maximum(X_tmp, st["min_td"].unsqueeze(0))
            if st["max_td"] is not None:
                X_tmp = torch.minimum(X_tmp, st["max_td"].unsqueeze(0))
            X_cf = torch.where(edit_mask.bool(), X_tmp, X_orig)
            Delta = X_cf - X_orig

        return X_cf, Delta

    def _project_final_discrete(
        self,
        X_orig: torch.Tensor,   # (N,T,D)
        X_cf: torch.Tensor,     # (N,T,D)
        st: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Final projection for step/integer constraints (non-differentiable).
        """
        step = st["step_size"]
        integer_mask = st["integer_mask"]
        if step is None and integer_mask is None:
            return X_cf

        edit_mask = st["edit_mask_td"].unsqueeze(0).bool()
        Xp = X_cf.clone()

        if step is not None:
            step_safe = torch.where(step > 0, step, torch.ones_like(step))
            step_td = step_safe.view(1, 1, self.D)
            delta = Xp - X_orig
            delta_q = torch.round(delta / step_td) * step_td
            Xp = torch.where(edit_mask, X_orig + delta_q, X_orig)

        if integer_mask is not None:
            intm = (integer_mask.view(1, 1, self.D) > 0) & edit_mask
            Xp = torch.where(intm, torch.round(Xp), Xp)

        if st["min_td"] is not None:
            Xp = torch.where(edit_mask, torch.maximum(Xp, st["min_td"].unsqueeze(0)), X_orig)
        if st["max_td"] is not None:
            Xp = torch.where(edit_mask, torch.minimum(Xp, st["max_td"].unsqueeze(0)), X_orig)

        return Xp

    # ------------------------ Validity wrappers ------------------------

    def _validity_loss(self, y_model: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        if target.task_type == "regression":
            return validity_loss_regression(
                y_model,
                target_value=target.target_value,
                target_range=target.target_range,
            )
        elif target.task_type == "binary":
            if target.target_class is None:
                raise ValueError("TargetSpec.target_class required for binary classification.")
            return validity_loss_binary(y_model, target_class=target.target_class, margin=target.margin)
        elif target.task_type == "multiclass":
            if target.target_class is None:
                raise ValueError("TargetSpec.target_class required for multiclass classification.")
            return validity_loss_multiclass(y_model, target_class=target.target_class, margin=target.margin)
        else:
            raise ValueError(f"Unknown task_type={target.task_type}")

    @torch.no_grad()
    def _validity_error_metric(self, y_model: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        """
        Used only for tracking/early stopping (not backprop).
        Returns per-sample error indicator/value.
        """
        if target.task_type == "regression":
            y = y_model.squeeze(-1) if (y_model.ndim > 1 and y_model.shape[-1] == 1) else y_model.squeeze()
            if y.ndim == 0:
                y = y.unsqueeze(0)

            if target.target_range is not None:
                lo, hi = target.target_range
                lo_t = torch.as_tensor(lo, dtype=y.dtype, device=y.device)
                hi_t = torch.as_tensor(hi, dtype=y.dtype, device=y.device)
                return torch.relu(lo_t - y) + torch.relu(y - hi_t)

            return torch.abs(y - float(target.target_value))

        elif target.task_type == "binary":
            if y_model.ndim == 2 and y_model.shape[-1] == 2:
                pred = torch.argmax(y_model, dim=-1)
            else:
                logits = y_model.squeeze()
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                pred = (torch.sigmoid(logits) >= 0.5).long()
            return (pred != int(target.target_class)).float()

        elif target.task_type == "multiclass":
            pred = torch.argmax(y_model, dim=-1)
            return (pred != int(target.target_class)).float()

        raise ValueError(f"Unknown task_type={target.task_type}")

    # ------------------------ Main optimisation ------------------------

    def generate(
        self,
        query_instance: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: TargetSpec,
        schema: TSFeatureSchema,
        num_cfs: int = 1,
        lr: Optional[float] = None,
        max_iter: Optional[int] = None,
        loss_weights: Optional[LossWeights] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        query_instance:
          - Tensor (T,D), OR
          - Dict[str, Tensor], where one entry (sequence_key) is (T,D)

        Returns:
          best_cfs: (N, T, D) sequence counterfactuals
          info: dict of diagnostics
        """
        lw = loss_weights or LossWeights()
        lr = self.config.lr if lr is None else lr
        max_iter = self.config.max_iter if max_iter is None else max_iter

        x0 = self._extract_sequence(query_instance)  # (T,D)
        st = self._schema_to_tensors(schema)

        X_orig = x0.unsqueeze(0).repeat(num_cfs, 1, 1)  # (N,T,D)

        # Learnable basis weights
        W = (torch.randn(num_cfs, self.K, self.D, device=self.device) * self.config.init_std).requires_grad_(True)

        optimizer = optim.Adam([W], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=self.config.eta_min)

        best_score = float("inf")
        best_total = float("inf")
        best_cfs = X_orig.clone().detach()
        history = []
        no_improve_count = 0

        for i in range(max_iter):
            optimizer.zero_grad()

            # A) Basis projection: Delta = Phi W
            Delta = torch.einsum("tk,nkd->ntd", self.Phi, W)

            # B) Apply mutability / time masks / bounds
            X_cf, Delta = self._apply_constraints(X_orig, Delta, st)

            # C) Model forward on tensor or dict payload
            model_in = self._inject_sequence(query_instance, X_cf)
            y_raw = self.model(model_in) if isinstance(model_in, dict) else self.model(model_in)
            y_model = self.output_adapter(y_raw)

            # D) Losses
            l_valid = self._validity_loss(y_model, target)
            l_prox = proximity_loss(Delta, mad_inv=st["mad_inv"], feature_cost=st["change_cost"])
            l_sparse = sparsity_loss(W)
            l_smooth = smoothness_loss(Delta)
            l_group = group_channel_sparsity_loss(Delta, schema.action_groups)

            if lw.state_lock > 0:
                l_state_lock = torch.mean((Delta * st["state"].view(1, 1, self.D)) ** 2)
            else:
                l_state_lock = torch.zeros((), device=self.device)

            total_loss = (
                lw.validity * l_valid
                + lw.proximity * l_prox
                + lw.sparsity * l_sparse
                + lw.smoothness * l_smooth
                + lw.channel_sparsity * l_group
                + lw.state_lock * l_state_lock
            )

            if num_cfs > 1 and lw.diversity > 0:
                l_div = dpp_diversity_loss(W)
                total_loss = total_loss + lw.diversity * l_div
            else:
                l_div = torch.zeros((), device=self.device)

            # E) Optimisation step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([W], max_norm=self.config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            # F) Tracking
            with torch.no_grad():
                validity_err = self._validity_error_metric(y_model, target).mean().item()
                t_loss = float(total_loss.item())

                if (validity_err < best_score) or (abs(validity_err - best_score) < 1e-12 and t_loss < best_total):
                    best_score = validity_err
                    best_total = t_loss
                    best_cfs = X_cf.detach().clone()
                    self.last_weights_ = W.detach().clone()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                history.append({
                    "iter": i,
                    "total_loss": t_loss,
                    "l_valid": float(l_valid.item()),
                    "l_prox": float(l_prox.item()),
                    "l_sparse": float(l_sparse.item()),
                    "l_smooth": float(l_smooth.item()),
                    "l_group": float(l_group.item()),
                    "l_div": float(l_div.item()),
                    "validity_err": float(validity_err),
                })

                if verbose and (i % 100 == 0 or i == max_iter - 1):
                    print(
                        f"Iter {i:04d} | total={t_loss:.4f} | valid={l_valid.item():.4f} "
                        f"| prox={l_prox.item():.4f} | sparse={l_sparse.item():.4f} "
                        f"| smooth={l_smooth.item():.4f} | group={l_group.item():.4f} "
                        f"| div={l_div.item():.4f} | v_err={validity_err:.4f}"
                    )

                # Early stopping
                if target.task_type == "regression":
                    if validity_err < self.config.early_stop_tol_reg:
                        if verbose:
                            print(f"Regression target reached at iter {i}")
                        break
                else:
                    # classification success indicator becomes 0
                    if validity_err == 0.0:
                        if verbose:
                            print(f"Classification target reached at iter {i}")
                        break

                if no_improve_count >= self.config.early_stop_patience:
                    if verbose:
                        print(f"Early stop (no improvement) at iter {i}")
                    break

        # Final projection for discrete / step constraints
        best_cfs = self._project_final_discrete(X_orig, best_cfs, st)

        info = {
            "best_validity_err": best_score,
            "best_total_loss": best_total,
            "weights": None if self.last_weights_ is None else self.last_weights_.detach().cpu(),
            "history": history,
            "feature_names": schema.feature_names,
            "roles": schema.roles,
        }
        self.last_info_ = info

        return best_cfs, info
