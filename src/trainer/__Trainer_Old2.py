"""
Trainer Module for RUL Prediction Models

This module provides a comprehensive training framework for deep learning models
used in Remaining Useful Life (RUL) prediction tasks, specifically designed for
the CMAPSS dataset.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.EarlyStopping import EarlyStopping


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    num_epochs: int = 180
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    early_stopping_patience: int = 20
    early_stopping_start_epoch: int = 80
    gradient_clip_value: float = 1.0
    
    # Scheduler parameters
    scheduler_factor: float = 0.6
    scheduler_patience: int = 30
    scheduler_mode: str = 'min'
    
    # Logging and saving
    save_every_n_epochs: int = 10
    print_every_n_epochs: int = 5
    
    # Paths
    save_path: str = '../outputs/saved_models'
    model_name: str = 'model'


class RULMetrics:
    """Utility class for RUL-specific metrics calculation"""
    
    @staticmethod
    def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate NASA scoring function for RUL prediction.
        
        This score penalizes late predictions more heavily than early predictions,
        which is crucial for maintenance scheduling.
        
        Args:
            y_true: Ground truth RUL values
            y_pred: Predicted RUL values
            
        Returns:
            NASA score (lower is better)
        """
        diff = y_pred - y_true
        score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
        return float(score)
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for RUL prediction.
        
        Args:
            y_true: Ground truth RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Dictionary containing all calculated metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        score = RULMetrics.nasa_score(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'nasa_score': float(score)
        }


class Trainer:
    """
    Trainer class for RUL prediction models.
    
    This class handles the complete training pipeline including:
    - Training and validation loops
    - Metrics calculation and tracking
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Training history logging
    
    Attributes:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on (CPU/GPU)
        config: Training configuration
        history: Training history tracker
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to run training on
            config: Training configuration (uses defaults if None)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or TrainingConfig()
        
        # Initialize training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.scheduler_mode,
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience
        )
        
        # Initialize history tracker
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'val_r2': [],
            'val_nasa_score': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_score = float('inf')
        self.best_epoch = 0
        
        # Setup early stopping
        self._setup_early_stopping()
    

    # callback that saves PURE state_dict (so inference is simple)
    def _save_best_weights(epoch: int, is_best: bool):
        if is_best:
            torch.save(self.model.state_dict(), best_weights_path)
            
    def _setup_early_stopping(self) -> None:
        """Setup early stopping mechanism"""
        os.makedirs(self.config.save_path, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.save_path,
            f"{self.config.model_name}_checkpoint.pth"
        )
        
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            verbose=True,
            start_epoch=self.config.early_stopping_start_epoch,
            mode="min",                   # NASA score lower is better
            save_fn=_save_best_weights,    # keep saving format consistent
        )
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]",
            leave=False
        )
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data).squeeze()
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_value
                )
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        progress_bar.close()
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (metrics dictionary, predictions array, actuals array)
        """
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        actuals_list = []
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs} [Val]  ",
            leave=False
        )
        
        with torch.no_grad():
            for data, target in progress_bar:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data).squeeze()
                loss = self.criterion(output, target)
                
                # Track loss and predictions
                total_loss += loss.item()
                predictions_list.extend(output.cpu().numpy())
                actuals_list.extend(target.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        progress_bar.close()
        
        # Convert to numpy arrays
        predictions = np.array(predictions_list)
        actuals = np.array(actuals_list)
        
        # Calculate all metrics
        metrics = RULMetrics.calculate_all_metrics(actuals, predictions)
        metrics['loss'] = total_loss / num_batches
        
        return metrics, predictions, actuals
    
    def save_history(self) -> None:
        """Save training history to JSON file"""
        json_path = os.path.join(
            self.config.save_path,
            f"{self.config.model_name}_history.json"
        )
        
        # Convert to serializable format
        history_serializable = {
            key: [float(v) for v in values]
            for key, values in self.history.items()
        }
        
        with open(json_path, 'w') as f:
            json.dump(history_serializable, f, indent=4)
    
    # def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
    #     """
    #     Save model checkpoint.
        
    #     Args:
    #         epoch: Current epoch number
    #         is_best: Whether this is the best model so far
    #     """
    #     checkpoint = {
    #         # "preprocess": {
    #         #     "feature_cols": self.preprocess_feature_cols,   # list[str]
    #         #     "mean": self.preprocess_mean,                   # list/np array
    #         #     "std": self.preprocess_std,                     # list/np array
    #         #     "sequence_length": self.preprocess_seq_len,
    #         #     "label_transform": self.label_transform,         # e.g. "cycles" / "divide_max_rul" / "zscore"
    #         #     "label_params": self.label_params               # dict, e.g. {"max_rul":125} or {"mu":..., "sigma":...}
    #         # },
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #         'best_score': self.best_score,
    #         'history': self.history,
    #         'config': self.config.__dict__
    #     }
        
    #     filename = f"{self.config.model_name}_{'best' if is_best else 'final'}.pth"
    #     save_path = os.path.join(self.config.save_path, filename)
    #     print(f"Trainer: Saving checkpoint to: {save_path}")
    #     torch.save(checkpoint, save_path)

    def save_checkpoint(self, epoch: int, is_best: bool = False, preprocess: dict | None = None) -> None:
        """
        Save model weights + full training checkpoint.

        Files written:
        - <model_name>_{best|final}.pth   : pure model.state_dict()  (for easy inference / your notebook)
        - <model_name>_{best|final}.ckpt  : full checkpoint (resume + reproducibility)
        """
        import os
        import json
        import torch

        os.makedirs(self.config.save_path, exist_ok=True)
        tag = "best" if is_best else "final"

        # If caller didn’t pass preprocess, try to pick up something attached on the trainer
        if preprocess is None:
            preprocess = getattr(self, "preprocess", None)

        # ---------- (A) Save PURE state_dict for inference ----------
        weights_filename = f"{self.config.model_name}_{tag}.pth"
        weights_path = os.path.join(self.config.save_path, weights_filename)
        torch.save(self.model.state_dict(), weights_path)

        # ---------- (B) Save FULL checkpoint for resume ----------
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_score": self.best_score,
            "history": self.history,
            "config": self.config.__dict__,
            "preprocess": preprocess,  # feature_cols, mean/std, seq_len, max_rul, etc (JSON-serialisable dict)
        }

        ckpt_filename = f"{self.config.model_name}_{tag}.ckpt"
        ckpt_path = os.path.join(self.config.save_path, ckpt_filename)
        torch.save(checkpoint, ckpt_path)

        # Optional: keep a tiny JSON sidecar for human inspection (won’t break anything if it fails)
        try:
            meta = {
                "epoch": epoch,
                "best_score": float(self.best_score) if self.best_score is not None else None,
                "model_name": self.config.model_name,
                "dataset": preprocess.get("dataset") if isinstance(preprocess, dict) else None,
                "seq_len": preprocess.get("sequence_length") if isinstance(preprocess, dict) else None,
                "max_rul": preprocess.get("max_rul") if isinstance(preprocess, dict) else None,
                "n_features": len(preprocess.get("feature_cols", [])) if isinstance(preprocess, dict) else None,
                "weights_file": weights_filename,
                "ckpt_file": ckpt_filename,
            }
            with open(os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'best_score' in checkpoint:
            self.best_score = checkpoint['best_score']
    
    def _print_epoch_summary(self, epoch: int, train_loss: float, val_metrics: Dict[str, float]) -> None:
        """Print detailed epoch summary"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"Epoch [{epoch}/{self.config.num_epochs}] Summary")
        tqdm.write(f"{'='*80}")
        tqdm.write(f"  Train Loss:     {train_loss:.6f}")
        tqdm.write(f"  Val Loss:       {val_metrics['loss']:.6f}")
        tqdm.write(f"  Val RMSE:       {val_metrics['rmse']:.6f}")
        tqdm.write(f"  Val MAE:        {val_metrics['mae']:.6f}")
        tqdm.write(f"  Val R²:         {val_metrics['r2']:.6f}")
        tqdm.write(f"  Val NASA Score: {val_metrics['nasa_score']:.2f}")
        tqdm.write(f"  Learning Rate:  {current_lr:.8f}")
        tqdm.write(f"{'='*80}\n")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Execute the complete training loop.
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*80}")
        print(f"Training Model: {self.config.model_name}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Device: {self.device}")
        print(f"  Save Path: {self.config.save_path}")
        print(f"{'='*80}\n")
        
        # Main training loop
        epoch_progress = tqdm(
            range(1, self.config.num_epochs + 1),
            desc="Overall Progress",
            position=0
        )
        
        for epoch in epoch_progress:
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics, predictions, actuals = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['val_nasa_score'].append(val_metrics['nasa_score'])
            self.history['learning_rate'].append(current_lr)
            
            # Update progress bar
            epoch_progress.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'nasa_score': f"{val_metrics['nasa_score']:.2f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Print detailed summary periodically
            if epoch % self.config.print_every_n_epochs == 0 or epoch == 1:
                self._print_epoch_summary(epoch, train_loss, val_metrics)
            
            # Track best model
            if val_metrics['nasa_score'] < self.best_score:
                self.best_score = val_metrics['nasa_score']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping check
            self.early_stopping(val_metrics['loss'], self.model, epoch)
            
            if self.early_stopping.early_stop:
                tqdm.write(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
            
            # Periodic history saving
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_history()
        
        epoch_progress.close()
        
        # Load best model
        self.early_stopping.load_checkpoint(self.model)
        
        # Save final model and history
        self.save_checkpoint(self.best_epoch, is_best=False)
        self.save_history()
        
        # Print training summary
        print(f"\n{'='*80}")
        print(f"Training Completed!")
        print(f"{'='*80}")
        print(f"  Best Epoch:       {self.best_epoch}")
        print(f"  Best NASA Score:  {self.best_score:.2f}")
        print(f"  Model saved to:   {self.config.save_path}")
        print(f"{'='*80}\n")
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Tuple of (metrics dictionary, predictions array, actuals array)
        """
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        actuals_list = []
        
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)
        
        with torch.no_grad():
            for data, target in progress_bar:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data).squeeze()
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions_list.extend(output.cpu().numpy())
                actuals_list.extend(target.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        progress_bar.close()
        
        predictions = np.array(predictions_list)
        actuals = np.array(actuals_list)
        
        metrics = RULMetrics.calculate_all_metrics(actuals, predictions)
        metrics['loss'] = total_loss / len(test_loader)
        
        return metrics, predictions, actuals
