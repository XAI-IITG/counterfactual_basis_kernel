import torch
import numpy as np
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', start_epoch=1):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'saved_model/checkpoint.pt'
            start_epoch (int): Epoch number (1-based) from which to start monitoring for early stopping.
                               Saving the best model will still occur before this epoch.
                               Default: 1 (start immediately)
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.start_epoch = start_epoch # Store the epoch to start checking
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        # Ensure the directory for the path exists during initialization
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)


    def __call__(self, val_loss, model, epoch): # Add epoch parameter
        """
        Args:
            val_loss (float): Validation loss for the current epoch.
            model (torch.nn.Module): Model to save if validation loss improves.
            epoch (int): The current epoch number (1-based).
        """
        score = -val_loss

        # --- Check for improvement and save best model ALWAYS ---
        if self.best_score is None:
            # First call or first improvement
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0 # Reset counter on improvement
        elif score > self.best_score + self.delta:
            # Significant improvement detected
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0 # Reset counter on improvement
        else:
            # --- No improvement ---
            # Only increment counter and check for stopping *after* start_epoch
            if epoch >= self.start_epoch:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Epoch {epoch} >= Start Epoch {self.start_epoch})')
                if self.counter >= self.patience:
                    self.early_stop = True
            # Optional: Print message if before start_epoch and verbose
            elif self.verbose:
                 print(f'Epoch {epoch}: Early stopping checks deferred until epoch {self.start_epoch}. Loss did not improve.')

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            # Ensure val_loss_min is updated *before* printing if it's the first save
            old_min = self.val_loss_min
            print(f'Validation loss decreased ({old_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        # Ensure the directory exists before saving (redundant if checked in init, but safe)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss # Update the minimum loss

    def load_checkpoint(self, model):
        '''Loads the saved model state dictionary.'''
        if os.path.exists(self.path):
            model.load_state_dict(torch.load(self.path))
            print(f"Loaded checkpoint from '{self.path}'")
        else:
            print(f"Checkpoint file not found at '{self.path}', model weights not loaded.")