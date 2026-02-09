import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.EarlyStopping import EarlyStopping
import json
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.best_val_loss = float('inf')

    def train_epoch(self, model, train_loader, criterion, optimizer, device, epoch, num_epochs):
        """Train for one epoch with progress bar"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Train", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        progress_bar.close()
        return avg_loss

    def evaluate(self, model, test_loader, criterion, device, epoch, num_epochs):
        """Evaluate model with progress bar"""
        model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{num_epochs} Val  ", leave=False)
        
        with torch.no_grad():
            for data, target in progress_bar:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()
                loss = criterion(output, target)
                total_loss += loss.item()
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # RUL-specific score (NASA scoring function)
        def score_func(y_true, y_pred):
            diff = y_pred - y_true
            score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
            return score
        
        score = score_func(actuals, predictions)
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'score': score
        }
        
        progress_bar.close()
        return metrics, predictions, actuals

    def save_history_to_json(self, history, save_path, model_name):
        """Save training history to JSON file"""
        os.makedirs(save_path, exist_ok=True)
        json_path = os.path.join(save_path, f"{model_name}_history.json")
        
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, values in history.items():
            history_serializable[key] = [float(v) for v in values]
        
        with open(json_path, 'w') as f:
            json.dump(history_serializable, f, indent=4)
        
        print(f"Training history saved to: {json_path}")

    def train_model(self, model, train_loader, test_loader, epochs=50, lr=0.001, 
                model_name='model', save_path='../outputs/saved_models',
                early_stopping_patience=15):
        """Complete training loop with tqdm, JSON logging, and early stopping"""
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=5)
        
        # Initialize early stopping
        os.makedirs(save_path, exist_ok=True)
        early_stopping_path = os.path.join(save_path, f"{model_name}_checkpoint.pth")
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True,
            path=early_stopping_path,
            start_epoch=10  # Start monitoring after 10 epochs
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'val_r2': [],
            'val_score': [],
            'learning_rate': []
        }
        
        best_score = float('inf')
        best_epoch = 0
        
        print(f"\nTraining {model_name}...")
        print("="*80)
        
        # Main training loop with progress bar
        epoch_progress = tqdm(range(epochs), desc="Overall Progress")
        
        for epoch in epoch_progress:
            current_epoch = epoch + 1
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, 
                                    device, current_epoch, epochs)
            
            # Evaluate
            val_metrics, _, _ = self.evaluate(model, test_loader, criterion, device, 
                                        current_epoch, epochs)
            
            # Update scheduler
            scheduler.step(val_metrics['loss'])
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_r2'].append(val_metrics['r2'])
            history['val_score'].append(val_metrics['score'])
            history['learning_rate'].append(current_lr)
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'val_score': f"{val_metrics['score']:.2f}",
                'lr': f"{current_lr:.6f}"
            })
            
            # Print detailed progress
            if current_epoch % 5 == 0 or current_epoch == 1:
                tqdm.write(f"\nEpoch [{current_epoch}/{epochs}]")
                tqdm.write(f"  Train Loss: {train_loss:.4f}")
                tqdm.write(f"  Val Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                        f"MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}, Score: {val_metrics['score']:.2f}")
                tqdm.write(f"  Learning Rate: {current_lr:.6f}")
            
            # Track best model
            if val_metrics['score'] < best_score:
                best_score = val_metrics['score']
                best_epoch = current_epoch
            
            # Early stopping check
            early_stopping(val_metrics['loss'], model, current_epoch)
            
            if early_stopping.early_stop:
                tqdm.write(f"\nEarly stopping triggered at epoch {current_epoch}")
                break
            
            # Save history periodically (every 10 epochs)
            if current_epoch % 10 == 0:
                self.save_history_to_json(history, save_path, model_name)
        
        epoch_progress.close()
        
        # Load best model
        early_stopping.load_checkpoint(model)
        
        # Save final model and history
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'history': history
        }, os.path.join(save_path, f"{model_name}_final.pth"))
        
        # Save final history to JSON
        self.save_history_to_json(history, save_path, model_name)
        
        print(f"\nTraining completed!")
        print(f"Best epoch: {best_epoch}, Best score: {best_score:.2f}")
        print(f"Model saved to: {save_path}/{model_name}_final.pth")
        print("="*80)
        
        return history
