"""
Training Script for RUL Prediction Models

This script trains multiple deep learning models (LSTM, GRU, CNN-LSTM, Transformer)
on the CMAPSS dataset for Remaining Useful Life prediction.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.cmapss.CMAPSSDataLoader import CMAPSSDataLoader
from src.data_loader.cmapss.CMAPSSTimeSeriesDataset import CMAPSSTimeSeriesDataset
from src.data_loader.cmapss.CMAPSSDatasetWrapper import CMAPSSDatasetWrapper
from src.models.LSTMModel import LSTMModel
from src.models.GRUModel import GRUModel
from src.models.CNNLSTMModel import CNNLSTMModel
from src.models.TransformerModel import TransformerModel
from src.trainer.Trainer import Trainer, TrainingConfig
from src.utils.plots import plot_training_history, plot_predictions, plot_model_comparison


def load_and_preprocess_data(dataset_name='FD001', sequence_length=50, batch_size=32):
    """
    Load and preprocess CMAPSS dataset.
    
    Args:
        dataset_name: Name of the dataset (FD001, FD002, FD003, FD004)
        sequence_length: Length of input sequences
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, feature_cols)
    """
    print(f"\n{'='*80}")
    print(f"Loading and Preprocessing {dataset_name}")
    print(f"{'='*80}\n")
    
    # Initialize data loader
    loader = CMAPSSDataLoader()
    train_df, test_df, rul_true = loader.load_dataset(dataset_name)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Add RUL to training data
    train_df = loader.add_rul(train_df, max_rul=125)
    
    # Add RUL to test data
    test_df_with_rul = test_df.copy()
    max_cycles_test = test_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles_test['RUL'] = rul_true['RUL'].values
    test_df_with_rul = test_df_with_rul.merge(
        max_cycles_test[['unit_id', 'RUL']], 
        on='unit_id', 
        how='left'
    )
    
    # Remove constant features
    train_df, test_df_with_rul, feature_cols = loader.remove_constant_features(
        train_df, test_df_with_rul, threshold=0.01
    )
    
    # Normalize data
    train_norm, test_norm, mean_stats, std_stats = loader.normalize_data(
        train_df, test_df_with_rul, feature_cols
    )
    
    print(f"\nPreprocessed data:")
    print(f"  Train: {train_norm.shape}")
    print(f"  Test: {test_norm.shape}")
    print(f"  Features: {len(feature_cols)}")
    
    # Create training dataset
    train_dataset = CMAPSSTimeSeriesDataset(
        train_norm, 
        sequence_length=sequence_length,
        feature_cols=feature_cols
    )
    
    # Create test sequences (last sequence of each unit)
    test_sequences = []
    test_targets = []
    for unit_id in test_norm['unit_id'].unique():
        unit_data = test_norm[test_norm['unit_id'] == unit_id].sort_values('cycle')
        features = unit_data[feature_cols].values
        rul = unit_data['RUL'].values
        
        if len(features) >= sequence_length:
            test_sequences.append(features[-sequence_length:])
            test_targets.append(rul[-1])
    
    test_sequences = np.array(test_sequences, dtype=np.float32)
    test_targets = np.array(test_targets, dtype=np.float32)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_dataset = CMAPSSDatasetWrapper(test_sequences, test_targets)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, feature_cols


def create_model(model_type, input_size, device):
    """
    Create and initialize a model.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'cnn_lstm', 'transformer')
        input_size: Number of input features
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    if model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
    elif model_type == 'gru':
        model = GRUModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
    elif model_type == 'cnn_lstm':
        model = CNNLSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{model_type.upper()} Model:")
    print(f"  Total parameters: {num_params:,}")
    
    return model


def train_single_model(
    model_type,
    train_loader,
    val_loader,
    input_size,
    device,
    num_epochs=50,
    learning_rate=0.001,
    save_path='../outputs/saved_models'
):
    """
    Train a single model.
    
    Args:
        model_type: Type of model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        input_size: Number of input features
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Path to save model
        
    Returns:
        Training history
    """
    # Create model
    model = create_model(model_type, input_size, device)
    
    # Create training configuration
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=32,
        early_stopping_patience=25,
        early_stopping_start_epoch=80,
        gradient_clip_value=1.0,
        save_path=save_path,
        model_name=f"{model_type}_model"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train model
    history = trainer.train()
    
    return history, trainer


def main():
    """Main training script"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print(f"\nUsing device: {device}")
    
    # Load and preprocess data
    train_loader, test_loader, feature_cols = load_and_preprocess_data(
        dataset_name='FD001',
        sequence_length=50,
        batch_size=32
    )
    
    input_size = len(feature_cols)
    
    # Define models to train
    models_config = {
        'lstm': {'lr': 0.001, 'epochs': 160},
        'gru': {'lr': 0.001, 'epochs': 160},
        'cnn_lstm': {'lr': 0.001, 'epochs': 160},
        'transformer': {'lr': 0.0005, 'epochs': 160}
    }
    
    # Train all models
    histories = {}
    trainers = {}
    
    for model_type, config in models_config.items():
        print(f"\n{'#'*80}")
        print(f"# Training {model_type.upper()} Model")
        print(f"{'#'*80}\n")
        
        history, trainer = train_single_model(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=test_loader,
            input_size=input_size,
            device=device,
            num_epochs=config['epochs'],
            learning_rate=config['lr'],
            save_path='../outputs/saved_models'
        )
        
        histories[model_type] = history
        trainers[model_type] = trainer
    
    # Plot comparison of all models
    print(f"\n{'='*80}")
    print("Generating Comparison Plots")
    print(f"{'='*80}\n")
    
    plot_training_history(
        list(histories.values()),
        [name.upper() for name in histories.keys()]
    )
    
    # Plot predictions for each model
    for model_type, trainer in trainers.items():
        plot_predictions(
            trainer.model,
            test_loader,
            device,
            model_type.upper()
        )
    
    # Generate model comparison report
    plot_model_comparison(histories, list(histories.keys()))
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()