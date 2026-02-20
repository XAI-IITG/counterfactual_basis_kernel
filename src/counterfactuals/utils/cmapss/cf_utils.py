import torch
import numpy as np

def get_unit_sequences(df, unit_id, sequence_length, feature_cols):
    """Extract all sequences for a specific unit"""
    unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
    features = unit_data[feature_cols].values
    rul = unit_data['RUL'].values
    
    sequences = []
    ruls = []
    cycles = []
    
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
        ruls.append(rul[i+sequence_length-1])
        cycles.append(unit_data['cycle'].iloc[i+sequence_length-1])
    
    return np.array(sequences), np.array(ruls), np.array(cycles)

def get_last_sequence(df, unit_id, sequence_length, feature_cols):
    """Get the last sequence for a unit (typical test scenario)"""
    unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
    features = unit_data[feature_cols].values
    rul = unit_data['RUL'].values[-1]
    
    if len(features) >= sequence_length:
        return features[-sequence_length:], rul
    else:
        return None, None

def predict_rul(model, sequence, device):
    """Predict RUL for a single sequence - handles both numpy and tensor inputs"""
    model.eval()
    with torch.no_grad():
        # Convert to tensor if numpy
        if isinstance(sequence, np.ndarray):
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        elif isinstance(sequence, torch.Tensor):
            # Already a tensor
            if sequence.dim() == 2:
                seq_tensor = sequence.unsqueeze(0).to(device)
            else:
                seq_tensor = sequence.to(device)
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(sequence)}")
        
        # Get prediction
        pred = model(seq_tensor)
        
        # Handle different output shapes
        if pred.dim() > 1:
            pred = pred.squeeze()
        
        # Convert to scalar
        if pred.dim() == 0:
            return pred.item()
        else:
            return pred[0].item() if len(pred) == 1 else pred.mean().item()

def get_valid_target_rul(current_rul, increase_range=(30, 50), max_rul=125):
    """
    Generate a valid target RUL for counterfactual generation.
    
    Args:
        current_rul: Current predicted RUL
        increase_range: Tuple of (min_increase, max_increase) in cycles
        max_rul: Maximum physically meaningful RUL (from dataset)
    
    Returns:
        target_rul: Valid target RUL value
    """
    min_increase, max_increase = increase_range
    
    # Ensure we don't exceed maximum RUL cap
    available_increase = max_rul - current_rul
    max_feasible_increase = min(max_increase, available_increase)
    
    if max_feasible_increase < min_increase:
        # If current RUL is already too high, use smaller increase
        target_rul = min(current_rul + min_increase, max_rul)
        print(f"⚠️ Current RUL ({current_rul:.1f}) is close to max. Target adjusted to {target_rul:.1f}")
    else:
        # Normal case: random increase within range
        increase = np.random.uniform(min_increase, max_feasible_increase)
        target_rul = current_rul + increase
    
    return max(target_rul, 1.0)  # Ensure target is always positive
