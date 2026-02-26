import torch
import torch.nn as nn

class BearingTransformerModel(nn.Module):
    """
    Transformer with:
    - 1D Conv input smoothing (reduces vibration noise)
    - Heavier dropout for cross-bearing generalization
    - Mean pooling instead of last-token (more robust)
    """
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_ff: int = 256, dropout: float = 0.3):
        super().__init__()
        
        # 1D conv to smooth noisy vibration features
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_size, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, x, return_hidden=False):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # Conv expects (B, D, T)
        h = self.input_conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        h = h + self.pos_enc[:, :T, :]
        h = self.transformer(h)
        h = self.norm(h)
        
        # Mean pooling over time (more robust than last-token)
        h_pool = h.mean(dim=1)  # (B, d_model)
        out = self.head(h_pool)  # (B, 1)
        
        if return_hidden:
            return out, {"transformer_out": h}
        return out
