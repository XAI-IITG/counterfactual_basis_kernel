import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, True, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        self.logits_e = nn.Parameter(torch.zeros(e_layers))
        self.logits_h = nn.Parameter(torch.zeros(n_heads))
        

    def forward(self, x, use_fused_series=False):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        
        if use_fused_series:
            # full_series: list of [B, H, L, L] for each encoder layer
            full_series = torch.stack(series, dim=0)  # [E, B, H, L, L]
            E, B, H, L, _ = full_series.shape

            # Apply softmax to get fusion weights (defined in __init__ earlier)
            w_e = torch.softmax(self.logits_e, dim=0)  # [E]
            w_h = torch.softmax(self.logits_h, dim=0)  # [H]

            # Fuse attentions across encoder layers and heads
            # Broadcast: [E,1,1,1,1] * [1,1,H,1,1] * [E,B,H,L,L] → [E,B,H,L,L]
            fused_series = (w_e.view(E,1,1,1,1) * w_h.view(1,1,H,1,1) * full_series).sum(dim=(0,2))  # [B, L, L]

            # ---- Use fused attention to reconstruct ----
            enc_out = self.embedding(x)  # [B, L, D]

            # Project input to value vectors
            value_proj = self.encoder.attn_layers[0].attention.value_projection(enc_out)
            D_head = value_proj.shape[-1] // H
            V = value_proj.view(B, L, H, D_head)  # [B, L, H, D']
            V = V.permute(0, 2, 1, 3).contiguous()  # [B, H, L, D']

            # Apply fused attention to per-head values
            # fused_series: [B, L, L], V: [B, H, L, D']
            out = torch.einsum("bls,bhsd->bhld", fused_series, V)  # [B, H, L, D']
            out = out.permute(0, 2, 1, 3).contiguous()             # [B, L, H, D']
            out = out.view(B, L, -1)                               # [B, L, H * D']

            # Final projection layer
            out = self.encoder.attn_layers[0].attention.out_projection(out)  # → [B, L, D]

            out = self.projection(out)
            return out


        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
