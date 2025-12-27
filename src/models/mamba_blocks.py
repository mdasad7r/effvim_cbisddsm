import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig


class BiMambaEncoder(nn.Module):
    """
    CNN feature map (B, C, H, W) -> tokens (B, L, C) -> proj -> pos emb
    -> forward Mamba + backward Mamba -> fuse -> LayerNorm -> GAP -> (B, dim)
    """
    def __init__(self, in_channels: int, dim: int, n_layers: int, d_state: int, dropout: float, max_tokens: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, dim)
        self.drop = nn.Dropout(dropout)

        cfg = MambaConfig(d_model=dim, n_layers=n_layers, d_state=d_state)
        self.mamba_f = Mamba(cfg)
        self.mamba_b = Mamba(cfg)

        self.norm = nn.LayerNorm(dim)

        # fixed-size learned pos emb, sized for max_tokens, then sliced
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        x = feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # (B, L, C)
        x = self.proj(x)  # (B, L, dim)

        L = x.shape[1]
        x = x + self.pos_emb[:, :L, :]
        x = self.drop(x)

        y_f = self.mamba_f(x)  # (B, L, dim)
        y_b = self.mamba_b(torch.flip(x, dims=[1]))
        y_b = torch.flip(y_b, dims=[1])

        y = self.norm(y_f + y_b)
        return y.mean(dim=1)  # (B, dim)
