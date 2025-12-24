import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig


class BiMambaEncoder(nn.Module):
    """
    Takes a CNN feature map (B, C, H, W), flattens to tokens (B, L, C),
    projects to dim, adds learnable pos emb, then runs a forward Mamba and
    a backward Mamba and merges them. Outputs a pooled vector (B, dim).
    """
    def __init__(self, in_channels: int, dim: int, n_layers: int, d_state: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_channels, dim)
        self.drop = nn.Dropout(dropout)

        cfg = MambaConfig(d_model=dim, n_layers=n_layers, d_state=d_state)
        self.mamba_f = Mamba(cfg)
        self.mamba_b = Mamba(cfg)

        self.norm = nn.LayerNorm(dim)

        self.pos_emb = None  # lazy init because token count L depends on backbone output

    def _maybe_init_pos(self, L: int, dim: int, device):
        if self.pos_emb is None or self.pos_emb.shape[1] != L:
            self.pos_emb = nn.Parameter(torch.zeros(1, L, dim, device=device))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W)
        b, c, h, w = feat.shape
        x = feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # (B, L, C)
        x = self.proj(x)  # (B, L, dim)

        self._maybe_init_pos(x.shape[1], x.shape[2], x.device)
        x = x + self.pos_emb
        x = self.drop(x)

        y_f = self.mamba_f(x)  # (B, L, dim)
        x_rev = torch.flip(x, dims=[1])
        y_b = self.mamba_b(x_rev)
        y_b = torch.flip(y_b, dims=[1])

        y = y_f + y_b
        y = self.norm(y)

        # GAP over tokens
        z = y.mean(dim=1)  # (B, dim)
        return z
