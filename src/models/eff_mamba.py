import torch
import torch.nn as nn
import timm

from src.models.mamba_blocks import BiMambaEncoder


class EffMambaClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        out_index: int,
        image_size: int,
        mamba_dim: int,
        mamba_layers: int,
        mamba_d_state: int,
        dropout: float,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.out_index = out_index

        # Build head with dummy forward so shapes are fixed
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.backbone(dummy)[out_index]  # (1, C, H, W)
            _, c, h, w = feat.shape
            max_tokens = h * w

        self.encoder = BiMambaEncoder(
            in_channels=c,
            dim=mamba_dim,
            n_layers=mamba_layers,
            d_state=mamba_d_state,
            dropout=dropout,
            max_tokens=max_tokens,
        )
        self.classifier = nn.Linear(mamba_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[self.out_index]
        z = self.encoder(feat)
        return self.classifier(z)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def backbone_params(self):
        return list(self.backbone.parameters())

    def head_params(self):
        return list(self.encoder.parameters()) + list(self.classifier.parameters())
