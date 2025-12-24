import torch
import torch.nn as nn
import timm

from src.models.vim_bimamba import BiMambaEncoder


class EffVimClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        out_index: int,
        mamba_dim: int,
        mamba_layers: int,
        mamba_d_state: int,
        dropout: float,
    ):
        super().__init__()

        # features_only returns list of feature maps from stages
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        self.out_index = out_index

        self.encoder = None  # lazy init after we see C,H,W

        self.mamba_dim = mamba_dim
        self.mamba_layers = mamba_layers
        self.mamba_d_state = mamba_d_state
        self.dropout = dropout

        self.classifier = None  # lazy init

    def _build_heads(self, feat: torch.Tensor):
        # feat: (B, C, H, W)
        _, c, h, w = feat.shape
        self.encoder = BiMambaEncoder(
            in_channels=c,
            dim=self.mamba_dim,
            n_layers=self.mamba_layers,
            d_state=self.mamba_d_state,
            dropout=self.dropout,
        )
        self.classifier = nn.Linear(self.mamba_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feat = feats[self.out_index]

        if self.encoder is None or self.classifier is None:
            self._build_heads(feat)
            # move lazy-created modules to same device as feature map
            self.encoder = self.encoder.to(feat.device)
            self.classifier = self.classifier.to(feat.device)

        z = self.encoder(feat)
        logits = self.classifier(z)

        return logits

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def head_parameters(self):
        params = []
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        if self.classifier is not None:
            params += list(self.classifier.parameters())
        return params
