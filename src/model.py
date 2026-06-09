import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

_FEAT_DIM = 1280  # EfficientNet-B0 output channels


class MobDetector(nn.Module):
    """Multi-task model: mob classification + bounding box regression.

    Forward returns (cls_logits, bbox_pred) where:
        cls_logits: (B, num_classes)
        bbox_pred:  (B, 4) - YOLO-normalized (cx, cy, w, h) in [0, 1]
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = base.features          # (B, 3, H, W) → (B, 1280, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)   # → (B, 1280, 1, 1)
        self.drop = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(_FEAT_DIM, num_classes)
        self.bbox_head = nn.Linear(_FEAT_DIM, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.drop(self.pool(self.backbone(x)).flatten(1))  # (B, 1280)
        cls_logits = self.cls_head(feats)                           # (B, num_classes)
        bbox_pred = self.bbox_head(feats).sigmoid()                 # (B, 4)
        return cls_logits, bbox_pred
