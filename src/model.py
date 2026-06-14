import timm
import torch
import torch.nn as nn

from src.config import cfg


class MobDetector(nn.Module):
    """Multi-task model: mob classification + bounding box regression.

    Uses a timm backbone (default: MobileNetV4-small) so the architecture can be
    swapped by changing `backbone` in config.toml without touching this file.

    Forward returns (cls_logits, bbox_pred) where:
        cls_logits: (B, num_classes)
        bbox_pred:  (B, 4) - YOLO-normalized (cx, cy, w, h) in [0, 1]
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = cfg.model.backbone,
        dropout: float = cfg.model.dropout,
    ) -> None:
        super().__init__()
        # num_classes=0 + global_pool="" makes the backbone return the spatial feature
        # map (B, C, H', W') instead of a flat vector, so GradCAM can hook it directly.
        self.backbone: nn.Module = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool=""
        )
        # Dummy forward pass to derive feat_dim without hardcoding per-backbone values.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *cfg.model.img_size)
            feat_map = self.backbone(dummy)
            feat_dim: int = feat_map.shape[1]

        self.pool_cls = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)

        # 4x4 spatial pool after channel reduction preserves finer location info
        # than global pool, with far fewer params than a direct linear on H'xW'.
        _BBOX_POOL = 4
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(_BBOX_POOL),
            nn.Flatten(1),
            nn.Linear(256 * _BBOX_POOL * _BBOX_POOL, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.backbone(x)                                          # (B, C, H', W')
        feats_cls = self.drop(self.pool_cls(feat_map).flatten(1))            # (B, C)
        cls_logits = self.cls_head(feats_cls)                                # (B, num_classes)
        bbox_pred = self.bbox_head(feat_map).sigmoid()                       # (B, 4)
        return cls_logits, bbox_pred
