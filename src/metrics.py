import math

import torch


def _to_corners(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert YOLO (cx, cy, w, h) to (x1, y1, x2, y2)."""
    return (
        boxes[:, 0] - boxes[:, 2] / 2,
        boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2,
    )


def bbox_iou(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """IoU for YOLO-format (cx, cy, w, h) boxes. Both tensors: (B, 4).

    reduction: "mean" → scalar, "none" → per-sample tensor (B,).
    """
    p_x1, p_y1, p_x2, p_y2 = _to_corners(pred)
    t_x1, t_y1, t_x2, t_y2 = _to_corners(target)
    inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
            (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    union = (p_x2 - p_x1) * (p_y2 - p_y1) + (t_x2 - t_x1) * (t_y2 - t_y1) - inter
    iou = inter / union.clamp(min=1e-6)
    return iou.mean() if reduction == "mean" else iou


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complete IoU loss for YOLO-format (cx, cy, w, h) boxes. Both: (B, 4).

    Combines IoU, center-distance penalty, and aspect-ratio consistency.
    Returns scalar mean loss in [0, ~2].
    """
    p_x1, p_y1, p_x2, p_y2 = _to_corners(pred)
    t_x1, t_y1, t_x2, t_y2 = _to_corners(target)

    inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
            (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    union = p_area + t_area - inter
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box diagonal²
    c2 = (torch.min(p_x1, t_x1) - torch.max(p_x2, t_x2)).pow(2) + \
         (torch.min(p_y1, t_y1) - torch.max(p_y2, t_y2)).pow(2)

    # Center distance²
    rho2 = (pred[:, 0] - target[:, 0]).pow(2) + (pred[:, 1] - target[:, 1]).pow(2)

    # Aspect ratio consistency
    v = (4 / math.pi ** 2) * (
        torch.atan(target[:, 2] / target[:, 3].clamp(min=1e-6)) -
        torch.atan(pred[:, 2] / pred[:, 3].clamp(min=1e-6))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=1e-6)

    ciou = iou - rho2 / c2.clamp(min=1e-6) - alpha * v
    return (1 - ciou).mean()
