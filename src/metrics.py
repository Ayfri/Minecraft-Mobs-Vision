import torch


def bbox_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean IoU for YOLO-format (cx, cy, w, h) boxes. Both tensors: (B, 4)."""
    p_x1, p_y1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
    p_x2, p_y2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    t_x1, t_y1 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
    t_x2, t_y2 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2

    inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
            (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    union = (p_x2 - p_x1) * (p_y2 - p_y1) + (t_x2 - t_x1) * (t_y2 - t_y1) - inter
    return (inter / union.clamp(min=1e-6)).mean()
