"""Training script for MobDetector — classification + bounding box regression."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import make_splits
from src.model import MobDetector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
BBOX_WEIGHT = 5.0  # Scale bbox loss relative to cls loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bbox_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean IoU between YOLO-format (cx, cy, w, h) boxes. Both tensors: (B, 4)."""
    p_x1, p_y1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
    p_x2, p_y2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    t_x1, t_y1 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
    t_x2, t_y2 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2

    inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
            (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    union = (p_x2 - p_x1) * (p_y2 - p_y1) + (t_x2 - t_x1) * (t_y2 - t_y1) - inter
    return (inter / union.clamp(min=1e-6)).mean()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_epoch(
    model: MobDetector,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cls_fn: nn.CrossEntropyLoss,
    bbox_fn: nn.SmoothL1Loss,
) -> tuple[float, float, float]:
    model.train()
    total = cls_sum = bbox_sum = 0.0

    for imgs, labels, bboxes in tqdm(loader, desc="train", leave=False):
        imgs, labels, bboxes = imgs.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            cls_logits, bbox_pred = model(imgs)
            cls_loss = cls_fn(cls_logits, labels)
            bbox_loss = bbox_fn(bbox_pred, bboxes)
            loss = cls_loss + BBOX_WEIGHT * bbox_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        cls_sum += cls_loss.item()
        bbox_sum += bbox_loss.item()

    n = len(loader)
    return total / n, cls_sum / n, bbox_sum / n


@torch.no_grad()
def eval_epoch(
    model: MobDetector,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]],
    cls_fn: nn.CrossEntropyLoss,
    bbox_fn: nn.SmoothL1Loss,
) -> tuple[float, float, float]:
    model.eval()
    total = correct = samples = 0
    iou_sum = 0.0

    for imgs, labels, bboxes in tqdm(loader, desc="val  ", leave=False):
        imgs, labels, bboxes = imgs.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE)

        with torch.amp.autocast(device_type="cuda"):
            cls_logits, bbox_pred = model(imgs)
            cls_loss = cls_fn(cls_logits, labels)
            bbox_loss = bbox_fn(bbox_pred, bboxes)
            loss = cls_loss + BBOX_WEIGHT * bbox_loss

        total += loss.item()
        correct += (cls_logits.argmax(1) == labels).sum().item()
        samples += labels.size(0)
        iou_sum += bbox_iou(bbox_pred.float(), bboxes.float()).item()

    n = len(loader)
    return total / n, correct / samples, iou_sum / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")

    train_ds, val_ds, _ = make_splits(DATA_DIR)
    num_classes = len(train_ds.classes)
    print(f"Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model: MobDetector = torch.compile(MobDetector(num_classes).to(DEVICE))  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler()
    cls_fn = nn.CrossEntropyLoss()
    bbox_fn = nn.SmoothL1Loss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_cls, tr_bbox = train_epoch(model, train_loader, optimizer, scaler, cls_fn, bbox_fn)
        val_loss, val_acc, val_iou = eval_epoch(model, val_loader, cls_fn, bbox_fn)
        scheduler.step()

        print(
            f"[{epoch:02d}/{EPOCHS}] {time.perf_counter() - t0:.1f}s | "
            f"train {tr_loss:.4f} (cls {tr_cls:.4f} bbox {tr_bbox:.4f}) | "
            f"val {val_loss:.4f}  acc {val_acc:.3f}  iou {val_iou:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")
            print(f"  → best model saved (acc={best_acc:.3f})")

    torch.save(model.state_dict(), CKPT_DIR / "last.pth")
    print(f"Done. Best val acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()
