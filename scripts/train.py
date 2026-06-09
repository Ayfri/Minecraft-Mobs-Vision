"""Training script: classification + bounding box regression on Minecraft mobs."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import make_splits
from src.metrics import bbox_iou
from src.model import MobDetector

DATA_DIR     = Path("data")
CKPT_DIR     = Path("checkpoints")
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
BBOX_WEIGHT  = 5.0  # bbox loss is much smaller in magnitude than cls, rebalance
PATIENCE     = 8    # stop early if val accuracy plateaus to avoid overfitting
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
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
            cls_loss  = cls_fn(cls_logits, labels)
            bbox_loss = bbox_fn(bbox_pred, bboxes)
            loss      = cls_loss + BBOX_WEIGHT * bbox_loss

        scaler.scale(loss).backward()  # type: ignore[union-attr]
        scaler.step(optimizer)
        scaler.update()

        total    += loss.item()
        cls_sum  += cls_loss.item()
        bbox_sum += bbox_loss.item()

    n = len(loader)
    return total / n, cls_sum / n, bbox_sum / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
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
            cls_loss  = cls_fn(cls_logits, labels)
            bbox_loss = bbox_fn(bbox_pred, bboxes)
            loss      = cls_loss + BBOX_WEIGHT * bbox_loss

        total   += loss.item()
        correct += (cls_logits.argmax(1) == labels).sum().item()
        samples += labels.size(0)
        iou_sum += bbox_iou(bbox_pred.float(), bboxes.float()).item()

    n = len(loader)
    return total / n, correct / samples, iou_sum / n


def main() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")

    train_ds, val_ds, _ = make_splits(DATA_DIR)
    num_classes = len(train_ds.classes)
    print(f"{train_ds} | {val_ds}")

    train_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model    = MobDetector(num_classes).to(DEVICE)
    try:
        import triton  # noqa: F401
        compiled: nn.Module = torch.compile(model)  # type: ignore[assignment]
    except ImportError:
        compiled = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler()
    cls_fn    = nn.CrossEntropyLoss()
    bbox_fn   = nn.SmoothL1Loss()

    best_acc = 0.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_cls, tr_bbox = train_epoch(compiled, train_loader, optimizer, scaler, cls_fn, bbox_fn)
        val_loss, val_acc, val_iou = eval_epoch(compiled, val_loader, cls_fn, bbox_fn)
        scheduler.step()

        print(
            f"[{epoch:02d}/{EPOCHS}] {time.perf_counter() - t0:.1f}s | "
            f"train {tr_loss:.4f} (cls {tr_cls:.4f} bbox {tr_bbox:.4f}) | "
            f"val {val_loss:.4f}  acc {val_acc:.3f}  iou {val_iou:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")
            print(f"  → best model saved (acc={best_acc:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop: no improvement for {PATIENCE} epochs.")
                break

    torch.save(model.state_dict(), CKPT_DIR / "last.pth")
    print(f"Done. Best val acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()
