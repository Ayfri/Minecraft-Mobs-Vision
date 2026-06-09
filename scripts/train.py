"""Training script: classification + bounding box regression on Minecraft mobs."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import make_splits
from src.metrics import bbox_iou, ciou_loss
from src.model import MobDetector

DATA_DIR        = Path("data")
CKPT_DIR        = Path("checkpoints")
BATCH_SIZE      = 96
EPOCHS          = 50
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 4
BBOX_WEIGHT     = 1.0   # CIoU loss is in [0,~2] range, well-balanced with cls loss
PATIENCE        = 8     # stop early if val accuracy plateaus to avoid overfitting
WARMUP_EPOCHS   = 5     # freeze backbone, train heads only before full fine-tuning
BACKBONE_LR_MUL = 0.1  # backbone LR relative to heads after warmup
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cls_fn: nn.CrossEntropyLoss,
) -> tuple[float, float, float]:
    model.train()
    total = cls_sum = bbox_sum = 0.0

    for imgs, labels, bboxes in tqdm(loader, desc="train", leave=False):
        imgs   = imgs.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)
        bboxes = bboxes.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            cls_logits, bbox_pred = model(imgs)
            cls_loss  = cls_fn(cls_logits, labels)
            bbox_loss = ciou_loss(bbox_pred.float(), bboxes.float())
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
) -> tuple[float, float, float]:
    model.eval()
    total = correct = samples = 0
    iou_sum = 0.0

    for imgs, labels, bboxes in tqdm(loader, desc="val  ", leave=False):
        imgs   = imgs.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)
        bboxes = bboxes.to(DEVICE)

        with torch.amp.autocast(device_type="cuda"):
            cls_logits, bbox_pred = model(imgs)
            cls_loss  = cls_fn(cls_logits, labels)
            bbox_loss = ciou_loss(bbox_pred.float(), bboxes.float())
            loss      = cls_loss + BBOX_WEIGHT * bbox_loss

        total   += loss.item()
        correct += (cls_logits.argmax(1) == labels).sum().item()
        samples += labels.size(0)
        iou_sum += bbox_iou(bbox_pred.float(), bboxes.float()).item()

    n = len(loader)
    return total / n, correct / samples, iou_sum / n


def main() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = True         # auto-tune conv kernels for fixed input size
    torch.set_float32_matmul_precision("high")    # TF32 on Ampere/Ada — 2× faster matmuls
    print(f"Device: {DEVICE}")

    train_ds, val_ds, _ = make_splits(DATA_DIR)
    num_classes = len(train_ds.classes)
    print(f"{train_ds} | {val_ds}")

    train_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )

    model = MobDetector(num_classes).to(DEVICE, memory_format=torch.channels_last)
    try:
        import triton  # noqa: F401
        compiled: nn.Module = torch.compile(model)  # type: ignore[assignment]
    except ImportError:
        compiled = model

    cls_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # Phase 1 - warmup: freeze backbone, only heads receive gradients
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=WARMUP_EPOCHS,
    )
    print(f"Phase 1: backbone frozen for {WARMUP_EPOCHS} warmup epochs.")

    # Combined metric weights acc+iou to drive both tasks toward a good checkpoint
    best_score = 0.0
    no_improve  = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            # Phase 2 - full fine-tune: backbone at lower LR, heads at full LR
            for p in model.backbone.parameters():
                p.requires_grad = True
            head_params = list(model.cls_head.parameters()) + list(model.bbox_head.parameters())
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": LR * BACKBONE_LR_MUL},
                    {"params": head_params, "lr": LR},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - WARMUP_EPOCHS,
            )
            print(f"Phase 2: backbone unfrozen (lr={LR * BACKBONE_LR_MUL:.1e}), heads lr={LR:.1e}")

        t0 = time.perf_counter()
        tr_loss, tr_cls, tr_bbox = train_epoch(compiled, train_loader, optimizer, scaler, cls_fn)
        val_loss, val_acc, val_iou = eval_epoch(compiled, val_loader, cls_fn)
        scheduler.step()

        print(
            f"[{epoch:02d}/{EPOCHS}] {time.perf_counter() - t0:.1f}s | "
            f"train {tr_loss:.4f} (cls {tr_cls:.4f} bbox {tr_bbox:.4f}) | "
            f"val {val_loss:.4f}  acc {val_acc:.3f}  iou {val_iou:.3f}"
        )

        score = 0.7 * val_acc + 0.3 * val_iou
        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save(model.state_dict(), CKPT_DIR / "best.pth")
            print(f"  → best model saved (acc={val_acc:.3f}  iou={val_iou:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop: no improvement for {PATIENCE} epochs.")
                break

    torch.save(model.state_dict(), CKPT_DIR / "last.pth")
    print(f"Done. Best combined score: {best_score:.3f}")


if __name__ == "__main__":
    main()
