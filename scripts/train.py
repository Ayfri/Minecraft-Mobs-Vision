"""Training script: classification + bounding box regression on Minecraft mobs."""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import cfg
from src.dataset import make_splits
from src.metrics import bbox_iou, ciou_loss
from src.model import MobDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            loss      = cls_loss + cfg.training.bbox_weight * bbox_loss

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
            loss      = cls_loss + cfg.training.bbox_weight * bbox_loss

        total   += loss.item()
        correct += (cls_logits.argmax(1) == labels).sum().item()
        samples += labels.size(0)
        iou_sum += bbox_iou(bbox_pred.float(), bboxes.float()).item()

    n = len(loader)
    return total / n, correct / samples, iou_sum / n


def main() -> None:
    t = cfg.training
    d = cfg.data
    m = cfg.model

    cfg.data.ckpt_dir.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")    # TF32 on Ampere/Ada, ~2x faster matmuls
    print(f"Device: {DEVICE}")
    print(f"Backbone: {m.backbone}  img_size={m.img_size}  "
          f"equalize_v={cfg.augmentation.equalize_v}  posterize_bits={cfg.augmentation.posterize_bits}")

    train_ds, val_ds, _ = make_splits(d.data_dir)
    num_classes = len(train_ds.classes)
    print(f"{train_ds} | {val_ds}")

    # drop_last keeps every train batch at batch_size, so torch.compile sees one
    # static shape and compiles once instead of recompiling for a partial last batch.
    train_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        train_ds, batch_size=t.batch_size, shuffle=True,
        num_workers=t.num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=t.prefetch_factor, drop_last=True,
    )
    # val runs in the main process (num_workers=0): its transform is trivial (no
    # augmentation), so a single thread hits 0.7s/first-batch. Spawning a second
    # persistent worker pool while train's pool is alive + CUDA is initialized
    # deadlocks on Windows, which is what froze the run at "val 0/21".
    val_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        val_ds, batch_size=t.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = MobDetector(num_classes).to(DEVICE).to(memory_format=torch.channels_last)  # type: ignore[no-matching-overload]
    try:
        import triton  # noqa: F401
        compiled: nn.Module = torch.compile(model)  # type: ignore[assignment]
        print("Torch compiled with Triton backend.")
    except ImportError:
        compiled = model

    cls_fn = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
    scaler = torch.amp.GradScaler()

    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t.lr, weight_decay=t.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t.warmup_epochs,
    )
    print(f"Phase 1: backbone frozen for {t.warmup_epochs} warmup epochs.")

    best_score = 0.0
    no_improve  = 0

    for epoch in range(1, t.epochs + 1):
        if epoch == t.warmup_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            head_params = list(model.cls_head.parameters()) + list(model.bbox_head.parameters())
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": t.lr * t.backbone_lr_mul},
                    {"params": head_params, "lr": t.lr},
                ],
                weight_decay=t.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t.epochs - t.warmup_epochs,
            )
            print(f"Phase 2: backbone unfrozen (lr={t.lr * t.backbone_lr_mul:.1e}), heads lr={t.lr:.1e}")

        t0 = time.perf_counter()
        tr_loss, tr_cls, tr_bbox = train_epoch(compiled, train_loader, optimizer, scaler, cls_fn)
        val_loss, val_acc, val_iou = eval_epoch(compiled, val_loader, cls_fn)
        scheduler.step()

        print(
            f"[{epoch:02d}/{t.epochs}] {time.perf_counter() - t0:.1f}s | "
            f"train {tr_loss:.4f} (cls {tr_cls:.4f} bbox {tr_bbox:.4f}) | "
            f"val {val_loss:.4f}  acc {val_acc:.3f}  iou {val_iou:.3f}"
        )

        score = 0.7 * val_acc + 0.3 * val_iou
        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save(model.state_dict(), cfg.data.ckpt_dir / "best.pth")
            print(f"  → best model saved (acc={val_acc:.3f}  iou={val_iou:.3f})")
        else:
            no_improve += 1
            if no_improve >= t.patience:
                print(f"Early stop: no improvement for {t.patience} epochs.")
                break

    torch.save(model.state_dict(), cfg.data.ckpt_dir / "last.pth")
    print(f"Done. Best combined score: {best_score:.3f}")


if __name__ == "__main__":
    main()
