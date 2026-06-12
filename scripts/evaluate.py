"""Evaluate a trained MobDetector checkpoint on the held-out test split."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import MobDataset, make_splits
from src.metrics import bbox_iou
from src.model import MobDetector

CKPT        = Path("checkpoints/best.pth")
DATA_DIR    = Path("data")
BATCH_SIZE  = 64
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: MobDetector,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]],
    classes: list[str],
    dist_blocks: np.ndarray | None = None,
) -> None:
    model.eval()
    top1 = top5 = total = 0
    iou_sum = 0.0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    # Per-sample accumulators for distance analysis
    all_correct: list[int] = []
    all_iou: list[float] = []

    for imgs, labels, bboxes in tqdm(loader, desc="eval"):
        imgs, labels, bboxes = imgs.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE)

        with torch.amp.autocast(device_type="cuda"):
            cls_logits, bbox_pred = model(imgs)

        top_k = cls_logits.topk(min(5, len(classes)), dim=1).indices
        top1  += (top_k[:, 0] == labels).sum().item()
        top5  += (top_k == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        batch_iou = bbox_iou(bbox_pred.float(), bboxes.float(), reduction="none")
        iou_sum += batch_iou.sum().item()

        for label, pred, iou in zip(labels.tolist(), top_k[:, 0].tolist(), batch_iou.tolist()):
            per_class_total[label]   = per_class_total.get(label, 0) + 1
            per_class_correct[label] = per_class_correct.get(label, 0) + (1 if label == pred else 0)
            all_correct.append(1 if label == pred else 0)
            all_iou.append(iou)

    n = len(loader)
    print(f"Top-1 accuracy : {top1 / total:.4f}")
    print(f"Top-5 accuracy : {top5 / total:.4f}")
    print(f"Mean IoU       : {iou_sum / total:.4f}")
    print()

    # sorted worst-first so the most problematic classes are visible at a glance
    rows = [
        (classes[i], per_class_correct.get(i, 0), per_class_total.get(i, 0))
        for i in range(len(classes))
        if per_class_total.get(i, 0) > 0
    ]
    rows.sort(key=lambda r: r[1] / r[2])
    print(f"{'Mob':<22} {'Acc':>6}  (correct/total)")
    print("-" * 42)
    for name, cor, tot in rows:
        print(f"  {name:<20} {cor / tot:>6.3f}  ({cor}/{tot})")

    if dist_blocks is not None and len(all_correct) == len(dist_blocks):
        _print_distance_breakdown(dist_blocks, all_correct, all_iou)


def _print_distance_breakdown(
    dist_blocks: np.ndarray,
    correct: list[int],
    ious: list[float],
) -> None:
    df = pd.DataFrame({
        "dist": dist_blocks,
        "correct": correct,
        "iou": ious,
    })
    df["bucket"] = pd.qcut(df["dist"], q=4, precision=1,
                           labels=["Q1 (close)", "Q2", "Q3", "Q4 (far)"])
    print()
    print(f"{'Distance bucket':<20} {'Range (blocks)':>18}  {'Acc':>6}  {'mIoU':>6}  {'n':>5}")
    print("-" * 62)
    for bucket, grp in df.groupby("bucket", observed=True):
        lo, hi = grp["dist"].min(), grp["dist"].max()
        print(
            f"  {str(bucket):<18} {f'{lo:.0f}–{hi:.0f}':>18}"
            f"  {grp['correct'].mean():>6.3f}"
            f"  {grp['iou'].mean():>6.3f}"
            f"  {len(grp):>5}"
        )


def main() -> None:
    _, _, test_ds = make_splits(DATA_DIR)
    assert isinstance(test_ds, MobDataset)

    test_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model = MobDetector(len(test_ds.classes), backbone=config.BACKBONE).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    print(f"Checkpoint : {CKPT}")
    print(f"Backbone   : {config.BACKBONE}  img_size={config.IMG_SIZE}")
    print(f"Test set   : {test_ds}")
    print()
    dist_blocks = test_ds._df["dist_blocks"].to_numpy() if "dist_blocks" in test_ds._df.columns else None
    evaluate(model, test_loader, test_ds.classes, dist_blocks=dist_blocks)


if __name__ == "__main__":
    main()
