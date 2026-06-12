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

# Minecraft time buckets (ticks 0-24000)
_TIME_BUCKETS = [
    ("Dawn",  0,     2_000),
    ("Day",   2_000, 12_000),
    ("Dusk",  12_000, 14_000),
    ("Night", 14_000, 24_000),
]


@torch.no_grad()
def evaluate(
    model: MobDetector,
    loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]],
    classes: list[str],
    meta_df: pd.DataFrame | None = None,
) -> None:
    model.eval()
    top1 = top5 = total = 0
    iou_sum = 0.0

    all_label:   list[int]   = []
    all_pred:    list[int]   = []
    all_correct: list[int]   = []
    all_iou:     list[float] = []

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
            all_label.append(label)
            all_pred.append(pred)
            all_correct.append(1 if label == pred else 0)
            all_iou.append(iou)

    print(f"Top-1 accuracy : {top1 / total:.4f}")
    print(f"Top-5 accuracy : {top5 / total:.4f}")
    print(f"Mean IoU       : {iou_sum / total:.4f}")
    print()

    _print_per_class(all_label, all_pred, all_iou, classes)
    _print_confusion_matrix(all_label, all_pred, classes, top_n=15)

    if meta_df is not None and len(meta_df) == total:
        results = meta_df.copy().reset_index(drop=True)
        results["_correct"] = all_correct
        results["_iou"]     = all_iou
        results["_pred"]    = all_pred

        if "weather" in results.columns:
            _print_condition_breakdown(results, "weather", "Weather")

        if "time_ticks" in results.columns:
            _print_time_breakdown(results)

        _print_bbox_size_breakdown(results)

        if "dist_blocks" in results.columns:
            _print_distance_breakdown(results)


def _print_per_class(
    labels: list[int],
    preds: list[int],
    ious: list[float],
    classes: list[str],
) -> None:
    per_class: dict[int, dict[str, float]] = {}
    for label, pred, iou in zip(labels, preds, ious):
        e = per_class.setdefault(label, {"cor": 0, "tot": 0, "iou": 0.0})
        e["tot"] += 1
        e["iou"] += iou
        if label == pred:
            e["cor"] += 1

    rows = [
        (classes[i], e["cor"], e["tot"], e["iou"] / e["tot"])
        for i, e in per_class.items()
    ]
    rows.sort(key=lambda r: r[1] / r[2])  # worst accuracy first

    print(f"{'Mob':<22} {'Acc':>6}  {'mIoU':>6}  (correct/total)")
    print("-" * 54)
    for name, cor, tot, miou in rows:
        print(f"  {name:<20} {cor / tot:>6.3f}  {miou:>6.3f}  ({cor}/{tot})")
    print()


def _print_confusion_matrix(
    labels: list[int],
    preds: list[int],
    classes: list[str],
    top_n: int = 15,
) -> None:
    from collections import Counter
    errors: Counter[tuple[str, str]] = Counter()
    for label, pred in zip(labels, preds):
        if label != pred:
            errors[(classes[label], classes[pred])] += 1

    if not errors:
        print("No classification errors found.")
        return

    top = errors.most_common(top_n)
    print(f"Top-{top_n} most common misclassifications")
    print(f"  {'True label':<22} {'Predicted as':<22}  {'Count':>6}")
    print("  " + "-" * 54)
    for (true_cls, pred_cls), cnt in top:
        print(f"  {true_cls:<22} {pred_cls:<22}  {cnt:>6}")
    print()


def _print_condition_breakdown(df: pd.DataFrame, col: str, label: str) -> None:
    print(f"{label} breakdown")
    print(f"  {col:<20} {'Acc':>6}  {'mIoU':>6}  {'n':>6}")
    print("  " + "-" * 42)
    for val, grp in df.groupby(col, observed=True):
        print(
            f"  {str(val):<20} {grp['_correct'].mean():>6.3f}"
            f"  {grp['_iou'].mean():>6.3f}  {len(grp):>6}"
        )
    print()


def _print_time_breakdown(df: pd.DataFrame) -> None:
    def _bucket(t: float) -> str:
        for name, lo, hi in _TIME_BUCKETS:
            if lo <= t < hi:
                return name
        return "Night"  # 23000-24000 wraps into Night

    df = df.copy()
    df["_time_bucket"] = df["time_ticks"].apply(_bucket)
    order = [name for name, *_ in _TIME_BUCKETS]
    print("Time of day breakdown")
    print(f"  {'Period':<12} {'Tick range':>16}  {'Acc':>6}  {'mIoU':>6}  {'n':>6}")
    print("  " + "-" * 52)
    for name, lo, hi in _TIME_BUCKETS:
        grp = df[df["_time_bucket"] == name]
        if grp.empty:
            continue
        print(
            f"  {name:<12} {f'{lo}–{hi}':>16}"
            f"  {grp['_correct'].mean():>6.3f}"
            f"  {grp['_iou'].mean():>6.3f}"
            f"  {len(grp):>6}"
        )
    print()


def _print_bbox_size_breakdown(df: pd.DataFrame) -> None:
    if "w" not in df.columns or "h" not in df.columns:
        return
    df = df.copy()
    df["_area"] = df["w"] * df["h"]
    df["_size"] = pd.qcut(df["_area"], q=3, labels=["Small", "Medium", "Large"])
    print("Bounding-box size breakdown  (by ground-truth box area)")
    print(f"  {'Size':<10} {'Area range':>20}  {'Acc':>6}  {'mIoU':>6}  {'n':>6}")
    print("  " + "-" * 54)
    for size, grp in df.groupby("_size", observed=True):
        lo, hi = grp["_area"].min(), grp["_area"].max()
        print(
            f"  {str(size):<10} {f'{lo:.4f}–{hi:.4f}':>20}"
            f"  {grp['_correct'].mean():>6.3f}"
            f"  {grp['_iou'].mean():>6.3f}"
            f"  {len(grp):>6}"
        )
    print()


def _print_distance_breakdown(df: pd.DataFrame) -> None:
    df = df.copy()
    df["_bucket"] = pd.qcut(df["dist_blocks"], q=4, precision=1,
                            labels=["Q1 (close)", "Q2", "Q3", "Q4 (far)"])
    print("Distance breakdown")
    print(f"  {'Bucket':<18} {'Range (blocks)':>16}  {'Acc':>6}  {'mIoU':>6}  {'n':>6}")
    print("  " + "-" * 56)
    for bucket, grp in df.groupby("_bucket", observed=True):
        lo, hi = grp["dist_blocks"].min(), grp["dist_blocks"].max()
        print(
            f"  {str(bucket):<18} {f'{lo:.0f}–{hi:.0f}':>16}"
            f"  {grp['_correct'].mean():>6.3f}"
            f"  {grp['_iou'].mean():>6.3f}"
            f"  {len(grp):>6}"
        )
    print()


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

    evaluate(model, test_loader, test_ds.classes, meta_df=test_ds._df)


if __name__ == "__main__":
    main()
