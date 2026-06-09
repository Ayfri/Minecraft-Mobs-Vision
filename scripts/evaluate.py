"""Evaluate a trained MobDetector checkpoint on the held-out test split."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
) -> None:
    model.eval()
    top1 = top5 = total = 0
    iou_sum = 0.0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    for imgs, labels, bboxes in tqdm(loader, desc="eval"):
        imgs, labels, bboxes = imgs.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE)

        with torch.amp.autocast(device_type="cuda"):
            cls_logits, bbox_pred = model(imgs)

        top_k = cls_logits.topk(min(5, len(classes)), dim=1).indices
        top1  += (top_k[:, 0] == labels).sum().item()
        top5  += (top_k == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
        iou_sum += bbox_iou(bbox_pred.float(), bboxes.float()).item()

        for label, pred in zip(labels.tolist(), top_k[:, 0].tolist()):
            per_class_total[label]   = per_class_total.get(label, 0) + 1
            per_class_correct[label] = per_class_correct.get(label, 0) + (1 if label == pred else 0)

    n = len(loader)
    print(f"Top-1 accuracy : {top1 / total:.4f}")
    print(f"Top-5 accuracy : {top5 / total:.4f}")
    print(f"Mean IoU       : {iou_sum / n:.4f}")
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


def main() -> None:
    _, _, test_ds = make_splits(DATA_DIR)
    assert isinstance(test_ds, MobDataset)

    test_loader: DataLoader[tuple[torch.Tensor, int, torch.Tensor]] = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model = MobDetector(len(test_ds.classes)).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    print(f"Checkpoint : {CKPT}")
    print(f"Test set   : {test_ds}")
    print()
    evaluate(model, test_loader, test_ds.classes)


if __name__ == "__main__":
    main()
