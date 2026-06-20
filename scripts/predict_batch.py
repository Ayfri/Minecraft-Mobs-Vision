"""Run inference on all images in a folder and write annotated results to another folder."""

import argparse
from pathlib import Path

import torch

from scripts.predict import _apply_heatmap, _draw_result, _gradcam, _letterbox, _load_classes
from src.config import cfg
from src.model import MobDetector
from src.transforms import val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/test_input"))
    parser.add_argument("--output", type=Path, default=Path("data/test_output"))
    parser.add_argument("--checkpoint", type=Path, default=cfg.data.ckpt_dir / "best.pth")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--heatmap-alpha", type=float, default=0.5, metavar="A")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {args.input}")
        return

    classes = _load_classes(cfg.data.data_dir)
    model = MobDetector(len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    target_w, target_h = cfg.model.img_size[1], cfg.model.img_size[0]

    from PIL import Image
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_lb = _letterbox(img, target_w, target_h)
        tensor: torch.Tensor = val_transform(img_lb).unsqueeze(0).to(DEVICE)

        if args.heatmap:
            cls_logits, bbox_pred = model(tensor)
        else:
            with torch.no_grad():
                cls_logits, bbox_pred = model(tensor)

        probs = cls_logits.softmax(dim=1)[0]
        top_k = probs.topk(min(args.top, len(classes)))

        cx, cy, w, h = bbox_pred[0].tolist()
        best_idx = top_k.indices[0].item()
        best_label = classes[best_idx]
        best_conf = top_k.values[0].item()

        print(f"\n{img_path.name}")
        print(f"  bbox  cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
        for prob, idx in zip(top_k.values.tolist(), top_k.indices.tolist(), strict=True):
            print(f"  {classes[idx]:<22} {prob * 100:>6.2f}%")

        base = img_lb
        if args.heatmap:
            cam = _gradcam(model, tensor, best_idx)
            base = _apply_heatmap(img_lb, cam, alpha=args.heatmap_alpha)

        out_path = args.output / f"{img_path.stem}_pred.png"
        _draw_result(base, cx, cy, w, h, best_label, best_conf).save(out_path)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
