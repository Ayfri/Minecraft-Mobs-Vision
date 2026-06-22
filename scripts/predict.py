"""Run inference on a single image and print the predicted mob + bounding box."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from src.config import cfg
from src.model import MobDetector
from src.transforms import val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _letterbox(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Pad image to target_w/target_h aspect ratio with black bars (no stretch)."""
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    out.paste(img, ((target_w - nw) // 2, (target_h - nh) // 2))
    return out


def _load_classes(data_dir: Path) -> list[str]:
    """Build class list in class_id order (matches training), not alphabetical."""
    frames = pd.read_csv(data_dir / "frames.csv")
    frames = frames[frames["negative"] != 1]
    boxes  = pd.read_csv(data_dir / "boxes.csv")
    df = frames.merge(boxes, on="frame")
    id_to_mob: dict[int, str] = (
        df[["class_id", "mob"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["mob"]
        .to_dict()
    )
    num_classes = max(id_to_mob) + 1
    return [id_to_mob.get(i, f"class_{i}") for i in range(num_classes)]


def _draw_result(img: Image.Image, cx: float, cy: float, w: float, h: float, label: str, conf: float) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    iw, ih = img.size

    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=3)

    text = f"{label}  {conf * 100:.1f}%"
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = x1, max(0, y1 - th - 6)
    draw.rectangle([tx, ty, tx + tw + 8, ty + th + 6], fill=(255, 80, 0))
    draw.text((tx + 4, ty + 3), text, fill=(255, 255, 255), font=font)

    return out


def _gradcam(model: MobDetector, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Return a (H, W) float32 saliency map in [0, 1].

    Uses gradient × input attribution on the raw pixel tensor rather than
    hooking a backbone feature layer. GradCAM via feature-map hooks degenerates
    to spatially uniform weights for architectures with global operations
    (GRN, LayerNorm2d, …) like ConvNeXtV2: the GAP backward distributes gradients
    equally over all spatial positions, so only activation magnitudes remain -—
    which are diffuse for architectures with large receptive fields.
    Grad × input stays spatially sharp for any backbone.
    """
    inp = tensor.clone().detach().requires_grad_(True)
    logits, _ = model(inp)
    model.zero_grad()
    logits[0, class_idx].backward()

    # Sum absolute attribution across RGB channels → (H, W)
    cam = (inp.grad[0] * inp[0]).abs().sum(dim=0).detach().cpu().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam.astype(np.float32)


def _apply_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a jet-coloured GradCAM heatmap on the image."""
    t = cam
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    jet = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    heatmap = Image.fromarray(jet, mode="RGB").resize(img.size, Image.Resampling.BILINEAR)
    return Image.blend(img.convert("RGB"), heatmap, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to an image file.")
    parser.add_argument("--checkpoint", type=Path, default=cfg.data.ckpt_dir / "best.pth")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions to show.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: <input>_pred.png).")
    parser.add_argument("--heatmap", action="store_true", help="Overlay a GradCAM heatmap on the output image.")
    parser.add_argument(
        "--heatmap-alpha", type=float, default=0.5, metavar="A",
        help="Heatmap blend strength in [0, 1] (default: 0.5).",
    )
    args = parser.parse_args()

    classes = _load_classes(cfg.data.data_dir)

    model = MobDetector(len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    target_w, target_h = cfg.model.img_size[1], cfg.model.img_size[0]
    img_for_model = _letterbox(img, target_w, target_h)
    tensor: torch.Tensor = val_transform(img_for_model).unsqueeze(0).to(DEVICE)

    if args.heatmap:
        cls_logits, bbox_pred = model(tensor)
    else:
        with torch.no_grad():
            cls_logits, bbox_pred = model(tensor)

    probs = cls_logits.softmax(dim=1)[0]
    top_k = probs.topk(min(args.top, len(classes)))

    cx, cy, w, h = bbox_pred[0].tolist()
    best_idx   = top_k.indices[0].item()
    best_label = classes[best_idx]
    best_conf  = top_k.values[0].item()

    print(f"Bounding box  : cx={cx:.4f}  cy={cy:.4f}  w={w:.4f}  h={h:.4f}")
    print(f"Top-{args.top} predictions:")
    for prob, idx in zip(top_k.values.tolist(), top_k.indices.tolist(), strict=True):
        print(f"  {classes[idx]:<22} {prob * 100:>6.2f}%")

    base = img_for_model
    if args.heatmap:
        cam  = _gradcam(model, tensor, best_idx)
        base = _apply_heatmap(img_for_model, cam, alpha=args.heatmap_alpha)

    out_path = args.output or args.image.with_stem(args.image.stem + "_pred")
    _draw_result(base, cx, cy, w, h, best_label, best_conf).save(out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
