"""Run inference on a single image and print the predicted mob + bounding box."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.model import MobDetector
from src.transforms import val_transform

CKPT     = Path("checkpoints/best.pth")
DATA_DIR = Path("data")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """Return a (H, W) float32 GradCAM saliency map in [0, 1]."""
    gradients: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    target_layer = model.backbone[-1]

    fwd_hook = target_layer.register_forward_hook(lambda _m, _i, o: activations.append(o))
    bwd_hook = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))

    logits, _ = model(tensor)
    model.zero_grad()
    logits[0, class_idx].backward()

    fwd_hook.remove()
    bwd_hook.remove()

    grads = gradients[0]          # (1, C, H', W')
    acts  = activations[0]        # (1, C, H', W')

    weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)

    ih, iw = tensor.shape[2], tensor.shape[3]
    cam = F.interpolate(cam, size=(ih, iw), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam.astype(np.float32)


def _apply_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a jet-coloured GradCAM heatmap on the image."""
    # Jet colormap via numpy (avoid matplotlib dependency)
    t = cam                       # (H, W) in [0, 1]
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    jet = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    heatmap = Image.fromarray(jet, mode="RGB").resize(img.size, Image.Resampling.BILINEAR)
    return Image.blend(img.convert("RGB"), heatmap, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to an image file.")
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions to show.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: <input>_pred.png).")
    parser.add_argument("--heatmap", action="store_true", help="Overlay a GradCAM heatmap on the output image.")
    parser.add_argument("--heatmap-alpha", type=float, default=0.5, metavar="A", help="Heatmap blend strength in [0, 1] (default: 0.5).")
    args = parser.parse_args()

    classes: list[str] = sorted(pd.read_csv(DATA_DIR / "frames.csv")["mob"].unique().tolist())

    model = MobDetector(len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    tensor: torch.Tensor = val_transform(img).unsqueeze(0).to(DEVICE)

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
    for prob, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
        print(f"  {classes[idx]:<22} {prob * 100:>6.2f}%")

    base = img
    if args.heatmap:
        cam  = _gradcam(model, tensor, best_idx)
        base = _apply_heatmap(img, cam, alpha=args.heatmap_alpha)

    out_path = args.output or args.image.with_stem(args.image.stem + "_pred")
    _draw_result(base, cx, cy, w, h, best_label, best_conf).save(out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
