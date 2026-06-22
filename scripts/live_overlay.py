"""Real-time mob identification overlay.

Captures the screen, runs the trained MobDetector on each frame, and shows the
predicted mob + bounding box live. Two modes:

- overlay (default): a transparent, click-through, always-on-top window drawn
  directly over the game. Launch Minecraft in **borderless/windowed** (exclusive
  fullscreen captures black), run this, play normally — the label floats on top.
- mirror (--mirror): a normal window that shows the captured region with the box
  drawn on it. More robust, good for screen-sharing a demo.

Usage:
    uv run -m scripts.live_overlay                 # overlay on primary screen
    uv run -m scripts.live_overlay --mirror        # mirror window instead
    uv run -m scripts.live_overlay --fps 8 --crop-frac 0.4

Zero extra dependencies: Pillow (ImageGrab/ImageTk) + Tkinter (stdlib) + ctypes.
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path

import torch
from PIL import Image, ImageGrab, ImageTk

from scripts.predict import _letterbox, _load_classes
from src.config import cfg
from src.model import MobDetector
from src.transforms import val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Win32 extended-style flags for a transparent, click-through, no-taskbar window.
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080


def _make_click_through(root: tk.Tk) -> None:
    """Make the Tk window ignore mouse input so the game underneath stays playable."""
    if sys.platform != "win32":
        return
    import ctypes

    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())  # type: ignore[attr-defined]
    style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)  # type: ignore[attr-defined]
    style |= _WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_TOOLWINDOW
    ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)  # type: ignore[attr-defined]


def _unletterbox(
    cx: float, cy: float, w: float, h: float,
    src_w: int, src_h: int, tw: int, th: int,
) -> tuple[float, float, float, float]:
    """Map a YOLO box predicted in letterboxed (tw, th) space back to source pixels.

    Returns (x1, y1, x2, y2) in the un-padded source image coordinate system.
    """
    scale = min(tw / src_w, th / src_h)
    pad_x = (tw - src_w * scale) / 2
    pad_y = (th - src_h * scale) / 2
    x1 = ((cx - w / 2) * tw - pad_x) / scale
    y1 = ((cy - h / 2) * th - pad_y) / scale
    x2 = ((cx + w / 2) * tw - pad_x) / scale
    y2 = ((cy + h / 2) * th - pad_y) / scale
    return x1, y1, x2, y2


class LivePredictor:
    """Loads the model once and predicts (label, conf, screen-space box) per grab."""

    def __init__(self, checkpoint: Path, crop_frac: float, region: tuple[int, int, int, int]) -> None:
        self.classes = _load_classes(cfg.data.data_dir)
        self.model = MobDetector(len(self.classes)).to(DEVICE)
        self.model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        self.model.eval()
        self.crop_frac = crop_frac
        self.rx, self.ry, self.rw, self.rh = region
        self.tw, self.th = cfg.model.img_size[1], cfg.model.img_size[0]

    def grab(self) -> Image.Image:
        box = (self.rx, self.ry, self.rx + self.rw, self.ry + self.rh)
        return ImageGrab.grab(bbox=box).convert("RGB")

    @torch.no_grad()
    def predict(self, frame: Image.Image) -> tuple[str, float, tuple[float, float, float, float]]:
        # Center-crop to match the generator's 40% center crop so the mob scale lines
        # up with the training distribution (big accuracy boost vs feeding the full view).
        cw, ch = int(self.rw * self.crop_frac), int(self.rh * self.crop_frac)
        ox, oy = (self.rw - cw) // 2, (self.rh - ch) // 2
        crop = frame.crop((ox, oy, ox + cw, oy + ch))

        lb = _letterbox(crop, self.tw, self.th)
        tensor = val_transform(lb).unsqueeze(0).to(DEVICE)
        logits, bbox = self.model(tensor)
        probs = logits.softmax(1)[0]
        idx = int(probs.argmax())

        cx, cy, w, h = bbox[0].tolist()
        x1, y1, x2, y2 = _unletterbox(cx, cy, w, h, cw, ch, self.tw, self.th)
        # crop-space -> full-region screen space
        screen_box = (self.rx + ox + x1, self.ry + oy + y1, self.rx + ox + x2, self.ry + oy + y2)
        return self.classes[idx], float(probs[idx]), screen_box


def _run_overlay(pred: LivePredictor, fps: int, threshold: float) -> None:
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "black")
    root.config(bg="black")
    root.geometry(f"{pred.rw}x{pred.rh}+{pred.rx}+{pred.ry}")
    canvas = tk.Canvas(root, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    root.update_idletasks()
    _make_click_through(root)

    root.bind("<Escape>", lambda _e: root.destroy())
    delay = max(1, int(1000 / fps))

    def tick() -> None:
        label, conf, (x1, y1, x2, y2) = pred.predict(pred.grab())
        canvas.delete("all")
        # canvas is positioned at the region origin, so subtract it for local coords.
        lx1, ly1, lx2, ly2 = x1 - pred.rx, y1 - pred.ry, x2 - pred.rx, y2 - pred.ry
        color = "#ff8000" if conf >= threshold else "#888888"
        if conf >= threshold:
            canvas.create_rectangle(lx1, ly1, lx2, ly2, outline=color, width=3)
        text = f"{label}  {conf * 100:.0f}%"
        tx, ty = max(8, lx1), max(20, ly1 - 14)
        # Black halo for readability over any backdrop, then the colored text.
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            canvas.create_text(tx + dx, ty + dy, text=text, anchor="w", fill="black",
                               font=("Consolas", 20, "bold"))
        canvas.create_text(tx, ty, text=text, anchor="w", fill=color,
                           font=("Consolas", 20, "bold"))
        root.after(delay, tick)

    print("Overlay running. Press Esc (with the overlay focused) or Ctrl+C here to quit.")
    root.after(delay, tick)
    root.mainloop()


def _run_mirror(pred: LivePredictor, fps: int, threshold: float) -> None:
    root = tk.Tk()
    root.title("MobDetector — live")
    root.attributes("-topmost", True)
    scale = 0.5
    disp_w, disp_h = int(pred.rw * scale), int(pred.rh * scale)
    label_widget = tk.Label(root)
    label_widget.pack()
    info = tk.Label(root, font=("Consolas", 16, "bold"), fg="#ff8000")
    info.pack(fill="x")
    root.bind("<Escape>", lambda _e: root.destroy())
    delay = max(1, int(1000 / fps))

    from PIL import ImageDraw

    def tick() -> None:
        frame = pred.grab()
        label, conf, (x1, y1, x2, y2) = pred.predict(frame)
        draw = ImageDraw.Draw(frame)
        if conf >= threshold:
            draw.rectangle([x1 - pred.rx, y1 - pred.ry, x2 - pred.rx, y2 - pred.ry],
                           outline=(255, 128, 0), width=4)
        disp = frame.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(disp)
        label_widget.config(image=photo)
        label_widget.image = photo  # type: ignore[attr-defined]  # keep a ref
        info.config(text=f"{label}   {conf * 100:.1f}%")
        root.after(delay, tick)

    print("Mirror window running. Press Esc or Ctrl+C to quit.")
    root.after(delay, tick)
    root.mainloop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=cfg.data.ckpt_dir / "best.pth")
    p.add_argument("--fps", type=int, default=10, help="Target updates per second.")
    p.add_argument("--mirror", action="store_true", help="Show a mirror window instead of an overlay.")
    p.add_argument("--crop-frac", type=float, default=0.4,
                   help="Center fraction fed to the model (0.4 matches the dataset crop; 1.0 = full view).")
    p.add_argument("--threshold", type=float, default=0.25, help="Min confidence to draw the box.")
    p.add_argument("--region", type=str, default=None,
                   help="Capture region 'x,y,w,h' (default: full primary screen).")
    args = p.parse_args()

    if args.region:
        rx, ry, rw, rh = (int(v) for v in args.region.split(","))
    else:
        probe = ImageGrab.grab()
        rx, ry, rw, rh = 0, 0, probe.width, probe.height

    pred = LivePredictor(args.checkpoint, args.crop_frac, (rx, ry, rw, rh))
    print(f"Device {DEVICE} | region {rw}x{rh}+{rx}+{ry} | model in {pred.tw}x{pred.th}")

    if args.mirror:
        _run_mirror(pred, args.fps, args.threshold)
    else:
        _run_overlay(pred, args.fps, args.threshold)


if __name__ == "__main__":
    main()
