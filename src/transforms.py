from typing import Any

import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps

from src.config import cfg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class EqualizeV(T.Transform):
    """Equalise the Value (brightness) channel in HSV.

    Applied deterministically to both train and val images to neutralise
    Minecraft's 24-hour lighting cycle without losing colour information.
    """

    def __call__(self, img: Image.Image) -> Image.Image:  # type: ignore[override]
        h, s, v = img.convert("HSV").split()
        v = ImageOps.equalize(v)
        return Image.merge("HSV", (h, s, v)).convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def build_static_transform(
    img_size: tuple[int, int] = cfg.model.img_size,
    *,
    equalize_v: bool = cfg.augmentation.equalize_v,
) -> T.Compose:
    """PIL pipeline run once per image when building the disk cache.

    Returns a PIL Image of shape (H, W) uint8.
    """
    steps: list[Any] = []
    if equalize_v:
        steps.append(EqualizeV())
    steps.append(T.Resize(img_size))
    return T.Compose(steps)


def build_runtime_transform(
    *,
    train: bool,
    posterize_bits: int | None = cfg.augmentation.effective_posterize_bits,
) -> T.Compose:
    """Tensor pipeline applied at __getitem__ time (operates on uint8 tensors).

    Input: uint8 tensor (3, H, W) from memmap cache.
    Output: float32 tensor (3, H, W) normalised for ImageNet.
    """
    a = cfg.augmentation
    steps: list[Any] = []
    if train:
        if posterize_bits is not None:
            steps.append(T.RandomPosterize(bits=posterize_bits, p=0.5))
        steps.append(T.ColorJitter(
            brightness=a.color_jitter_brightness,
            contrast=a.color_jitter_contrast,
            saturation=a.color_jitter_saturation,
            hue=a.color_jitter_hue,
        ))
    steps.extend([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return T.Compose(steps)


def build_transforms(
    img_size: tuple[int, int] = cfg.model.img_size,
    *,
    train: bool,
    posterize_bits: int | None = cfg.augmentation.effective_posterize_bits,
    equalize_v: bool = cfg.augmentation.equalize_v,
) -> T.Compose:
    """Full PIL→tensor pipeline for single-image inference (predict.py).

    Not used during training (the dataset uses the split static+runtime path).
    """
    a = cfg.augmentation
    steps: list[Any] = []
    if equalize_v:
        steps.append(EqualizeV())
    steps.append(T.Resize(img_size))
    if train:
        if posterize_bits is not None:
            steps.append(T.RandomPosterize(bits=posterize_bits, p=0.5))
        steps.append(T.ColorJitter(
            brightness=a.color_jitter_brightness,
            contrast=a.color_jitter_contrast,
            saturation=a.color_jitter_saturation,
            hue=a.color_jitter_hue,
        ))
    steps.extend([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return T.Compose(steps)


train_transform: T.Compose = build_transforms(train=True)
val_transform: T.Compose   = build_transforms(train=False)
