"""Central config, loaded once from config.toml at import time.

config.toml is gitignored and auto-generated from defaults on first run.
Edit it without touching any Python.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

_DEFAULT_TOML = """\
[augmentation]
color_jitter_brightness = 0.2
color_jitter_contrast = 0.2
color_jitter_hue = 0.05
color_jitter_saturation = 0.2
equalize_v = true
hflip_prob = 0.5
posterize_bits = 4

[data]
ckpt_dir = "checkpoints"
data_dir = "data"
image_ext = "jpg"
seed = 42
train_ratio = 0.70
val_ratio = 0.15

[model]
backbone = "mobilenetv4_conv_small.e2400_r224_in1k"
dropout = 0.3
img_size = [224, 384]

[training]
backbone_lr_mul = 0.1
batch_size = 64
bbox_weight = 1.0
epochs = 50
label_smoothing = 0.1
lr = 1e-3
num_workers = 4
patience = 8
prefetch_factor = 2
warmup_epochs = 3
weight_decay = 1e-4
"""


@dataclass(frozen=True)
class AugmentationConfig:
    color_jitter_brightness: float
    color_jitter_contrast: float
    color_jitter_hue: float
    color_jitter_saturation: float
    equalize_v: bool
    hflip_prob: float
    posterize_bits: int

    @property
    def effective_posterize_bits(self) -> int | None:
        # 0 is the sentinel for "disabled"; T.RandomPosterize requires bits >= 1.
        return self.posterize_bits if self.posterize_bits > 0 else None


@dataclass(frozen=True)
class DataConfig:
    ckpt_dir: Path
    data_dir: Path
    image_ext: str
    seed: int
    train_ratio: float
    val_ratio: float


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    dropout: float
    img_size: tuple[int, int]


@dataclass(frozen=True)
class TrainingConfig:
    backbone_lr_mul: float
    batch_size: int
    bbox_weight: float
    epochs: int
    label_smoothing: float
    lr: float
    num_workers: int
    patience: int
    prefetch_factor: int
    warmup_epochs: int
    weight_decay: float


@dataclass(frozen=True)
class Config:
    augmentation: AugmentationConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _load_config(path: Path = _CONFIG_PATH) -> Config:
    if not path.exists():
        path.write_text(_DEFAULT_TOML, encoding="utf-8")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    a = raw["augmentation"]
    d = raw["data"]
    m = raw["model"]
    t = raw["training"]

    return Config(
        augmentation=AugmentationConfig(
            color_jitter_brightness=float(a["color_jitter_brightness"]),
            color_jitter_contrast=float(a["color_jitter_contrast"]),
            color_jitter_hue=float(a["color_jitter_hue"]),
            color_jitter_saturation=float(a["color_jitter_saturation"]),
            equalize_v=bool(a["equalize_v"]),
            hflip_prob=float(a["hflip_prob"]),
            posterize_bits=int(a["posterize_bits"]),
        ),
        data=DataConfig(
            ckpt_dir=Path(d["ckpt_dir"]),
            data_dir=Path(d["data_dir"]),
            image_ext=d["image_ext"],
            seed=int(d["seed"]),
            train_ratio=float(d["train_ratio"]),
            val_ratio=float(d["val_ratio"]),
        ),
        model=ModelConfig(
            backbone=m["backbone"],
            dropout=float(m["dropout"]),
            img_size=(int(m["img_size"][0]), int(m["img_size"][1])),
        ),
        training=TrainingConfig(
            backbone_lr_mul=float(t["backbone_lr_mul"]),
            batch_size=int(t["batch_size"]),
            bbox_weight=float(t["bbox_weight"]),
            epochs=int(t["epochs"]),
            label_smoothing=float(t["label_smoothing"]),
            lr=float(t["lr"]),
            num_workers=int(t["num_workers"]),
            patience=int(t["patience"]),
            prefetch_factor=int(t["prefetch_factor"]),
            warmup_epochs=int(t["warmup_epochs"]),
            weight_decay=float(t["weight_decay"]),
        ),
    )


cfg: Config = _load_config()
