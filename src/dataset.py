from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.transforms import train_transform, val_transform


class MobDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    """Loads Minecraft mob frames with classification labels and YOLO bounding boxes.

    Each item: (image_tensor (3, 224, 224), class_id: int, bbox (4,) as cx cy w h)
    """

    classes: list[str]
    class_to_idx: dict[str, int]

    def __init__(
        self,
        data_dir: Path | str,
        indices: list[int] | None = None,
        train: bool = True,
    ) -> None:
        data_dir = Path(data_dir)
        df = pd.read_csv(data_dir / "frames.csv").merge(
            pd.read_csv(data_dir / "boxes.csv"), on="frame"
        )
        # Compute class list from full dataset before subsetting
        self.classes = sorted(df["mob"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self._df = df.iloc[indices].reset_index(drop=True) if indices is not None else df.reset_index(drop=True)
        self._images_dir = data_dir / "images"
        self._transform = train_transform if train else val_transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        row = self._df.iloc[idx]
        img = Image.open(self._images_dir / f"{row['frame']}.png").convert("RGB")
        img_t: torch.Tensor = self._transform(img)
        bbox = torch.tensor([row["cx"], row["cy"], row["w"], row["h"]], dtype=torch.float32)
        return img_t, int(row["class_id"]), bbox


def make_splits(
    data_dir: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[MobDataset, MobDataset, MobDataset]:
    """Split dataset into train/val/test with a fixed seed."""
    data_dir = Path(data_dir)
    n = len(
        pd.read_csv(data_dir / "frames.csv").merge(
            pd.read_csv(data_dir / "boxes.csv"), on="frame"
        )
    )
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return (
        MobDataset(data_dir, idx[:n_train], train=True),
        MobDataset(data_dir, idx[n_train : n_train + n_val], train=False),
        MobDataset(data_dir, idx[n_train + n_val :], train=False),
    )
