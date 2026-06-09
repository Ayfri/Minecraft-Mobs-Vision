from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.transforms import train_transform, val_transform


@lru_cache(maxsize=4)
def _load_merged(data_dir: str) -> pd.DataFrame:
    """Load and merge frames.csv + boxes.csv, cached per data_dir path."""
    p = Path(data_dir)
    return (
        pd.read_csv(p / "frames.csv")
        .merge(pd.read_csv(p / "boxes.csv"), on="frame")
        .reset_index(drop=True)
    )


class MobDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    """Minecraft mob frames with classification labels and YOLO bounding boxes.

    Each item: (image (3, 224, 224), class_id: int, bbox (4,) as cx cy w h)
    """

    classes: list[str]
    class_to_idx: dict[str, int]

    def __init__(
        self,
        data_dir: Path | str,
        indices: list[int] | None = None,
        train: bool = True,
    ) -> None:
        df = _load_merged(str(data_dir))

        # Compute class list from the full dataset regardless of subset
        all_mobs: list[str] = sorted(df["mob"].unique().tolist())
        self.classes = all_mobs
        self.class_to_idx = {c: i for i, c in enumerate(all_mobs)}

        self._df = df.iloc[indices].reset_index(drop=True) if indices is not None else df
        self._images_dir = Path(data_dir) / "images"
        self._transform = train_transform if train else val_transform
        self._train = train

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        split = "train" if self._transform is train_transform else "val/test"
        return f"MobDataset({split}, n={len(self)}, classes={len(self.classes)})"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        row = self._df.iloc[index]
        img = Image.open(self._images_dir / f"{row['frame']}.png").convert("RGB")
        img_t: torch.Tensor = self._transform(img)
        bbox = torch.tensor([row["cx"], row["cy"], row["w"], row["h"]], dtype=torch.float32)
        # Random horizontal flip: mirror image and update cx = 1 - cx
        if self._train and torch.rand(1).item() < 0.5:
            img_t = img_t.flip(-1)
            bbox[0] = 1.0 - bbox[0]
        return img_t, self.class_to_idx[row["mob"]], bbox


def make_splits(
    data_dir: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple["MobDataset", "MobDataset", "MobDataset"]:
    """Split the full dataset into train / val / test with a fixed seed."""
    n = len(_load_merged(str(data_dir)))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return (
        MobDataset(data_dir, idx[:n_train], train=True),
        MobDataset(data_dir, idx[n_train : n_train + n_val], train=False),
        MobDataset(data_dir, idx[n_train + n_val :], train=False),
    )
