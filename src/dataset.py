from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src import config
from src.transforms import build_runtime_transform, build_static_transform


@lru_cache(maxsize=4)
def _load_merged(data_dir: str) -> pd.DataFrame:
    """Load frames.csv + boxes.csv, keep the largest box per frame, cached."""
    p = Path(data_dir)
    images_dir = p / "images"
    df = (
        pd.read_csv(p / "frames.csv")
        .merge(pd.read_csv(p / "boxes.csv"), on="frame")
        .reset_index(drop=True)
    )
    exists = df["frame"].apply(lambda f: (images_dir / f"{f}.png").exists())
    df = df[exists].reset_index(drop=True)

    # Keep one box per frame: the largest by area.
    # Multiple boxes per frame exist for variation; single-target training
    # requires an unambiguous ground truth (multi-box detection is future work).
    df["_area"] = df["w"] * df["h"]
    df = df.loc[df.groupby("frame")["_area"].idxmax()].drop(columns="_area")
    return df.sort_values("frame").reset_index(drop=True)


def _build_cache(
    df: pd.DataFrame,
    images_dir: Path,
    cache_path: Path,
    img_size: tuple[int, int],
    equalize_v: bool,
) -> np.memmap:
    """Decode every PNG once, apply static transforms, store as uint8 memmap."""
    H, W = img_size
    N = len(df)
    fp = np.memmap(str(cache_path), dtype="uint8", mode="w+", shape=(N, H, W, 3))
    static = build_static_transform(img_size, equalize_v=equalize_v)
    for i, frame in enumerate(tqdm(df["frame"], desc="Building cache", leave=False)):
        img = Image.open(images_dir / f"{frame}.png").convert("RGB")
        img = static(img)
        fp[i] = np.asarray(img, dtype=np.uint8)
    fp.flush()
    return fp


def _open_cache(cache_path: Path, n: int, img_size: tuple[int, int]) -> np.memmap:
    H, W = img_size
    return np.memmap(str(cache_path), dtype="uint8", mode="r", shape=(n, H, W, 3))


class MobDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    """Minecraft mob frames with classification labels and YOLO bounding boxes.

    Each item: (image (3, H, W), class_id: int, bbox (4,) as cx cy w h)

    Images are pre-processed once into a disk-backed memmap cache so that the
    training loop reads raw uint8 slices instead of decoding PNGs every epoch.
    """

    classes: list[str]
    class_to_idx: dict[str, int]

    def __init__(
        self,
        data_dir: Path | str,
        indices: list[int] | None = None,
        train: bool = True,
        img_size: tuple[int, int] = config.IMG_SIZE,
        equalize_v: bool = config.EQUALIZE_V,
    ) -> None:
        data_dir = Path(data_dir)
        full_df = _load_merged(str(data_dir))

        all_mobs: list[str] = sorted(full_df["mob"].unique().tolist())
        self.classes = all_mobs
        self.class_to_idx = {c: i for i, c in enumerate(all_mobs)}

        self._df = full_df.iloc[indices].reset_index(drop=True) if indices is not None else full_df
        self._images_dir = data_dir / "images"
        self._train = train
        self._runtime = build_runtime_transform(train=train)

        # Memmap cache: keyed by full-dataset size + resolution + equalize flag
        H, W = img_size
        eq_flag = "1" if equalize_v else "0"
        cache_dir = data_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"{len(full_df)}_{H}x{W}_eq{eq_flag}.npy"

        if not cache_path.exists():
            _build_cache(full_df, self._images_dir, cache_path, img_size, equalize_v)

        if indices is not None:
            self._cache_indices: list[int] | None = list(indices)
        else:
            self._cache_indices = None

        # Store path/shape instead of the memmap object so the dataset pickles
        # cleanly for Windows spawn workers.  The memmap is opened lazily in
        # __getitem__ the first time each worker process accesses it.
        self._cache_path = cache_path
        self._cache_n = len(full_df)
        self._cache_img_size = img_size
        self._cache: np.memmap | None = None

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        split = "train" if self._train else "val/test"
        return f"MobDataset({split}, n={len(self)}, classes={len(self.classes)})"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if self._cache is None:
            self._cache = _open_cache(self._cache_path, self._cache_n, self._cache_img_size)
        row = self._df.iloc[index]
        # Map local index → global cache row
        cache_idx = self._cache_indices[index] if self._cache_indices is not None else index
        img_uint8 = torch.from_numpy(self._cache[cache_idx].copy()).permute(2, 0, 1)
        img_t: torch.Tensor = self._runtime(img_uint8)
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
    """Split the full dataset (one row per frame) into train / val / test."""
    n = len(_load_merged(str(data_dir)))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return (
        MobDataset(data_dir, idx[:n_train], train=True),
        MobDataset(data_dir, idx[n_train : n_train + n_val], train=False),
        MobDataset(data_dir, idx[n_train + n_val :], train=False),
    )
