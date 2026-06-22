from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import cfg
from src.transforms import build_runtime_transform, build_static_transform


@lru_cache(maxsize=4)
def _load_merged(data_dir: str) -> pd.DataFrame:
    """Load frames.csv + boxes.csv, keep the largest box per frame, cached."""
    p = Path(data_dir)
    images_dir = p / "images"
    frames = pd.read_csv(p / "frames.csv")
    frames = frames[frames["negative"] != 1]
    df = frames.merge(pd.read_csv(p / "boxes.csv"), on="frame").reset_index(drop=True)
    exists = df["frame"].apply(lambda f: (images_dir / f"{f}.{cfg.data.image_ext}").exists())
    df = df[exists].reset_index(drop=True)

    # Single-target training requires an unambiguous ground truth; keep the largest box
    # when multiple annotations exist per frame (multi-box detection is future work).
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
        img = Image.open(images_dir / f"{frame}.{cfg.data.image_ext}").convert("RGB")
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
        img_size: tuple[int, int] = cfg.model.img_size,
        equalize_v: bool = cfg.augmentation.equalize_v,
    ) -> None:
        data_dir = Path(data_dir)
        full_df = _load_merged(str(data_dir))

        # class_id order comes from the Kotlin mod's ClassMap, not alphabetical.
        # predict.py must use this same ordering or class indices won't match the model.
        id_to_mob: dict[int, str] = (
            full_df[["class_id", "mob"]]
            .drop_duplicates()
            .sort_values("class_id")
            .set_index("class_id")["mob"]
            .to_dict()
        )
        num_classes = max(id_to_mob) + 1
        self.classes = [id_to_mob.get(i, f"class_{i}") for i in range(num_classes)]
        self.class_to_idx = {mob: cid for cid, mob in id_to_mob.items()}

        self._df = full_df.iloc[indices].reset_index(drop=True) if indices is not None else full_df
        self._images_dir = data_dir / "images"
        self._train = train
        self._runtime = build_runtime_transform(train=train)

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
        # cleanly for Windows spawn workers; the memmap is opened lazily per worker.
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
        cache_idx = self._cache_indices[index] if self._cache_indices is not None else index
        img_uint8 = torch.from_numpy(self._cache[cache_idx].copy()).permute(2, 0, 1)
        img_t: torch.Tensor = self._runtime(img_uint8)
        bbox = torch.tensor([row["cx"], row["cy"], row["w"], row["h"]], dtype=torch.float32)
        if self._train and torch.rand(1).item() < cfg.augmentation.hflip_prob:
            img_t = img_t.flip(-1)
            bbox[0] = 1.0 - bbox[0]
        return img_t, int(row["class_id"]), bbox


def _grouped_splits(
    data_dir: Path | str,
    full_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[MobDataset, MobDataset, MobDataset]:
    """Split by spawn-block instead of by frame.

    The generator orbits the camera around a single mob fixed at one world position,
    relocating it every ~10 shots. All frames sharing (class_id, mob_x, mob_y, mob_z)
    are one orbit batch with the same backdrop; a per-frame split scatters them across
    train/test, leaking the backdrop. Here each whole block lands in a single split,
    so the test set shares no spawn with train and the metrics are honest.
    """
    groups = full_df.groupby(["class_id", "mob_x", "mob_y", "mob_z"]).indices
    keys = list(groups.keys())
    order = torch.randperm(
        len(keys), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    n_train = int(len(keys) * train_ratio)
    n_val = int(len(keys) * val_ratio)

    def _collect(group_keys: list[int]) -> list[int]:
        out: list[int] = []
        for k in group_keys:
            out.extend(int(i) for i in groups[keys[k]])
        return out

    return (
        MobDataset(data_dir, _collect(order[:n_train]), train=True),
        MobDataset(data_dir, _collect(order[n_train : n_train + n_val]), train=False),
        MobDataset(data_dir, _collect(order[n_train + n_val :]), train=False),
    )


def make_splits(
    data_dir: Path | str,
    train_ratio: float = cfg.data.train_ratio,
    val_ratio: float = cfg.data.val_ratio,
    seed: int = cfg.data.seed,
    images_per_class: int | None = cfg.data.images_per_class,
    group_by_spawn: bool = cfg.data.group_by_spawn,
) -> tuple[MobDataset, MobDataset, MobDataset]:
    """Split the full dataset (one row per frame) into train / val / test."""
    full_df = _load_merged(str(data_dir))

    if group_by_spawn:
        if images_per_class is not None:
            print("[make_splits] group_by_spawn=True uses all frames; ignoring images_per_class.")
        return _grouped_splits(data_dir, full_df, train_ratio, val_ratio, seed)

    if images_per_class is not None:
        sampled = full_df.groupby("class_id", group_keys=False).apply(
            lambda g: g.sample(min(len(g), images_per_class), random_state=seed)
        )
        idx = torch.randperm(
            len(sampled), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        # Map back to positions in the full (cached) df so memmap indices stay valid.
        pos = sampled.index.tolist()
        idx = [pos[i] for i in idx]
    else:
        n = len(full_df)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return (
        MobDataset(data_dir, idx[:n_train], train=True),
        MobDataset(data_dir, idx[n_train : n_train + n_val], train=False),
        MobDataset(data_dir, idx[n_train + n_val :], train=False),
    )
