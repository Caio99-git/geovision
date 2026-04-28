"""EuroSAT and Sentinel-2 dataset loaders.

EuroSAT MS has 27,000 64x64 13-band Sentinel-2 patches in 10 land-use classes,
organized as `<root>/<ClassName>/<ClassName>_<id>.tif`.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset

from config import EUROSAT_CLASSES


class EuroSATDataset(Dataset):
    def __init__(self, files: list[Path], transform=None):
        self.files = list(files)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(EUROSAT_CLASSES)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)  # [13, H, W]

        label = self.class_to_idx[path.parent.name]

        if self.transform is not None:
            img = np.transpose(img, (1, 2, 0))           # [H, W, 13] for albumentations
            img = self.transform(image=img)["image"]
            img = np.transpose(img, (2, 0, 1))           # back to [13, H, W]

        return img.astype(np.float32), label


def collect_eurosat_files(root: Path) -> list[Path]:
    root = Path(root)
    files: list[Path] = []
    for cls in EUROSAT_CLASSES:
        files.extend((root / cls).glob("*.tif"))
    return sorted(files)


def stratified_split(
    files: list[Path],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    by_class: dict[str, list[Path]] = {}
    for f in files:
        by_class.setdefault(f.parent.name, []).append(f)

    train, val, test = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    return train, val, test
