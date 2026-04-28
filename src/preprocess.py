"""Multi-spectral preprocessing for EuroSAT MS / Sentinel-2.

Per-band normalization stats are precomputed over the EuroSAT MS train split.
Recompute with `compute_band_stats` if the split changes. Albumentations is
used because torchvision.transforms assume 1- or 3-channel imagery.
"""
from __future__ import annotations

import albumentations as A
import numpy as np

EUROSAT_BAND_MEAN = np.array(
    [
        1353.7, 1117.2, 1041.9,  946.6, 1199.2, 2003.0,
        2374.0, 2301.2,  732.2,   12.1, 1820.7, 1118.2, 2599.8,
    ],
    dtype=np.float32,
)

EUROSAT_BAND_STD = np.array(
    [
         245.7,  333.0,  395.0,  593.7,  566.4,  861.2,
        1086.6, 1117.9,  404.9,    4.8, 1002.6,  761.3, 1231.6,
    ],
    dtype=np.float32,
)


def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=EUROSAT_BAND_MEAN.tolist(),
                std=EUROSAT_BAND_STD.tolist(),
                max_pixel_value=1.0,
            ),
        ]
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(
                mean=EUROSAT_BAND_MEAN.tolist(),
                std=EUROSAT_BAND_STD.tolist(),
                max_pixel_value=1.0,
            ),
        ]
    )


def compute_band_stats(loader) -> tuple[np.ndarray, np.ndarray]:
    """Streaming per-band mean/std over a DataLoader. Use to refresh constants
    above when the train split changes.
    """
    n = 0
    s = np.zeros(13, dtype=np.float64)
    s2 = np.zeros(13, dtype=np.float64)
    for batch, _ in loader:
        b = batch.numpy() if hasattr(batch, "numpy") else np.asarray(batch)
        flat = b.reshape(b.shape[0], 13, -1)
        s += flat.sum(axis=(0, 2))
        s2 += (flat ** 2).sum(axis=(0, 2))
        n += flat.shape[0] * flat.shape[2]
    mean = s / n
    var = np.maximum(s2 / n - mean ** 2, 0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
