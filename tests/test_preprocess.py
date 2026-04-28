import pytest

np = pytest.importorskip("numpy")
A = pytest.importorskip("albumentations")

from src.preprocess import (
    EUROSAT_BAND_MEAN,
    EUROSAT_BAND_STD,
    get_train_transforms,
    get_val_transforms,
)


def test_band_stats_shapes():
    assert EUROSAT_BAND_MEAN.shape == (13,)
    assert EUROSAT_BAND_STD.shape == (13,)
    assert EUROSAT_BAND_MEAN.dtype == np.float32
    assert EUROSAT_BAND_STD.dtype == np.float32


def test_band_stats_positive():
    assert (EUROSAT_BAND_STD > 0).all()


def test_train_transforms_preserves_shape():
    img = np.random.rand(64, 64, 13).astype(np.float32) * 1000
    out = get_train_transforms()(image=img)["image"]
    assert out.shape == (64, 64, 13)


def test_val_transforms_normalizes():
    img = (EUROSAT_BAND_MEAN.reshape(1, 1, 13) * np.ones((64, 64, 13))).astype(np.float32)
    out = get_val_transforms()(image=img)["image"]
    assert np.allclose(out, 0, atol=1e-4)
