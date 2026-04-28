from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.data_loading import stratified_split
from config import EUROSAT_CLASSES


def _fake_files(per_class: int = 100, tmp_path: Path = Path("/tmp/eurosat")) -> list[Path]:
    files = []
    for cls in EUROSAT_CLASSES:
        for i in range(per_class):
            files.append(tmp_path / cls / f"{cls}_{i}.tif")
    return files


def test_stratified_split_proportions():
    files = _fake_files(per_class=100)
    train, val, test = stratified_split(files, val_frac=0.1, test_frac=0.1, seed=0)

    assert len(train) + len(val) + len(test) == len(files)
    assert len(test) == 10 * 10
    assert len(val) == 10 * 10
    assert len(train) == 10 * 80


def test_stratified_split_each_class_represented():
    files = _fake_files(per_class=50)
    train, val, test = stratified_split(files, val_frac=0.2, test_frac=0.2, seed=0)
    for split in (train, val, test):
        classes = {f.parent.name for f in split}
        assert classes == set(EUROSAT_CLASSES)


def test_stratified_split_deterministic():
    files = _fake_files(per_class=50)
    a = stratified_split(files, seed=42)
    b = stratified_split(files, seed=42)
    assert [f.name for f in a[0]] == [f.name for f in b[0]]
