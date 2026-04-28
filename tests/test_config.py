from config import (
    AOI_AMAZON,
    EUROSAT_CLASSES,
    PROJECT_ROOT,
    SENTINEL2_BANDS,
)


def test_eurosat_has_ten_classes():
    assert len(EUROSAT_CLASSES) == 10


def test_sentinel2_has_thirteen_bands():
    assert len(SENTINEL2_BANDS) == 13


def test_amazon_aoi_has_bbox():
    assert "bbox" in AOI_AMAZON
    assert len(AOI_AMAZON["bbox"]) == 4


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
