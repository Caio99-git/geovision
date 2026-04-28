from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_METRICS = PROJECT_ROOT / "data" / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

SENTINEL2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]

AOI_AMAZON = {
    "name": "Rondonia_Amazon",
    "description": "Deforestation monitoring polygon in Rondonia, Brazil",
    "bbox": (-63.5, -10.5, -63.0, -10.0),
    "crs": "EPSG:4326",
    "default_dates": ("2018-08-01", "2024-08-01"),
}
