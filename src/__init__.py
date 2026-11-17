"""Sugarcane multi-temporal yield forecasting (Sentinel-2 + XGBoost)."""
from pathlib import Path

__version__ = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(exist_ok=True)

# Exact column layout of "Timeseries data General.xlsx" (sheet "Time series data").
SHEET_NAME = "Time series data"
SCHEMA_COLUMNS = [
    "FIELD", "IRRIGATION SYSTEM", "VARIETY", "RATOON", "YIELD(Tons per ha)",
    "Month", "AGE", "Cloud Cover (%)", "NDVI", "SAVI", "EVI", "NDRE",
]
INDEX_COLUMNS = ["NDVI", "SAVI", "EVI", "NDRE"]
S2_BANDS = ["B02", "B03", "B04", "B05", "B08"]  # Blue, Green, Red, Red-edge 1, NIR
