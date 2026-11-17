"""Generate data/sample_historical.xlsx (SYNTHETIC) and data/sample_field.geojson.

The workbook follows the exact layout of "Timeseries data General.xlsx" so it works as
a template and as a demo dataset. Values are simulated - do not use the resulting
models for real decisions. Run:  python scripts/make_sample_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import DATA_DIR, preprocessing as pp  # noqa: E402

rng = np.random.default_rng(2024)
VARIETY_EFFECT = {"N32": 0.0, "579": 14.0, "N25": -8.0}
IRRIGATION = {"DRIP": 0.0, "CENTER PIVOT": -4.0, "FURROW": -9.0}


def simulate_cycle(field_id, irrigation, variety, ratoon, start, months):
    ages = np.arange(months + 1)
    peak_age = rng.uniform(5, 7)
    vigour = rng.normal(0, 0.06)
    base = 0.17 + (0.82 + vigour - 0.17) / (1 + np.exp(-(ages - 3.0) / 1.1))
    senesce = np.where(ages > peak_age + 3, 0.035 * (ages - peak_age - 3), 0)
    ndvi = np.clip(base - senesce + rng.normal(0, 0.015, len(ages)), 0.05, 0.88)
    cloud = np.clip(rng.exponential(7, len(ages)), 0, 80).round(2)
    cloud[rng.random(len(ages)) < 0.25] = 0.0
    dip = cloud > 30  # contaminated scenes look "dark"
    ndvi[dip] *= rng.uniform(0.35, 0.7, dip.sum())
    savi = 0.75 * ndvi ** 1.1 + rng.normal(0, 0.005, len(ages))
    evi = 0.9 * ndvi ** 1.2 + rng.normal(0, 0.01, len(ages))
    ndre = 0.78 * ndvi ** 1.05 + rng.normal(0, 0.008, len(ages))
    canopy = float(np.mean(ndvi[2:7]))  # early canopy development drives final yield
    yld = (135 + 160 * (canopy - 0.70) - 11.5 * ratoon + VARIETY_EFFECT[variety] + IRRIGATION[irrigation]
           + 1.8 * (months - 12) + rng.normal(0, 4))
    months_lbl = pd.period_range(start, periods=months + 1, freq="M").astype(str)
    return pd.DataFrame({
        "cycle_id": f"{field_id}_R{ratoon}_{months_lbl[0]}", "field": str(field_id), "irrigation": irrigation,
        "variety": variety, "ratoon": float(ratoon), "yield_tph": round(float(np.clip(yld, 25, 190)), 1),
        "month": list(months_lbl), "age": ages, "cloud": cloud,
        "ndvi": ndvi.round(4), "savi": savi.round(4), "evi": evi.round(4), "ndre": ndre.round(4),
        "cycle_months": months,
    })[pp.LONG_COLS]


def main() -> None:
    frames = []
    fields = [(9101 + i, v, irr) for i, (v, irr) in enumerate([
        ("N32", "DRIP"), ("N32", "DRIP"), ("579", "DRIP"), ("579", "DRIP"), ("N25", "DRIP"),
        ("N32", "DRIP"), ("579", "CENTER PIVOT"), ("N32", "CENTER PIVOT"), ("N25", "FURROW"), ("579", "FURROW"),
        ("N32", "DRIP"), ("579", "DRIP")])]
    for fid, variety, irr in fields:
        start = pd.Period(f"{rng.integers(2019, 2021)}-{rng.integers(5, 10):02d}", "M")
        for ratoon in range(4):
            months = int(rng.choice([10, 11, 12, 12, 13, 14]))
            frames.append(simulate_cycle(fid, irr, variety, ratoon, start, months))
            start = start + months  # next ratoon starts in the harvest month
    long = pd.concat(frames, ignore_index=True)

    path = DATA_DIR / "sample_historical.xlsx"
    pp.write_historical_workbook(long, path)

    wb = load_workbook(path)
    ws = wb.create_sheet("README")
    lines = [
        ("SYNTHETIC SAMPLE - same layout as 'Timeseries data General.xlsx'", True),
        ("Sheet 'Time series data' is the only sheet read by the app.", False),
        ("One crop cycle = one block: FIELD, IRRIGATION SYSTEM, VARIETY, RATOON, YIELD(Tons per ha) are merged cells over the block.", False),
        ("The first month of the block sits on the header row; AGE is blank there (age 0) and 1, 2, 3... below.", False),
        ("Month is text 'YYYY-MM'. Cloud Cover (%) is the scene-level cloud cover of the best image of that month.", False),
        ("NDVI, SAVI, EVI, NDRE are field means. Leave blank where no image exists. Separate cycles with one blank row.", False),
        ("To add your own data: copy a block, overwrite its values, keep column order and header names unchanged.", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name="Arial", size=10, bold=bold)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 120
    wb.save(path)

    # ~9 ha placeholder rectangle (WGS84) - replace with a real field boundary.
    lon, lat, dx, dy = 34.9000, -16.2500, 0.0030, 0.0030
    ring = [[lon, lat], [lon + dx, lat], [lon + dx, lat + dy], [lon, lat + dy], [lon, lat]]
    (DATA_DIR / "sample_field.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "Sample field (placeholder)"},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }, indent=2))
    print(f"Wrote {path} ({long['cycle_id'].nunique()} cycles) and sample_field.geojson")


if __name__ == "__main__":
    main()
