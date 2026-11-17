"""Sentinel-2 L2A monthly band extraction from the Copernicus Data Space Ecosystem.

Refactored from the original notebook. Differences worth knowing about:

* Credentials are NEVER hard-coded: they come from arguments, environment
  variables (CDSE_CLIENT_ID / CDSE_CLIENT_SECRET) or Streamlit secrets.
* ``dataMask`` (and optionally the SCL scene-classification band) is requested so
  no-data pixels and, if enabled, cloud/shadow pixels are excluded from the means.
* Per-pixel EVI is clipped before averaging; a handful of pixels with a near-zero
  EVI denominator can otherwise blow up the field mean (values > 30,000 were seen
  in the historical workbook).
* Months with no usable scene are returned as NaN rows instead of being skipped,
  so the monthly grid is always complete.

``sentinelhub``, ``geopandas`` and ``rasterio`` are imported lazily so the pure
helpers (index maths, month splitting) work without them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from . import S2_BANDS

log = logging.getLogger(__name__)

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
# SCL classes treated as unusable when mask_scl=True:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud medium/high prob, 10 thin cirrus
SCL_INVALID = (0, 1, 3, 8, 9, 10)
MAX_PIXELS = 2500  # Sentinel Hub Process API limit per side

ProgressCB = Callable[[int, int, str], None]


class CredentialsError(RuntimeError):
    """Raised when no CDSE OAuth client is configured."""


# --------------------------------------------------------------------------- #
# Credentials / configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CDSECredentials:
    client_id: str
    client_secret: str


def get_credentials(client_id: str | None = None, client_secret: str | None = None) -> CDSECredentials:
    """Resolve credentials: explicit args > environment > ``st.secrets``."""
    cid = client_id or os.getenv("CDSE_CLIENT_ID")
    sec = client_secret or os.getenv("CDSE_CLIENT_SECRET")
    if not (cid and sec):
        try:  # Streamlit secrets are optional
            import streamlit as st

            cid = cid or st.secrets.get("CDSE_CLIENT_ID")
            sec = sec or st.secrets.get("CDSE_CLIENT_SECRET")
        except Exception:  # noqa: BLE001 - no secrets file / streamlit not installed
            pass
    if not (cid and sec):
        raise CredentialsError(
            "CDSE credentials not found. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET as "
            "environment variables, add them to .streamlit/secrets.toml, or enter them in the sidebar."
        )
    return CDSECredentials(str(cid), str(sec))


def credentials_available() -> bool:
    try:
        get_credentials()
        return True
    except CredentialsError:
        return False


def _sh() -> SimpleNamespace:
    """Lazy import of sentinelhub with a helpful error."""
    try:
        import sentinelhub as sh
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install 'sentinelhub' (pip install sentinelhub) to download imagery.") from exc
    return SimpleNamespace(
        SHConfig=sh.SHConfig, SentinelHubCatalog=sh.SentinelHubCatalog,
        SentinelHubRequest=sh.SentinelHubRequest, DataCollection=sh.DataCollection,
        MimeType=sh.MimeType, CRS=sh.CRS, BBox=sh.BBox,
        bbox_to_dimensions=sh.bbox_to_dimensions, Geometry=sh.Geometry,
    )


def build_config(creds: CDSECredentials):
    sh = _sh()
    config = sh.SHConfig()
    config.sh_client_id = creds.client_id
    config.sh_client_secret = creds.client_secret
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    return config


# --------------------------------------------------------------------------- #
# AOI handling
# --------------------------------------------------------------------------- #
def load_aoi(source):
    """Load a GeoJSON field boundary and return one WGS84 (Multi)Polygon.

    ``source`` may be a path, raw bytes/str, a parsed dict or a file-like object
    (e.g. a Streamlit UploadedFile). All features are dissolved into one geometry.
    """
    import geopandas as gpd

    if hasattr(source, "read"):
        source = source.read()
    if isinstance(source, (bytes, bytearray)):
        source = source.decode("utf-8")
    if isinstance(source, str):
        text = source.strip()
        if text.startswith("{"):
            gj = json.loads(text)
        else:
            with open(source, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
    elif isinstance(source, dict):
        gj = source
    else:
        # os.PathLike
        with open(source, "r", encoding="utf-8") as fh:
            gj = json.load(fh)

    kind = gj.get("type")
    if kind == "FeatureCollection":
        feats = gj.get("features", [])
    elif kind == "Feature":
        feats = [gj]
    else:  # bare geometry
        feats = [{"type": "Feature", "properties": {}, "geometry": gj}]
    feats = [f for f in feats if f.get("geometry")]
    if not feats:
        raise ValueError("GeoJSON contains no geometries.")

    crs = "EPSG:4326"
    m = re.search(r"EPSG:+(\d+)", json.dumps(gj.get("crs", {})))
    if m:
        crs = f"EPSG:{m.group(1)}"
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"AOI must be a Polygon/MultiPolygon, got {geom.geom_type}.")
    return geom


def geometry_area_ha(geom) -> float:
    """Geodesic-ish area (UTM projected) of a WGS84 geometry in hectares."""
    import geopandas as gpd

    s = gpd.GeoSeries([geom], crs=4326)
    return float(s.to_crs(s.estimate_utm_crs()).area.iloc[0] / 10_000.0)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable, no network)
# --------------------------------------------------------------------------- #
def monthly_periods(start, end) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-month windows.

    The first window starts on day 1 of the start month (as in the original
    notebook, so the first row is the planting month); the last is clipped to ``end``.
    """
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("End date is before start date.")
    out = []
    for ms in pd.date_range(start=start_ts, end=end_ts, freq="MS"):
        me = min(ms + pd.offsets.MonthEnd(1), end_ts)
        out.append((ms.strftime("%Y-%m-%d"), me.strftime("%Y-%m-%d")))
    return out


def compute_indices(blue, red, nir, re1, L: float = 0.5, evi_clip: tuple[float, float] = (-1.0, 2.0)) -> dict:
    """Per-pixel NDVI, SAVI, EVI and NDRE from reflectance arrays (0-1 scale)."""
    eps = 1e-8
    blue, red, nir, re1 = (np.asarray(a, dtype="float64") for a in (blue, red, nir, re1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + eps)
        savi = ((nir - red) / (nir + red + L + eps)) * (1 + L)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps))
        ndre = (nir - re1) / (nir + re1 + eps)
    evi = np.clip(evi, *evi_clip)
    return {"NDVI": ndvi, "SAVI": savi, "EVI": evi, "NDRE": ndre}


def build_evalscript(bands: list[str] = S2_BANDS) -> str:
    """Evalscript returning raw reflectance + SCL + dataMask as FLOAT32."""
    inputs = ",".join(f'"{b}"' for b in bands) + ',"SCL","dataMask"'
    returns = ",".join(f"sample.{b}" for b in bands) + ",sample.SCL,sample.dataMask"
    return (
        "//VERSION=3\n"
        "function setup() {\n"
        f"  return {{ input: [{inputs}], output: {{ bands: {len(bands) + 2}, sampleType: \"FLOAT32\" }} }};\n"
        "}\n"
        f"function evaluatePixel(sample) {{ return [{returns}]; }}\n"
    )


def _retry(fn, attempts: int = 3, delay: float = 3.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay * (2 ** i))


def _safe_size(bbox, resolution: float, sh) -> tuple[tuple[int, int], float]:
    """Pixel size at ``resolution`` m, coarsened if it would exceed the API limit."""
    res = float(resolution)
    size = sh.bbox_to_dimensions(bbox, resolution=res)
    while max(size) > MAX_PIXELS:
        res *= 1.25
        size = sh.bbox_to_dimensions(bbox, resolution=res)
    return size, res


# --------------------------------------------------------------------------- #
# Main extraction
# --------------------------------------------------------------------------- #
RESULT_COLUMNS = [
    "Month", "Best Acquisition Date", "Cloud Cover (%)",
    "NDVI", "SAVI", "EVI", "NDRE", "valid_pixel_frac", "status",
]


def extract_monthly_indices(
    geometry,
    start,
    end,
    creds: CDSECredentials | None = None,
    *,
    resolution: float = 10.0,
    mask_scl: bool = False,
    min_valid_frac: float = 0.2,
    progress_cb: ProgressCB | None = None,
) -> pd.DataFrame:
    """Least-cloudy Sentinel-2 L2A scene per month -> field-mean NDVI/SAVI/EVI/NDRE.

    Returns one row per calendar month between ``start`` and ``end`` with columns
    ``RESULT_COLUMNS``. ``Cloud Cover (%)`` is the scene-level ``eo:cloud_cover``
    (same definition as the historical workbook, so training and inference match).

    mask_scl : also drop SCL cloud/shadow/cirrus pixels before averaging.
    min_valid_frac : minimum share of field pixels that must be usable.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    sh = _sh()
    config = build_config(creds or get_credentials())
    collection = sh.DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url)
    catalog = sh.SentinelHubCatalog(config=config)
    evalscript = build_evalscript(S2_BANDS)
    ib = {b: i for i, b in enumerate(S2_BANDS)}
    i_scl, i_mask = len(S2_BANDS), len(S2_BANDS) + 1

    bbox = sh.BBox(geometry.bounds, crs=sh.CRS.WGS84)
    size, used_res = _safe_size(bbox, resolution, sh)
    if used_res != resolution:
        log.warning("Field too large for %s m pixels; using %.1f m.", resolution, used_res)
    transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, size[0], size[1])
    field_mask = geometry_mask([mapping(geometry)], out_shape=(size[1], size[0]), transform=transform, invert=True)
    n_field = int(field_mask.sum())
    if n_field == 0:
        raise ValueError("AOI is smaller than one pixel at this resolution.")

    periods = monthly_periods(start, end)
    rows = []
    for i, (m_start, m_end) in enumerate(periods, start=1):
        month = pd.Timestamp(m_start).strftime("%Y-%m")
        row = dict.fromkeys(RESULT_COLUMNS, np.nan)
        row.update({"Month": month, "Best Acquisition Date": None, "status": "ok"})
        if progress_cb:
            progress_cb(i, len(periods), f"Processing {month}")
        try:
            scenes = list(_retry(lambda: catalog.search(
                collection, intersects=mapping(geometry), time=(m_start, m_end),
                fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
            )))
            if not scenes:
                row["status"] = "no scene"
                rows.append(row)
                continue
            best = min(scenes, key=lambda s: s["properties"].get("eo:cloud_cover", 100))
            acq = best["properties"]["datetime"].split("T")[0]
            row["Best Acquisition Date"] = acq
            row["Cloud Cover (%)"] = round(float(best["properties"].get("eo:cloud_cover", 100)), 2)

            request = sh.SentinelHubRequest(
                evalscript=evalscript,
                input_data=[sh.SentinelHubRequest.input_data(
                    data_collection=collection, time_interval=(f"{acq}T00:00:00", f"{acq}T23:59:59"))],
                responses=[sh.SentinelHubRequest.output_response("default", sh.MimeType.TIFF)],
                bbox=bbox, geometry=sh.Geometry(mapping(geometry), sh.CRS.WGS84),
                size=size, config=config,
            )
            data = _retry(request.get_data)
            if not data:
                row["status"] = "no raster"
                rows.append(row)
                continue
            img = data[0]

            valid = field_mask & (img[:, :, i_mask] > 0)
            if mask_scl:
                valid &= ~np.isin(img[:, :, i_scl].astype(int), SCL_INVALID)
            for b in ("B02", "B04", "B05", "B08"):
                valid &= np.isfinite(img[:, :, ib[b]])
            frac = float(valid.sum() / n_field)
            row["valid_pixel_frac"] = round(frac, 3)
            if frac < min_valid_frac:
                row["status"] = f"only {frac:.0%} valid pixels"
                rows.append(row)
                continue

            idx = compute_indices(
                img[:, :, ib["B02"]][valid], img[:, :, ib["B04"]][valid],
                img[:, :, ib["B08"]][valid], img[:, :, ib["B05"]][valid],
            )
            for name, arr in idx.items():
                row[name] = round(float(np.nanmean(arr)), 4)
        except Exception as exc:  # noqa: BLE001 - keep going, record the failure
            log.warning("Month %s failed: %s", month, exc)
            row["status"] = f"error: {exc}"[:200]
        rows.append(row)

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
