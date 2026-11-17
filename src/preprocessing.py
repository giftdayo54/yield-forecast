"""Excel normalisation, quality control and feature engineering.

Historical layout handled (``Timeseries data General.xlsx``)
-----------------------------------------------------------
* One crop cycle = one block of monthly rows.
* FIELD / IRRIGATION SYSTEM / VARIETY / RATOON / YIELD are merged cells spanning the
  block, so pandas only sees them on the block's first row - which is also the first
  month row. Blocks are separated by a blank row.
* AGE is blank on the first row (age 0) and 1, 2, 3 ... afterwards.
* Stray label rows (e.g. a lone "FURROW") with no month data are ignored.

Everything is converted to a *tidy long* table (``LONG_COLS``) with one row per
cycle-month; all later steps work on that.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR, SCHEMA_COLUMNS, SHEET_NAME

HEADER_COLS = SCHEMA_COLUMNS[:5]
INDEX_KEYS = ["ndvi", "savi", "evi", "ndre"]
LONG_COLS = [
    "cycle_id", "field", "irrigation", "variety", "ratoon", "yield_tph",
    "month", "age", "cloud", "ndvi", "savi", "evi", "ndre", "cycle_months",
]
STORE_PATH = DATA_DIR / "training_store.csv"

_ALIASES = {
    "field": "FIELD", "field id": "FIELD", "field_id": "FIELD",
    "irrigation": "IRRIGATION SYSTEM", "irrigation system": "IRRIGATION SYSTEM",
    "variety": "VARIETY", "ratoon": "RATOON", "ratoon number": "RATOON",
    "yield": "YIELD(Tons per ha)", "yield(tons per ha)": "YIELD(Tons per ha)",
    "yield (tons per ha)": "YIELD(Tons per ha)", "yield (t/ha)": "YIELD(Tons per ha)",
    "yield(t/ha)": "YIELD(Tons per ha)",
    "month": "Month", "age": "AGE",
    "cloud cover (%)": "Cloud Cover (%)", "cloud cover": "Cloud Cover (%)", "cloud": "Cloud Cover (%)",
    "ndvi": "NDVI", "savi": "SAVI", "evi": "EVI", "ndre": "NDRE",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _canon_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = re.sub(r"\s+", " ", str(c).strip().lower())
        if key in _ALIASES:
            ren[c] = _ALIASES[key]
    return df.rename(columns=ren)


def _clean_id(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def _parse_month(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Period(v, "M")
    s = str(v).strip()
    for fmt in ("%Y-%m", "%Y/%m", "%b-%Y", "%B %Y", "%m/%Y"):
        try:
            return pd.Period(datetime.strptime(s, fmt), "M")
        except ValueError:
            continue
    try:
        return pd.Period(pd.to_datetime(s), "M")
    except Exception:  # noqa: BLE001
        return None


def months_between(a, b) -> int:
    """Whole calendar-month difference b - a (dates or timestamps)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def make_cycle_id(field_id: str, ratoon, start_month) -> str:
    r = "NA" if ratoon is None or pd.isna(ratoon) else int(ratoon)
    return f"{field_id}_R{r}_{start_month}"


# --------------------------------------------------------------------------- #
# Reading the historical workbook
# --------------------------------------------------------------------------- #
@dataclass
class ParseResult:
    df: pd.DataFrame
    warnings: list[str]

    @property
    def n_cycles(self) -> int:
        return int(self.df["cycle_id"].nunique()) if len(self.df) else 0


def read_historical_excel(source, sheet: str | None = None) -> ParseResult:
    """Parse the header-rows + monthly-rows workbook into a tidy long table."""
    if hasattr(source, "seek"):
        source.seek(0)
    xl = pd.ExcelFile(source)
    names = [sheet] if sheet else sorted(xl.sheet_names, key=lambda s: s != SHEET_NAME)
    raw = None
    for name in names:
        cand = _canon_columns(xl.parse(name))
        if {"Month", "NDVI", "SAVI", "EVI", "NDRE"} <= set(cand.columns):
            raw = cand
            break
    if raw is None:
        raise ValueError(
            "No sheet with the expected columns (Month, NDVI, SAVI, EVI, NDRE) was found. "
            f"Expected layout: {SCHEMA_COLUMNS}"
        )
    for c in SCHEMA_COLUMNS:
        if c not in raw.columns:
            raw[c] = np.nan
    raw = raw.reset_index(drop=True)

    warnings: list[str] = []
    value_cols = ["Month", "AGE", "Cloud Cover (%)", *[c.upper() for c in INDEX_KEYS]]
    cycles: list[dict] = []
    cur = None
    for i, r in raw.iterrows():
        xl_row = i + 2
        has_attr = bool(r[HEADER_COLS].notna().any())
        has_month = pd.notna(r["Month"])
        has_vals = bool(r[value_cols].notna().any())
        if not has_attr and not has_vals:  # blank spacer row closes the block
            cur = None
            continue
        if has_attr:
            cur = {"attrs": r[HEADER_COLS], "rows": [], "row": xl_row}
            cycles.append(cur)
        if has_month:
            if cur is None:
                warnings.append(f"Row {xl_row}: monthly data without a header block - skipped.")
            else:
                cur["rows"].append((xl_row, r))
        elif has_vals and not has_attr:
            warnings.append(f"Row {xl_row}: values without a Month - skipped.")

    frames = []
    for cyc in cycles:
        a = cyc["attrs"]
        if not cyc["rows"]:
            warnings.append(f"Row {cyc['row']}: header/label with no monthly rows - ignored.")
            continue
        recs = []
        for xl_row, r in cyc["rows"]:
            p = _parse_month(r["Month"])
            if p is None:
                warnings.append(f"Row {xl_row}: unreadable Month '{r['Month']}' - skipped.")
                continue
            recs.append({
                "period": p, "age_in": r["AGE"], "cloud": r["Cloud Cover (%)"],
                **{k: r[k.upper()] for k in INDEX_KEYS},
            })
        if not recs:
            continue
        d = pd.DataFrame(recs).sort_values("period").drop_duplicates("period", keep="first")
        full = pd.period_range(d["period"].iloc[0], d["period"].iloc[-1], freq="M")
        d = d.set_index("period").reindex(full)
        # AGE: blank on the first row means age 0. If the sheet labels the first month
        # with an explicit AGE (e.g. 1 because the age-0 row is missing), honour it.
        given = pd.to_numeric(d["age_in"], errors="coerce").to_numpy()
        pos = np.arange(len(full))
        first_given = np.flatnonzero(~np.isnan(given))
        offset = int(max(given[first_given[0]] - first_given[0], 0)) if len(first_given) else 0
        age = pos + offset
        bad = np.nansum((~np.isnan(given)) & (given != age))
        if bad:
            warnings.append(f"Row {cyc['row']}: AGE column disagrees with Month order in {int(bad)} rows; Month order used.")
        field_id = _clean_id(a["FIELD"])
        ratoon = pd.to_numeric(a["RATOON"], errors="coerce")
        start = str(full[0])
        out = pd.DataFrame({
            "cycle_id": make_cycle_id(field_id, ratoon, start),
            "field": field_id,
            "irrigation": _clean_id(a["IRRIGATION SYSTEM"]).upper(),
            "variety": _clean_id(a["VARIETY"]),
            "ratoon": ratoon,
            "yield_tph": pd.to_numeric(a["YIELD(Tons per ha)"], errors="coerce"),
            "month": [str(p) for p in full],
            "age": age,
            "cloud": pd.to_numeric(d["cloud"], errors="coerce").to_numpy(),
            **{k: pd.to_numeric(d[k], errors="coerce").to_numpy() for k in INDEX_KEYS},
            "cycle_months": int(age[-1]),
        })
        frames.append(out)

    if not frames:
        raise ValueError("No crop cycles could be parsed from the workbook.")
    df = pd.concat(frames, ignore_index=True)
    dup_ids = df["cycle_id"][df.duplicated(["cycle_id", "month"])].unique()
    if len(dup_ids):
        warnings.append(f"Duplicate cycle ids (same field, ratoon and start month): {list(dup_ids)[:5]}")
    return ParseResult(df[LONG_COLS], warnings)


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
@dataclass
class QCConfig:
    max_cloud_pct: float = 30.0
    bounds: dict = field(default_factory=lambda: {
        "ndvi": (-0.2, 1.0), "savi": (-0.2, 1.0), "evi": (-0.5, 1.5), "ndre": (-0.2, 1.0)})
    interpolate: bool = True
    max_interp_gap: int = 2
    yield_bounds: tuple = (5.0, 300.0)
    min_cycle_months: int = 6
    min_valid_months: int = 5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bounds"] = {k: list(v) for k, v in self.bounds.items()}
        d["yield_bounds"] = list(self.yield_bounds)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "QCConfig":
        base = cls()
        if not d:
            return base
        d = dict(d)
        if "bounds" in d:
            d["bounds"] = {k: tuple(v) for k, v in d["bounds"].items()}
        if "yield_bounds" in d:
            d["yield_bounds"] = tuple(d["yield_bounds"])
        return cls(**{**asdict(base), **d})


@dataclass
class QCResult:
    clean: pd.DataFrame
    report: pd.DataFrame
    dropped: pd.DataFrame


def apply_qc(df: pd.DataFrame, cfg: QCConfig | None = None, for_training: bool = True) -> QCResult:
    """Mask contaminated observations, interpolate short gaps, drop unusable cycles.

    * cloud > ``max_cloud_pct``  -> all indices of that month set to NaN
    * index outside ``bounds``   -> only that index set to NaN (e.g. EVI blow-ups)
    * interior gaps up to ``max_interp_gap`` months are linearly interpolated
      (raw values are kept in ``*_raw`` and the real observations flagged in ``obs_valid``)
    * ``for_training=True`` additionally removes cycles without a plausible yield,
      too short, or with too few valid months.
    """
    cfg = cfg or QCConfig()
    d = df.copy()
    n_cycles_in = d["cycle_id"].nunique()
    for k in INDEX_KEYS:
        d[f"{k}_raw"] = d[k]
    rows = []

    cloud_flag = d["cloud"] > cfg.max_cloud_pct
    for k in INDEX_KEYS:
        d.loc[cloud_flag, k] = np.nan
    rows.append(("Months masked: cloud cover > %.0f%%" % cfg.max_cloud_pct, int(cloud_flag.sum())))

    for k in INDEX_KEYS:
        lo, hi = cfg.bounds[k]
        bad = d[k].notna() & ((d[k] < lo) | (d[k] > hi) | ~np.isfinite(d[k]))
        d.loc[bad, k] = np.nan
        rows.append((f"{k.upper()} values outside [{lo}, {hi}]", int(bad.sum())))

    d["obs_valid"] = d["ndvi"].notna()
    n_interp = 0
    if cfg.interpolate:
        before = d[INDEX_KEYS].isna().sum().sum()
        for k in INDEX_KEYS:
            d[k] = d.groupby("cycle_id", sort=False)[k].transform(
                lambda s: s.interpolate(method="linear", limit=cfg.max_interp_gap, limit_area="inside"))
        n_interp = int(before - d[INDEX_KEYS].isna().sum().sum())
    rows.append((f"Values filled by interpolation (gap <= {cfg.max_interp_gap} months)", n_interp))

    dropped = []
    if for_training:
        g = d.groupby("cycle_id", sort=False)
        summary = pd.DataFrame({
            "yield": g["yield_tph"].first(),
            "months": g["age"].max(),
            "valid": g["obs_valid"].sum(),
        })
        lo, hi = cfg.yield_bounds
        reasons = {
            "yield missing": summary["yield"].isna(),
            f"yield outside [{lo:g}, {hi:g}] t/ha": summary["yield"].notna() & ((summary["yield"] < lo) | (summary["yield"] > hi)),
            f"cycle shorter than {cfg.min_cycle_months} months": summary["months"] < cfg.min_cycle_months,
            f"fewer than {cfg.min_valid_months} valid months": summary["valid"] < cfg.min_valid_months,
        }
        drop_mask = pd.Series(False, index=summary.index)
        for reason, m in reasons.items():
            m = m & ~drop_mask  # attribute each cycle to its first failing rule
            rows.append((f"Cycles dropped: {reason}", int(m.sum())))
            dropped += [{"cycle_id": cid, "reason": reason} for cid in summary.index[m]]
            drop_mask |= m
        d = d[~d["cycle_id"].isin(summary.index[drop_mask])]
    rows.append(("Cycles in -> out", f"{n_cycles_in} -> {d['cycle_id'].nunique()}"))
    return QCResult(
        d.reset_index(drop=True),
        pd.DataFrame(rows, columns=["Check", "Count"]),
        pd.DataFrame(dropped, columns=["cycle_id", "reason"]),
    )


# --------------------------------------------------------------------------- #
# Feature engineering (one feature vector per cycle per forecast horizon)
# --------------------------------------------------------------------------- #
def get_categories(df: pd.DataFrame) -> dict:
    def uniq(col):
        return sorted({str(v) for v in df[col].dropna().unique() if str(v) != ""})
    return {"variety": uniq("variety"), "irrigation": uniq("irrigation")}


def feature_columns(categories: dict) -> list[str]:
    cols = []
    for k in INDEX_KEYS:
        cols += [f"{k}_{s}" for s in ("mean", "max", "last", "auc", "slope3", "peak_age")]
    cols += ["cloud_mean", "valid_frac", "ratoon", "cycle_months"]
    cols += [f"variety_{v}" for v in categories["variety"]]
    cols += [f"irr_{v}" for v in categories["irrigation"]]
    return cols


def _series_stats(ages: np.ndarray, s: np.ndarray) -> dict:
    ok = ~np.isnan(s)
    nan = {"mean": np.nan, "max": np.nan, "last": np.nan, "auc": np.nan, "slope3": np.nan, "peak_age": np.nan}
    if not ok.any():
        return nan
    a, v = ages[ok], s[ok]
    auc = float(np.sum((v[1:] + v[:-1]) / 2.0 * np.diff(a))) if len(v) > 1 else np.nan
    slope = float(np.polyfit(a[-3:], v[-3:], 1)[0]) if len(v) >= 3 else np.nan
    return {"mean": float(v.mean()), "max": float(v.max()), "last": float(v[-1]), "auc": auc,
            "slope3": slope, "peak_age": float(a[int(np.argmax(v))])}


def build_features(df: pd.DataFrame, horizon: int, categories: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features from months 0..``horizon`` of each cycle (index = cycle_id).

    Returns ``(X, meta)``. ``meta`` carries field, yield, cycle_months and the highest
    age present in the series (used to decide training eligibility).
    """
    cols = feature_columns(categories)
    feats, metas = [], []
    for cid, g in df.groupby("cycle_id", sort=False):
        full_max_age = int(g["age"].max())
        g = g[g["age"] <= horizon].sort_values("age")
        if g.empty:
            continue
        ages = g["age"].to_numpy(float)
        f: dict = {}
        for k in INDEX_KEYS:
            for stat, val in _series_stats(ages, g[k].to_numpy(float)).items():
                f[f"{k}_{stat}"] = val
        f["cloud_mean"] = float(g["cloud"].mean()) if g["cloud"].notna().any() else np.nan
        f["valid_frac"] = float(g["obs_valid"].mean()) if "obs_valid" in g else float(g["ndvi"].notna().mean())
        first = g.iloc[0]
        f["ratoon"] = float(first["ratoon"]) if pd.notna(first["ratoon"]) else np.nan
        f["cycle_months"] = float(first["cycle_months"]) if "cycle_months" in g and pd.notna(first["cycle_months"]) else float(full_max_age)
        for v in categories["variety"]:
            f[f"variety_{v}"] = float(str(first["variety"]) == v)
        for v in categories["irrigation"]:
            f[f"irr_{v}"] = float(str(first["irrigation"]) == v)
        feats.append(pd.Series(f, name=cid))
        metas.append({"cycle_id": cid, "field": first["field"], "yield_tph": first["yield_tph"],
                      "cycle_months": f["cycle_months"], "max_age": full_max_age})
    X = pd.DataFrame(feats).reindex(columns=cols).astype(float)
    meta = pd.DataFrame(metas).set_index("cycle_id") if metas else pd.DataFrame(columns=["field", "yield_tph", "cycle_months", "max_age"])
    return X, meta


def build_training_set(clean: pd.DataFrame, horizon: int, categories: dict):
    """(X, y, groups) for cycles that have a yield, cover ages 0..horizon, and are
    still growing at ``horizon`` (so the harvest-month image is never used)."""
    X, meta = build_features(clean, horizon, categories)
    if meta.empty:
        return X, pd.Series(dtype=float), pd.Series(dtype=str)
    ok = meta["yield_tph"].notna() & (meta["max_age"] >= horizon) & (meta["cycle_months"] > horizon)
    X, meta = X[ok.to_numpy()], meta[ok.to_numpy()]
    return X, meta["yield_tph"].astype(float), meta["field"].astype(str)


# --------------------------------------------------------------------------- #
# Converting between extraction results, long table and the Excel layout
# --------------------------------------------------------------------------- #
def extraction_to_long(header: dict, ext: pd.DataFrame, cycle_months: int | None = None) -> pd.DataFrame:
    """CDSE extraction table (+ field metadata) -> tidy long table.

    ``header`` keys: field, irrigation, variety, ratoon, yield_tph (optional).
    ``cycle_months``: planned planting->harvest age in months (defaults to last row's age).
    """
    d = ext.copy().reset_index(drop=True)
    months = [str(_parse_month(m)) for m in d["Month"]]
    age = np.arange(len(d))
    ratoon = pd.to_numeric(header.get("ratoon"), errors="coerce")
    field_id = _clean_id(header.get("field"))
    yld = pd.to_numeric(header.get("yield_tph"), errors="coerce")
    return pd.DataFrame({
        "cycle_id": make_cycle_id(field_id, ratoon, months[0]),
        "field": field_id,
        "irrigation": _clean_id(header.get("irrigation")).upper(),
        "variety": _clean_id(header.get("variety")),
        "ratoon": ratoon,
        "yield_tph": yld if pd.notna(yld) and yld > 0 else np.nan,
        "month": months,
        "age": age,
        "cloud": pd.to_numeric(d["Cloud Cover (%)"], errors="coerce"),
        **{k: pd.to_numeric(d[k.upper()], errors="coerce") for k in INDEX_KEYS},
        "cycle_months": int(cycle_months) if cycle_months is not None else int(age[-1]),
    })[LONG_COLS]


def long_to_schema_frames(long: pd.DataFrame) -> list[pd.DataFrame]:
    """One DataFrame per cycle in the exact ``SCHEMA_COLUMNS`` layout (attributes on
    the first row only, AGE blank on the first row)."""
    out = []
    for _, g in long.groupby("cycle_id", sort=False):
        g = g.sort_values("age")
        f = pd.DataFrame(np.nan, index=range(len(g)), columns=SCHEMA_COLUMNS, dtype=object)
        first = g.iloc[0]
        f.loc[0, "FIELD"] = _num_or_str(first["field"])
        f.loc[0, "IRRIGATION SYSTEM"] = first["irrigation"]
        f.loc[0, "VARIETY"] = _num_or_str(first["variety"])
        f.loc[0, "RATOON"] = first["ratoon"]
        f.loc[0, "YIELD(Tons per ha)"] = first["yield_tph"]
        f["Month"] = g["month"].to_numpy()
        f["AGE"] = g["age"].to_numpy(dtype=float)
        f.loc[0, "AGE"] = np.nan
        f["Cloud Cover (%)"] = g["cloud"].round(2).to_numpy()
        for k in INDEX_KEYS:
            f[k.upper()] = g[k].round(4).to_numpy()
        out.append(f)
    return out


def _num_or_str(v):
    s = _clean_id(v)
    return int(s) if s.isdigit() else s


def write_historical_workbook(long: pd.DataFrame, target) -> None:
    """Write ``long`` in the historical layout: merged A:E header cells per cycle,
    first month on the header row, blank row between cycles. ``target`` is a path or
    a file-like object."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_NAME
    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    for j, name in enumerate(SCHEMA_COLUMNS, start=1):
        c = ws.cell(row=1, column=j, value=name)
        c.font, c.fill = bold, PatternFill("solid", fgColor="E8EFE0")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    row = 2
    for frame in long_to_schema_frames(long):
        start = row
        for _, r in frame.iterrows():
            for j, name in enumerate(SCHEMA_COLUMNS, start=1):
                v = r[name]
                if isinstance(v, float) and np.isnan(v):
                    v = None
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                c = ws.cell(row=row, column=j, value=v)
                c.font = font
                if name in ("NDVI", "SAVI", "EVI", "NDRE"):
                    c.number_format = "0.0000"
            row += 1
        if row - start > 1:
            for j in range(1, 6):
                ws.merge_cells(start_row=start, start_column=j, end_row=row - 1, end_column=j)
                ws.cell(row=start, column=j).alignment = Alignment(horizontal="center", vertical="center")
        row += 1  # blank spacer row
    for j, w in enumerate([10, 18, 10, 9, 18, 10, 7, 15, 9, 9, 9, 9], start=1):
        ws.column_dimensions[chr(64 + j)].width = w
    ws.freeze_panes = "A2"
    wb.save(target)


# --------------------------------------------------------------------------- #
# Persistent training store (long CSV) used for continuous learning
# --------------------------------------------------------------------------- #
def load_store(path: Path = STORE_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=LONG_COLS)
    df = pd.read_csv(path, dtype={"field": str, "variety": str, "irrigation": str, "cycle_id": str, "month": str})
    for c in ("variety", "irrigation"):
        df[c] = df[c].fillna("")
    return df[LONG_COLS]


def save_store(df: pd.DataFrame, path: Path = STORE_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[LONG_COLS].to_csv(path, index=False)


def upsert_cycles(store: pd.DataFrame, new: pd.DataFrame, replace_all: bool = False) -> tuple[pd.DataFrame, dict]:
    """Merge ``new`` into ``store``; a cycle with the same id is replaced by the new version."""
    if replace_all or store.empty:
        return new[LONG_COLS].copy(), {"added": int(new["cycle_id"].nunique()), "updated": 0}
    ids_new = set(new["cycle_id"])
    ids_old = set(store["cycle_id"])
    merged = pd.concat([store[~store["cycle_id"].isin(ids_new)], new[LONG_COLS]], ignore_index=True)
    return merged, {"added": len(ids_new - ids_old), "updated": len(ids_new & ids_old)}


def store_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"cycles": 0, "fields": 0, "with_yield": 0}
    g = df.groupby("cycle_id")["yield_tph"].first()
    return {"cycles": int(g.size), "fields": int(df["field"].nunique()), "with_yield": int(g.notna().sum())}
