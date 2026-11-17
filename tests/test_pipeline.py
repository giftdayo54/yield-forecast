"""Offline tests (no CDSE access needed).  Run:  pytest -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import DATA_DIR, SCHEMA_COLUMNS, cdse_downloader as cd  # noqa: E402
from src import explainer as ex, model_trainer as mt, preprocessing as pp  # noqa: E402

SAMPLE = DATA_DIR / "sample_historical.xlsx"


@pytest.fixture()
def tmp_models(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mt, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(mt, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(mt, "HISTORY_PATH", tmp_path / "history.csv")
    monkeypatch.setattr(mt, "CONFIG_PATH", tmp_path / "cfg.json")
    return tmp_path


@pytest.fixture(scope="module")
def clean_sample():
    return pp.apply_qc(pp.read_historical_excel(SAMPLE).df).clean


# ------------------------------ CDSE helpers ------------------------------- #
def test_compute_indices_known_values():
    r = cd.compute_indices([0.05], [0.10], [0.50], [0.30])
    assert r["NDVI"][0] == pytest.approx(0.4 / 0.6, abs=1e-6)
    assert r["SAVI"][0] == pytest.approx(0.4 / 1.1 * 1.5, abs=1e-6)
    assert r["NDRE"][0] == pytest.approx(0.2 / 0.8, abs=1e-6)


def test_evi_blowup_is_clipped():
    # denominator ~ 0 (bright blue) would give EVI ~ 1e7 per pixel without clipping
    r = cd.compute_indices([0.2], [0.1], [0.95], [0.5])
    assert -1.0 <= r["EVI"][0] <= 2.0


def test_monthly_periods_match_notebook_logic():
    p = cd.monthly_periods("2019-09-07", "2020-09-15")
    assert p[0] == ("2019-09-01", "2019-09-30")
    assert p[-1] == ("2020-09-01", "2020-09-15")
    assert len(p) == 13


def test_evalscript_requests_mask_and_scl():
    s = cd.build_evalscript()
    assert '"SCL"' in s and '"dataMask"' in s and "FLOAT32" in s


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    with pytest.raises(cd.CredentialsError):
        cd.get_credentials()


# ------------------------------ parsing ------------------------------------ #
def test_sample_parses_and_roundtrips():
    r = pp.read_historical_excel(SAMPLE)
    assert r.n_cycles == 48 and not r.warnings
    buf = io.BytesIO()
    pp.write_historical_workbook(r.df, buf)
    r2 = pp.read_historical_excel(buf)
    cols = ["cycle_id", "month", "age", "ndvi", "evi", "yield_tph", "variety", "irrigation"]
    pd.testing.assert_frame_equal(r.df[cols].reset_index(drop=True), r2.df[cols].reset_index(drop=True))


def test_written_workbook_has_exact_schema_and_merged_headers():
    from openpyxl import load_workbook

    ws = load_workbook(SAMPLE)["Time series data"]
    assert [c.value for c in ws[1]] == SCHEMA_COLUMNS
    assert any(str(r).startswith("A2:A") for r in ws.merged_cells.ranges)
    assert ws["G2"].value is None and ws["G3"].value == 1  # AGE blank on first row


def _quirky_workbook() -> io.BytesIO:
    """Mimics the real file: stray label row, a cycle whose first AGE is 1, blank spacers."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Time series data"
    ws.append(SCHEMA_COLUMNS)
    ws.append([4311, "DRIP", 579, 0, 150.0, "2019-07", None, 0, 0.17, 0.07, 0.07, 0.11])
    for i, m in enumerate(["2019-08", "2019-09", "2019-10", "2019-11", "2019-12"], start=1):
        ws.append([None] * 5 + [m, i, 5.0, 0.2 + 0.1 * i, 0.1 * i, 0.1 * i, 0.1 * i])
    ws.append([None] * 12)
    ws.append([4312, "DRIP", "N32", 1, 120.0, "2020-09", 1, 0, 0.5, 0.3, 0.3, 0.3])  # AGE starts at 1
    ws.append([None] * 5 + ["2020-10", 2, 0, 0.6, 0.4, 0.4, 0.4])
    ws.append([None] * 12)
    ws.append([None, "FURROW"] + [None] * 10)  # stray label, no data
    buf = io.BytesIO()
    wb.save(buf)
    return buf


def test_parser_handles_real_file_quirks():
    r = pp.read_historical_excel(_quirky_workbook())
    assert r.n_cycles == 2
    ages = r.df.groupby("field")["age"].agg(["min", "max"])
    assert tuple(ages.loc["4311"]) == (0, 5)
    assert tuple(ages.loc["4312"]) == (1, 2)  # explicit AGE honoured
    assert any("no monthly rows" in w for w in r.warnings)
    assert set(r.df["variety"]) == {"579", "N32"}  # numeric 579 normalised to text


# ------------------------------ QC ----------------------------------------- #
def _tiny_cycle(**over):
    n = 10
    d = pd.DataFrame({
        "cycle_id": "A_R0_2020-01", "field": "A", "irrigation": "DRIP", "variety": "N32", "ratoon": 0.0,
        "yield_tph": 100.0, "month": pd.period_range("2020-01", periods=n, freq="M").astype(str),
        "age": range(n), "cloud": 0.0, "ndvi": np.linspace(0.2, 0.8, n), "savi": np.linspace(0.1, 0.5, n),
        "evi": np.linspace(0.1, 0.6, n), "ndre": np.linspace(0.1, 0.5, n), "cycle_months": n - 1})
    for k, v in over.items():
        d[k] = v
    return d


def test_qc_masks_cloudy_months_and_interpolates():
    d = _tiny_cycle()
    d.loc[4, "cloud"] = 60.0
    out = pp.apply_qc(d).clean
    assert out.loc[4, "ndvi_raw"] == pytest.approx(d.loc[4, "ndvi"])  # raw kept
    assert not out.loc[4, "obs_valid"]
    assert out.loc[4, "ndvi"] == pytest.approx((d.loc[3, "ndvi"] + d.loc[5, "ndvi"]) / 2)  # interpolated


def test_qc_nulls_absurd_evi_only():
    d = _tiny_cycle()
    d.loc[3, "evi"] = 36197.7
    out = pp.apply_qc(d, pp.QCConfig(interpolate=False)).clean
    assert np.isnan(out.loc[3, "evi"]) and not np.isnan(out.loc[3, "ndvi"])


def test_qc_drops_unusable_cycles_for_training_only():
    d = _tiny_cycle(yield_tph=np.nan)
    assert pp.apply_qc(d, for_training=True).clean.empty
    assert len(pp.apply_qc(d, for_training=False).clean) == len(d)


# ------------------------------ features ----------------------------------- #
def test_features_use_only_months_up_to_horizon(clean_sample):
    cats = pp.get_categories(clean_sample)
    cid = clean_sample["cycle_id"].iloc[0]
    one = clean_sample[clean_sample["cycle_id"] == cid]
    X1, _ = pp.build_features(one, 4, cats)
    changed = one.copy()
    changed.loc[changed["age"] > 4, ["ndvi", "ndre", "evi", "savi", "cloud"]] = 0.99
    X2, _ = pp.build_features(changed, 4, cats)
    pd.testing.assert_frame_equal(X1, X2)


def test_training_set_excludes_cycles_ending_at_or_before_horizon(clean_sample):
    cats = pp.get_categories(clean_sample)
    X, y, g = pp.build_training_set(clean_sample, 11, cats)
    months = clean_sample.groupby("cycle_id")["cycle_months"].first()
    assert (months.loc[X.index] > 11).all()


# ------------------------------ training ----------------------------------- #
def test_cv_is_grouped_by_field(clean_sample, monkeypatch):
    seen = []
    orig = mt.GroupKFold

    class Spy(orig):
        def split(self, X, y=None, groups=None):
            for tr, te in super().split(X, y, groups):
                seen.append((set(np.asarray(groups)[tr]), set(np.asarray(groups)[te])))
                yield tr, te

    monkeypatch.setattr(mt, "GroupKFold", Spy)
    cats = pp.get_categories(clean_sample)
    X, y, g = pp.build_training_set(clean_sample, 4, cats)
    mt.cross_validate_horizon(X, y, g, mt.TrainConfig(tune=False))
    assert seen and all(not (a & b) for a, b in seen)


def test_train_save_forecast_explain_and_retrain(clean_sample, tmp_models):
    cfg = mt.TrainConfig(horizons=[3, 6], tune=False)
    res = mt.train_all(clean_sample, cfg)
    assert list(res.metrics["horizon"]) == [3, 6] and not res.skipped
    assert {"no-imagery RMSE", "imagery gain %"} <= set(res.metrics.columns)
    assert mt.list_horizons() == [3, 6]
    assert mt.load_registry()["categories"]["variety"]

    # forecast a cycle seen only up to month 7 (as-of mid month 8)
    full = clean_sample[clean_sample["cycle_id"] == clean_sample["cycle_id"].iloc[0]]
    partial = full[full["age"] <= 7].copy()
    partial["yield_tph"] = np.nan
    start = pd.Period(partial["month"].iloc[0], "M")
    planting, harvest = start.start_time.date(), (start + 12).start_time.date()
    as_of = (start + 7).start_time.date().replace(day=15)
    fc = mt.forecast_cycle(partial, planting, harvest, as_of)
    assert fc["horizon"] == 6  # month 7 is incomplete -> last complete age 6 -> horizon 6
    lo, hi = fc["pred"]["lo"], fc["pred"]["hi"]
    assert lo < fc["pred"]["pred"] < hi

    loc = ex.local_explanation(fc["bundle"], fc["X"])
    assert loc.prediction == pytest.approx(fc["pred"]["pred"], abs=1e-2)  # base + SHAP == prediction
    glob = ex.global_explanation(fc["bundle"])
    assert len(glob.importance) == len(fc["bundle"]["feature_names"])

    with pytest.raises(ValueError):  # too early for any trained horizon
        mt.forecast_cycle(partial[partial["age"] <= 1], planting, harvest, (start + 1).start_time.date())

    mt.train_all(clean_sample, cfg)  # retrain -> previous models archived
    assert len(list((tmp_models / "archive").glob("yield_h3_*.joblib"))) == 1
    assert len(mt.load_history()) == 4


def test_small_dataset_horizons_are_skipped(clean_sample, tmp_models):
    few = clean_sample[clean_sample["cycle_id"].isin(clean_sample["cycle_id"].unique()[:6])]
    res = mt.train_all(few, mt.TrainConfig(horizons=[3], tune=False), save=False)
    assert res.metrics.empty and res.skipped


# ------------------------------ store -------------------------------------- #
def test_store_upsert_replaces_same_cycle(tmp_path):
    df = pp.read_historical_excel(SAMPLE).df
    store, info = pp.upsert_cycles(pd.DataFrame(columns=pp.LONG_COLS), df.iloc[:30])
    n0 = store["cycle_id"].nunique()
    edited = df.iloc[:30].copy()
    edited["yield_tph"] = 999.0
    merged, info = pp.upsert_cycles(store, edited)
    assert info["added"] == 0 and info["updated"] == n0
    assert (merged["yield_tph"] == 999.0).all()
    pp.save_store(merged, tmp_path / "s.csv")
    assert pp.load_store(tmp_path / "s.csv")["cycle_id"].nunique() == n0
