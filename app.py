"""Sugarcane yield forecasting - Streamlit app (Sentinel-2 / CDSE + XGBoost + SHAP).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from src import DATA_DIR, INDEX_COLUMNS, OUTPUTS_DIR, __version__
from src import cdse_downloader as cd
from src import explainer as ex
from src import model_trainer as mt
from src import preprocessing as pp

st.set_page_config(page_title="Sugarcane Yield Forecaster", page_icon="🌾", layout="wide")

FORECAST_DIR = OUTPUTS_DIR / "forecasts"
LOG_PATH = OUTPUTS_DIR / "forecast_log.csv"
SAMPLE_XLSX = DATA_DIR / "sample_historical.xlsx"
SAMPLE_GEOJSON = DATA_DIR / "sample_field.geojson"
IRRIGATION_OPTIONS = ["DRIP", "CENTER PIVOT", "DRAG", "FURROW"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def get_creds() -> cd.CDSECredentials:
    return cd.get_credentials(st.session_state.get("cdse_id") or None, st.session_state.get("cdse_secret") or None)


def get_qc() -> pp.QCConfig:
    s = st.session_state
    cfg = pp.QCConfig(max_cloud_pct=float(s.get("qc_cloud", 30)), max_interp_gap=int(s.get("qc_gap", 2)),
                      interpolate=int(s.get("qc_gap", 2)) > 0, min_valid_months=int(s.get("qc_valid", 5)))
    cfg.bounds["evi"] = (-0.5, float(s.get("qc_evi_max", 1.5)))
    return cfg


def show_map(geom) -> None:
    import pydeck as pdk

    minx, miny, maxx, maxy = geom.bounds
    extent = max(maxx - minx, maxy - miny, 1e-4)
    layer = pdk.Layer("GeoJsonLayer", data=geom.__geo_interface__, stroked=True, filled=True,
                      get_fill_color=[46, 125, 50, 70], get_line_color=[46, 125, 50, 255], line_width_min_pixels=2)
    view = pdk.ViewState(latitude=geom.centroid.y, longitude=geom.centroid.x,
                         zoom=float(np.clip(np.log2(281 / extent), 3, 17)))
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view), height=320)


def aoi_input(key: str):
    """GeoJSON uploader (or bundled sample). Returns a shapely geometry or None."""
    up = st.file_uploader("Field boundary (GeoJSON)", type=["geojson", "json"], key=f"{key}_aoi")
    use_sample = st.checkbox("Use the bundled sample polygon", value=False, key=f"{key}_sample", disabled=up is not None)
    try:
        if up is not None:
            geom = cd.load_aoi(up.getvalue())
        elif use_sample:
            geom = cd.load_aoi(str(SAMPLE_GEOJSON))
        else:
            return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the GeoJSON: {exc}")
        return None
    st.caption(f"Geometry: {geom.geom_type} - {cd.geometry_area_ha(geom):,.1f} ha")
    show_map(geom)
    return geom


def field_inputs(key: str, with_yield: bool = False) -> dict:
    reg = mt.load_registry() or {}
    cats = reg.get("categories", {"variety": [], "irrigation": []})
    c1, c2, c3 = st.columns(3)
    field_id = c1.text_input("Field ID", value="", key=f"{key}_field")
    variety = c2.text_input("Variety", value="", key=f"{key}_variety",
                            help="Known varieties: " + (", ".join(cats["variety"]) or "none yet - train a model first"))
    irr_opts = sorted(set(IRRIGATION_OPTIONS) | set(cats["irrigation"]))
    irrigation = c3.selectbox("Irrigation system", irr_opts, key=f"{key}_irr")
    c4, c5, c6 = st.columns(3)
    ratoon = c4.number_input("Ratoon number (0 = plant crop)", 0, 15, 0, key=f"{key}_ratoon")
    planting = c5.date_input("Planting / ratoon-start date", value=date.today().replace(day=1), key=f"{key}_plant")
    harvest = c6.date_input("Expected / actual harvest date", value=(pd.Timestamp(date.today().replace(day=1)) + pd.Timedelta(days=365)).date(),
                            key=f"{key}_harvest")
    out = {"field": field_id.strip(), "variety": variety.strip(), "irrigation": irrigation, "ratoon": int(ratoon),
           "planting": planting, "harvest": harvest}
    if with_yield:
        y = st.number_input("Actual yield (t/ha) - optional, 0 = unknown", 0.0, 400.0, 0.0, 0.1, key=f"{key}_yield")
        out["yield_tph"] = y if y > 0 else np.nan
    return out


def validate_header(h: dict) -> str | None:
    if not h["field"]:
        return "Enter a Field ID."
    if not h["variety"]:
        return "Enter the variety."
    if h["harvest"] <= h["planting"]:
        return "Harvest date must be after the planting date."
    if pp.months_between(h["planting"], h["harvest"]) < 3:
        return "The cycle must be at least 3 months long."
    return None


def to_xlsx_bytes(long: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pp.write_historical_workbook(long, buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    st.sidebar.title("🌾 Yield Forecaster")
    st.sidebar.caption(f"v{__version__}")
    reg = mt.load_registry()
    st.sidebar.subheader("Model status")
    if reg:
        st.sidebar.success(f"Trained {reg['trained_at']}")
        st.sidebar.dataframe(pd.DataFrame(reg["horizons"]).T[["n_cycles", "rmse", "r2"]].round(2))
    else:
        st.sidebar.warning("No trained models yet - see tab 3.")
    with st.sidebar.expander("Copernicus (CDSE) credentials"):
        st.text_input("OAuth client ID", type="password", key="cdse_id")
        st.text_input("OAuth client secret", type="password", key="cdse_secret")
        st.caption("Environment/secrets credentials detected." if cd.credentials_available()
                   else "Nothing configured - enter above, or set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET.")
    with st.sidebar.expander("Quality-control settings", expanded=False):
        st.slider("Mask months with scene cloud cover above (%)", 0, 100, 30, key="qc_cloud")
        st.slider("Interpolate gaps up to (months, 0 = off)", 0, 4, 2, key="qc_gap")
        st.slider("Max plausible EVI", 0.8, 3.0, 1.5, 0.1, key="qc_evi_max")
        st.slider("Min. valid months per cycle (training)", 2, 10, 5, key="qc_valid")
        st.caption("Applied identically to training data and to new forecasts.")


# --------------------------------------------------------------------------- #
# Tab 1 - Sentinel-2 extraction
# --------------------------------------------------------------------------- #
def tab_extract() -> None:
    st.subheader("Download monthly Sentinel-2 indices from CDSE")
    st.caption("Least-cloudy Sentinel-2 L2A scene per month (B02, B03, B04, B05, B08) -> field-mean NDVI, SAVI, EVI, NDRE, "
               "exported in the layout of the historical workbook.")
    geom = aoi_input("ex")
    h = field_inputs("ex", with_yield=True)
    c1, c2 = st.columns(2)
    mask_scl = c1.checkbox("Also mask cloud/shadow pixels with the SCL band", value=False,
                           help="Off by default so values match the historical workbook. Turning it on gives cleaner "
                                "values but a slightly different definition than your training data.")
    res = c2.number_input("Pixel size (m)", 10.0, 60.0, 10.0, 10.0)
    if st.button("Download & compute indices", type="primary", key="ex_go"):
        err = validate_header(h)
        if err or geom is None:
            st.error(err or "Upload a GeoJSON (or tick the sample polygon).")
        else:
            bar = st.progress(0.0)
            try:
                ext = cd.extract_monthly_indices(geom, h["planting"], h["harvest"], get_creds(), resolution=res, mask_scl=mask_scl,
                                                 progress_cb=lambda i, n, m: bar.progress(i / n, text=m))
                st.session_state["extraction"] = {"header": h, "ext": ext, "area_ha": cd.geometry_area_ha(geom)}
                bar.empty()
            except cd.CredentialsError as exc:
                bar.empty()
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                bar.empty()
                st.error(f"Extraction failed: {exc}")
    data = st.session_state.get("extraction")
    if not data:
        return
    ext, hh = data["ext"], data["header"]
    ok = int((ext["status"] == "ok").sum())
    st.success(f"{ok} of {len(ext)} months have usable imagery.")
    st.dataframe(ext, hide_index=True)
    st.line_chart(ext.set_index("Month")[INDEX_COLUMNS])
    long = pp.extraction_to_long(hh, ext, cycle_months=pp.months_between(hh["planting"], hh["harvest"]))
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Excel (historical layout)", to_xlsx_bytes(long), f"{hh['field']}_R{hh['ratoon']}_timeseries.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("⬇ CSV", ext.to_csv(index=False).encode(), f"{hh['field']}_R{hh['ratoon']}_timeseries.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab 2 - historical data & QC
# --------------------------------------------------------------------------- #
def tab_data() -> None:
    st.subheader("Historical data management & quality control")
    src = st.radio("Data source", ["Upload workbook", "Bundled synthetic sample", "Current training store"], horizontal=True)
    df, warns = None, []
    try:
        if src == "Upload workbook":
            up = st.file_uploader("Historical workbook (.xlsx, same layout as 'Timeseries data General.xlsx')", type=["xlsx"])
            if up:
                r = pp.read_historical_excel(up)
                df, warns = r.df, r.warnings
        elif src == "Bundled synthetic sample":
            r = pp.read_historical_excel(SAMPLE_XLSX)
            df, warns = r.df, r.warnings
            st.info("Synthetic demo data - good for trying the app, not for real forecasts.")
        else:
            df = pp.load_store()
            if df.empty:
                st.info("The training store is empty. Load a workbook and save it to the store first.")
                df = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the workbook: {exc}")
    if df is None:
        return

    s = pp.store_summary(df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crop cycles", s["cycles"])
    m2.metric("Fields", s["fields"])
    m3.metric("Cycles with yield", s["with_yield"])
    m4.metric("Monthly rows", len(df))
    if warns:
        with st.expander(f"{len(warns)} parsing note(s)"):
            st.write("\n".join(f"- {w}" for w in warns))

    qc = pp.apply_qc(df, get_qc(), for_training=True)
    st.markdown("**QC report** (settings in the sidebar)")
    c1, c2 = st.columns([3, 2])
    c1.dataframe(qc.report.astype({"Count": str}), hide_index=True)
    if len(qc.dropped):
        c2.markdown("Dropped cycles")
        c2.dataframe(qc.dropped, hide_index=True)
    cid = st.selectbox("Inspect a cycle", qc.clean["cycle_id"].unique())
    g = qc.clean[qc.clean["cycle_id"] == cid].set_index("age")
    cc1, cc2 = st.columns(2)
    cc1.caption("NDVI: raw vs after QC (masked + interpolated)")
    cc1.line_chart(g[["ndvi_raw", "ndvi"]])
    cc2.caption("Scene cloud cover (%)")
    cc2.bar_chart(g["cloud"])

    if src != "Current training store":
        b1, b2 = st.columns(2)
        if b1.button("➕ Merge into training store"):
            store, info = pp.upsert_cycles(pp.load_store(), df)
            pp.save_store(store)
            st.success(f"Store updated: {info['added']} added, {info['updated']} replaced ({store['cycle_id'].nunique()} cycles total).")
        if b2.button("♻ Replace training store"):
            pp.save_store(df)
            st.success(f"Training store replaced with {df['cycle_id'].nunique()} cycles.")
    st.download_button("⬇ Cleaned long table (CSV)", qc.clean.to_csv(index=False).encode(), "cleaned_long.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab 3 - training
# --------------------------------------------------------------------------- #
def run_training(cfg: mt.TrainConfig) -> mt.TrainResult | None:
    store = pp.load_store()
    if store.empty:
        st.warning("The training store is empty - add data in tab 2 first.")
        return None
    qc = pp.apply_qc(store, get_qc(), for_training=True)
    if qc.clean.empty:
        st.error("No cycles left after QC.")
        return None
    bar = st.progress(0.0)
    res = mt.train_all(qc.clean, cfg, progress_cb=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    if not res.metrics.empty:
        res.metrics.to_csv(OUTPUTS_DIR / "cv_metrics.csv", index=False)
        res.oof.to_csv(OUTPUTS_DIR / "cv_oof_predictions.csv", index=False)
    return res


def show_training_result(res: mt.TrainResult) -> None:
    if res.metrics.empty:
        st.error("No horizon could be trained. " + "; ".join(f"h={h}: {r}" for h, r in res.skipped))
        return
    st.success(f"Trained and saved {len(res.metrics)} model(s) at {res.trained_at}.")
    for h, reason in res.skipped:
        st.warning(f"Horizon {h} skipped: {reason}")
    show = res.metrics.copy()
    show[["RMSE", "MAE", "R2", "MAPE %", "bias", "mean-baseline RMSE", "no-imagery RMSE", "imagery gain %"]] = \
        show[["RMSE", "MAE", "R2", "MAPE %", "bias", "mean-baseline RMSE", "no-imagery RMSE", "imagery gain %"]].round(2)
    st.dataframe(show, hide_index=True)
    st.caption("Field-grouped CV (no field appears in both train and test). RMSE/MAE in t/ha. "
               "'No-imagery RMSE' = same model using only ratoon, variety, irrigation and cycle length - "
               "'imagery gain %' is what Sentinel-2 adds on top.")
    weak = res.metrics[res.metrics["imagery gain %"] <= 0]["horizon"].tolist()
    if weak:
        st.warning(f"At horizon(s) {weak} the model does not beat the no-imagery baseline. With few cycles, extra monthly "
                   "features can overfit - add more cycles/fields before relying on imagery at those horizons.")
    h = st.selectbox("Observed vs predicted", res.metrics["horizon"].tolist())
    st.pyplot(mt.plot_observed_vs_predicted(res.oof, h))
    st.download_button("⬇ CV metrics (CSV)", res.metrics.to_csv(index=False).encode(), "cv_metrics.csv", "text/csv")


def tab_train() -> None:
    st.subheader("Train multi-horizon XGBoost models")
    store = pp.load_store()
    s = pp.store_summary(store)
    st.write(f"Training store: **{s['cycles']} cycles**, {s['fields']} fields, {s['with_yield']} with yield.")
    last = mt.load_last_config()
    c1, c2, c3, c4 = st.columns(4)
    horizons = c1.multiselect("Forecast horizons (months after planting)", list(range(2, 16)), default=last.horizons)
    n_splits = c2.slider("Outer CV folds (grouped by field)", 2, 10, last.n_splits)
    tune = c3.checkbox("Tune hyper-parameters (nested CV)", value=last.tune)
    n_iter = c4.slider("Random-search iterations", 5, 60, last.n_iter, disabled=not tune)
    if tune:
        st.caption("Nested tuning refits many small models per horizon; expect roughly a minute per few horizons on a laptop.")
    if st.button("🚀 Train models", type="primary"):
        if not horizons:
            st.error("Select at least one horizon.")
        else:
            res = run_training(mt.TrainConfig(horizons=horizons, n_splits=n_splits, tune=tune, n_iter=n_iter))
            if res is not None:
                st.session_state["train_result"] = res
    if "train_result" in st.session_state:
        show_training_result(st.session_state["train_result"])


# --------------------------------------------------------------------------- #
# Tab 4 - forecast
# --------------------------------------------------------------------------- #
def log_forecast(entry: dict, long: pd.DataFrame) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(FORECAST_DIR / f"{entry['forecast_id']}.csv", index=False)
    old = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([entry])], ignore_index=True).to_csv(LOG_PATH, index=False)


def run_forecast(h: dict, as_of: date, source: str, geom, upload, horizon_choice) -> dict:
    planned = pp.months_between(h["planting"], h["harvest"])
    if source.startswith("Download"):
        if geom is None:
            raise ValueError("Upload a GeoJSON boundary (or tick the sample polygon).")
        ext = cd.extract_monthly_indices(geom, h["planting"], as_of, get_creds())
        long = pp.extraction_to_long(h, ext, cycle_months=planned)
    else:
        if upload is None:
            raise ValueError("Upload the workbook produced in step 1.")
        parsed = pp.read_historical_excel(upload)
        long = parsed.df[parsed.df["cycle_id"] == parsed.df["cycle_id"].iloc[0]].copy()
        long["field"], long["variety"], long["irrigation"] = h["field"], h["variety"], h["irrigation"].upper()
        long["ratoon"], long["yield_tph"], long["cycle_months"] = float(h["ratoon"]), np.nan, planned
        long["cycle_id"] = pp.make_cycle_id(h["field"], h["ratoon"], long["month"].iloc[0])

    fc = mt.forecast_cycle(long, h["planting"], h["harvest"], as_of, get_qc(), horizon_choice)
    bundle, pred, hz = fc["bundle"], fc["pred"], fc["horizon"]
    local = ex.local_explanation(bundle, fc["X"])
    cv = bundle["cv"]
    entry = {
        "forecast_id": f"{fc['cycle_id']}__{datetime.now():%Y%m%d%H%M%S}", "created_at": f"{datetime.now():%Y-%m-%d %H:%M}",
        "field": h["field"], "variety": h["variety"], "irrigation": h["irrigation"], "ratoon": h["ratoon"],
        "planting": h["planting"], "harvest": h["harvest"], "as_of": as_of, "horizon": hz,
        "pred_tph": round(float(pred["pred"]), 1), "lo_tph": round(float(pred["lo"]), 1), "hi_tph": round(float(pred["hi"]), 1),
        "model_trained_at": bundle["trained_at"], "actual_yield": np.nan, "added_to_store": False,
    }
    log_forecast(entry, long)
    return {"entry": entry, "pred": pred, "local": local, "clean": fc["clean"], "notes": fc["notes"],
            "cv_rmse": cv["model"]["rmse"], "cv_noimg": cv["no_imagery"]["rmse"], "horizon": hz,
            "area_ha": cd.geometry_area_ha(geom) if geom is not None else None}


def tab_forecast() -> None:
    st.subheader("In-season yield forecast")
    h = field_inputs("fc")
    c1, c2 = st.columns(2)
    as_of = c1.date_input("Forecast as-of date", value=min(date.today(), h["harvest"]), key="fc_asof",
                          help="Imagery from planting up to this date is used.")
    available = mt.list_horizons()
    hz_choice = c2.selectbox("Model horizon", ["Auto"] + available, help="Auto = longest trained horizon covered by complete months.")
    source = st.radio("Time-series source", ["Download from CDSE", "Upload workbook from step 1"], horizontal=True)
    upload = None
    if source.startswith("Upload"):
        upload = st.file_uploader("Workbook from step 1", type=["xlsx"], key="fc_xlsx")
        st.caption("A GeoJSON is optional here - it is only used for the production estimate (t/ha x area).")
    geom = aoi_input("fc")

    if st.button("🔮 Run forecast", type="primary"):
        err = validate_header(h)
        if err:
            st.error(err)
        elif as_of <= h["planting"]:
            st.error("The as-of date must be after the planting date.")
        else:
            with st.spinner("Extracting imagery, building features, predicting..."):
                try:
                    st.session_state["forecast"] = run_forecast(h, as_of, source, geom, upload, hz_choice)
                except cd.CredentialsError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
    f = st.session_state.get("forecast")
    if not f:
        return
    e, p = f["entry"], f["pred"]
    st.markdown(f"### Field {e['field']} · ratoon {e['ratoon']} · horizon {f['horizon']} months")
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast yield", f"{p['pred']:.1f} t/ha")
    m2.metric("80% interval (from CV residuals)", f"{p['lo']:.0f} - {p['hi']:.0f} t/ha")
    if f["area_ha"]:
        m3.metric("Expected production", f"{p['pred'] * f['area_ha']:,.0f} t", f"{f['area_ha']:.1f} ha", delta_color="off")
    st.caption(f"Typical error of this model in grouped CV: RMSE {f['cv_rmse']:.1f} t/ha "
               f"(no-imagery baseline {f['cv_noimg']:.1f}). Model trained {e['model_trained_at']}.")
    for n in f["notes"]:
        st.warning(n)
    t1, t2 = st.columns(2)
    t1.markdown("**Why this forecast** (SHAP)")
    t1.pyplot(f["local"].waterfall_fig)
    t1.info(f["local"].summary)
    t2.markdown("**Satellite time series used**")
    g = f["clean"].set_index("month")
    t2.line_chart(g[["ndvi", "ndre", "evi", "savi"]])
    t2.dataframe(f["local"].contributions.head(12).round(3), hide_index=True)
    st.download_button("⬇ Forecast report (CSV)", pd.DataFrame([e]).to_csv(index=False).encode(), f"forecast_{e['field']}.csv", "text/csv")
    st.caption("This forecast was saved to the forecast log - enter the actual yield in tab 5 after harvest to feed it back into training.")


# --------------------------------------------------------------------------- #
# Tab 5 - continuous learning
# --------------------------------------------------------------------------- #
def tab_learning() -> None:
    st.subheader("Continuous learning")
    st.caption("Add newly harvested cycles -> retrain -> compare with the previous model. Previous models are archived in models/archive/.")
    store = pp.load_store()
    s = pp.store_summary(store)
    st.write(f"Training store: **{s['cycles']} cycles**, {s['fields']} fields.")

    st.markdown("#### A. Upload new harvest data")
    up = st.file_uploader("Workbook with newly harvested cycles (same layout, yield filled in)", type=["xlsx"], key="learn_xlsx")
    if up is not None:
        try:
            r = pp.read_historical_excel(up)
            n_y = r.df.groupby("cycle_id")["yield_tph"].first().notna().sum()
            st.write(f"Parsed {r.n_cycles} cycle(s), {n_y} with yield.")
            if r.warnings:
                st.caption(" | ".join(r.warnings[:3]))
            if st.button("Merge into training store", key="learn_merge"):
                merged, info = pp.upsert_cycles(store, r.df)
                pp.save_store(merged)
                st.success(f"{info['added']} added, {info['updated']} replaced.")
                st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read workbook: {exc}")

    st.markdown("#### B. Close the loop on logged forecasts")
    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    pending = log[(log["actual_yield"].isna()) & (~log["added_to_store"].astype(bool))] if len(log) else log
    if pending.empty:
        st.info("No forecasts waiting for an actual yield.")
    else:
        edit = st.data_editor(
            pending[["forecast_id", "field", "ratoon", "horizon", "pred_tph", "actual_yield"]],
            disabled=["forecast_id", "field", "ratoon", "horizon", "pred_tph"], hide_index=True, key="pending_editor",
            column_config={"actual_yield": st.column_config.NumberColumn("Actual yield (t/ha)", min_value=0.0, max_value=400.0)})
        st.caption("Only the imagery up to each forecast's as-of date is stored, so these cycles train horizons up to that age. "
                   "For full-cycle data, re-run tab 1 after harvest with the actual yield and use section A.")
        if st.button("Add completed forecasts to training store"):
            new, done = [], []
            for row in edit.itertuples():
                if pd.notna(row.actual_yield) and row.actual_yield > 0:
                    series = pd.read_csv(FORECAST_DIR / f"{row.forecast_id}.csv",
                                         dtype={"field": str, "variety": str, "irrigation": str, "cycle_id": str, "month": str})
                    series["yield_tph"] = float(row.actual_yield)
                    new.append(series[pp.LONG_COLS])
                    done.append((row.forecast_id, float(row.actual_yield)))
            if not new:
                st.warning("Enter at least one actual yield first.")
            else:
                merged, info = pp.upsert_cycles(pp.load_store(), pd.concat(new, ignore_index=True))
                pp.save_store(merged)
                for fid, y in done:
                    log.loc[log["forecast_id"] == fid, ["actual_yield", "added_to_store"]] = [y, True]
                log.to_csv(LOG_PATH, index=False)
                st.success(f"Added {info['added']} new / {info['updated']} updated cycle(s) to the store.")
                st.rerun()
        done_log = log[log["actual_yield"].notna()]
        if len(done_log):
            err = (done_log["pred_tph"] - done_log["actual_yield"])
            st.caption(f"Realised forecast error so far: mean {err.mean():+.1f} t/ha, MAE {err.abs().mean():.1f} t/ha "
                       f"over {len(done_log)} forecast(s).")

    st.markdown("#### C. Retrain")
    cfg = mt.load_last_config()
    st.write(f"Uses the last training settings: horizons {cfg.horizons}, tuning {'on' if cfg.tune else 'off'}, {cfg.n_splits} folds.")
    if st.button("♻ Retrain with the updated store", type="primary"):
        before = mt.load_registry()
        res = run_training(cfg)
        if res is not None and not res.metrics.empty:
            st.session_state["train_result"] = res
            cmp = res.metrics[["horizon", "n_cycles", "RMSE"]].rename(columns={"n_cycles": "cycles (new)", "RMSE": "RMSE (new)"})
            if before:
                old = pd.DataFrame(before["horizons"]).T.reset_index().rename(columns={"index": "horizon"})
                old["horizon"] = old["horizon"].astype(int)
                cmp = cmp.merge(old[["horizon", "n_cycles", "rmse"]].rename(columns={"n_cycles": "cycles (old)", "rmse": "RMSE (old)"}),
                                on="horizon", how="left")
            st.success("Models retrained. The previous version was archived.")
            st.dataframe(cmp.round(2), hide_index=True)
            st.caption("RMSEs come from different datasets, so a small change is not proof of improvement - judge the trend over several retrains.")
        elif res is not None:
            show_training_result(res)
    hist = mt.load_history()
    if len(hist) and hist["trained_at"].nunique() > 1:
        st.markdown("**Cross-validated RMSE over successive retrains**")
        st.line_chart(hist.pivot_table(index="trained_at", columns="horizon", values="RMSE"))


# --------------------------------------------------------------------------- #
# Tab 6 - explainability
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Computing SHAP values...")
def cached_global_explanation(horizon: int, trained_at: str):
    return ex.global_explanation(mt.load_bundle(horizon))


def tab_explain() -> None:
    st.subheader("Global model explainability (SHAP)")
    available = mt.list_horizons()
    if not available:
        st.info("Train models first (tab 3).")
        return
    hz = st.selectbox("Horizon (months)", available, key="ex_h")
    bundle = mt.load_bundle(hz)
    cv = bundle["cv"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cycles / fields", f"{bundle['n_cycles']} / {bundle['n_fields']}")
    c2.metric("CV RMSE", f"{cv['model']['rmse']:.1f} t/ha", f"{cv['model']['rmse'] - cv['no_imagery']['rmse']:+.1f} vs no-imagery", delta_color="inverse")
    c3.metric("CV R²", f"{cv['model']['r2']:.2f}")
    g = cached_global_explanation(hz, bundle["trained_at"])
    a, b = st.columns(2)
    a.markdown("**Mean |SHAP| (t/ha)**")
    a.pyplot(g.bar_fig)
    b.markdown("**Direction and spread of effects**")
    b.pyplot(g.beeswarm_fig)
    st.dataframe(g.importance.round(3), hide_index=True)
    with st.expander("Tuned hyper-parameters"):
        st.json(bundle["params"])


# --------------------------------------------------------------------------- #
def main() -> None:
    sidebar()
    st.title("Sugarcane Yield Forecasting")
    st.caption("Multi-temporal Sentinel-2 (CDSE) → vegetation indices → field-grouped XGBoost → SHAP-explained in-season forecasts")
    tabs = st.tabs(["1 · Sentinel-2 extraction", "2 · Historical data & QC", "3 · Train models",
                    "4 · Forecast", "5 · Continuous learning", "6 · Explainability"])
    for tab, fn in zip(tabs, (tab_extract, tab_data, tab_train, tab_forecast, tab_learning, tab_explain)):
        with tab:
            fn()


main()
