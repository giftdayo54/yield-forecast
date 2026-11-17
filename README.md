# 🌾 Sugarcane Yield Forecasting (Sentinel-2 + XGBoost + SHAP)

Streamlit app for multi-temporal, in-season sugarcane yield forecasting.

1. **Download** monthly Sentinel-2 L2A reflectance (B02, B03, B04, B05, B08) from the Copernicus Data Space Ecosystem (CDSE) for a GeoJSON field and a planting → harvest window.
2. **Compute** monthly NDVI, SAVI, EVI, NDRE and export them in the exact layout of `Timeseries data General.xlsx`.
3. **Manage historical data**: parse the merged-header Excel layout, run QC (cloud and index-bound filtering, gap interpolation), keep a persistent training store.
4. **Train** one XGBoost model per forecast horizon with field-grouped, nested cross-validation; save with `joblib`.
5. **Forecast** in-season for a new field/cycle, with an empirical 80 % interval and SHAP explanations.
6. **Keep learning**: add harvested cycles (uploaded workbooks or logged forecasts + actual yield), retrain, and compare with the previous model.

```
sugarcane-yield-app/
├── .streamlit/config.toml
├── data/            sample_historical.xlsx (SYNTHETIC template), sample_field.geojson, training_store.csv (created at runtime)
├── models/          yield_h{N}.joblib, registry.json, training_history.csv, archive/
├── outputs/         CV metrics, forecast_log.csv, forecasts/
├── src/             cdse_downloader.py · preprocessing.py · model_trainer.py · explainer.py
├── scripts/         make_sample_data.py
├── tests/           test_pipeline.py (offline)
├── app.py · requirements.txt · README.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Try it without CDSE credentials: tab 2 → *Bundled synthetic sample* → **Merge into training store** → tab 3 → **Train** → tab 4 → *Upload workbook from step 1* (any workbook produced by tab 1 or the sample's first cycle).

### CDSE credentials

Create an OAuth client at <https://shapps.dataspace.copernicus.eu/dashboard/> → *User settings → OAuth clients*. Then provide it via **one** of:

```bash
export CDSE_CLIENT_ID=sh-xxxxxxxx  CDSE_CLIENT_SECRET=xxxxxxxx
```
```toml
# .streamlit/secrets.toml  (git-ignored)
CDSE_CLIENT_ID = "sh-xxxxxxxx"
CDSE_CLIENT_SECRET = "xxxxxxxx"
```
or type them in the sidebar (kept only in the session). **Never commit credentials.** If a secret was ever pasted into a notebook or repo, revoke it in the dashboard and create a new one.

## Data layout (matches the historical workbook)

| FIELD | IRRIGATION SYSTEM | VARIETY | RATOON | YIELD(Tons per ha) | Month | AGE | Cloud Cover (%) | NDVI | SAVI | EVI | NDRE |
|---|---|---|---|---|---|---|---|---|---|---|---|

* One crop cycle = one block; the first five columns are merged over the block, the first month sits on the header row, AGE is blank there (age 0), blocks are separated by one blank row.
* The parser also copes with: stray label rows without data (e.g. a lone `FURROW`), cycles whose first AGE is 1 (no age-0 row), numeric varieties (`579`) vs text (`N32`), gaps in the month sequence, and Excel dates in the Month column.
* `Cloud Cover (%)` is the **scene-level** `eo:cloud_cover` of the best image of the month, the same definition as the original notebook.

## How the model works

* **Horizon models** – a model for horizon *h* uses only months 0…*h* after planting, so it can run in-season. Per index (NDVI/SAVI/EVI/NDRE): mean, max, latest, area-under-curve, 3-month slope, age at peak; plus mean cloud cover, share of clear months, ratoon, planned cycle length, variety and irrigation flags.
* **No harvest-month leakage** – cycles are only used for horizon *h* if they are still growing at *h*; the harvest-month image (bare field) never enters a feature.
* **Leakage-safe validation** – `GroupKFold` by Field ID; hyper-parameter search (`RandomizedSearchCV`) is nested inside each outer fold, so reported scores include the cost of tuning.
* **Baselines in every report** – training-fold mean, and a **no-imagery model** (ratoon, variety, irrigation, cycle length). *"Imagery gain %"* is what Sentinel-2 adds. Check it before trusting a horizon.
* **Uncertainty** – the 80 % interval is the 10th–90th percentile of out-of-fold residuals added to the point forecast (empirical, not a guarantee).
* **Explainability** – exact TreeSHAP (`shap.TreeExplainer`, with an XGBoost `pred_contribs` fallback): global bar/beeswarm and a per-forecast waterfall.
* **Continuous learning** – the store (`data/training_store.csv`) upserts by cycle id. Retraining archives previous models in `models/archive/` and appends CV scores to `models/training_history.csv`.

### Read this before relying on the numbers

* **Small data dominates.** On the 64-cycle / 11-field historical workbook used during development, ratoon number and variety alone explained most of the variance (a no-imagery XGBoost reached R² ≈ 0.92), and imagery added little (a few % RMSE at 3 months, nothing at 6+). Later horizons do **not** automatically do better: extra monthly features overfit at this sample size. More fields and cycles are the main lever.
* **Cycle length is an input.** `cycle_months` is the planned planting→harvest length at forecast time and the realised length in training. If harvest is routinely moved, forecasts inherit that mismatch.
* **Extraction changes vs the notebook** (to be fixed deliberately, but they shift values slightly): no-data pixels are excluded via `dataMask`; per-pixel EVI is clipped to [-1, 2] before averaging (unclipped means reached > 30,000); months with no scene are kept as blank rows; optional SCL cloud masking is **off** by default so new data stay comparable with history. Re-extract a few historical cycles and compare before mixing old and new data.
* The CDSE download code could not be run against the live service during development (no network access to CDSE); the index maths, month splitting and request construction are unit-tested, but run one real field end-to-end first.

## Deployment notes

Streamlit Community Cloud has an ephemeral filesystem: `data/training_store.csv`, `models/` and `outputs/` are lost on restart. For persistent use, host on a VM/container with a mounted volume, or swap the store/model paths for cloud storage.

## Tests

```bash
pytest -q        # 17 offline tests: index maths, parser quirks, QC, leakage guards, grouped CV, train→forecast→SHAP→retrain
python scripts/make_sample_data.py   # regenerate the synthetic sample
```
