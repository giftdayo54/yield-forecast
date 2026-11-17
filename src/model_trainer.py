"""Multi-horizon XGBoost yield models with leakage-safe, field-grouped validation.

Design notes
------------
* One model per forecast horizon ``h`` = months since planting/ratoon start. The model
  for horizon ``h`` only sees imagery from ages 0..h, so it can be used in-season.
* **Leakage control** - every cycle of a field stays in the same CV fold
  (``GroupKFold`` on Field ID). Cycles of one field share soil, management and
  geometry, so random splits would be optimistic.
* **Honest tuning** - hyper-parameter search is *nested*: inside each outer training
  fold a ``RandomizedSearchCV`` (also grouped by field) picks parameters, and the
  outer test fold is only ever predicted. The reported CV scores therefore include
  the cost of tuning. A baseline (training-fold mean yield) is reported alongside.
* Two baselines are reported with every CV: the training-fold mean yield, and a
  **no-imagery model** (same XGBoost on ratoon / variety / irrigation / cycle length
  only). The gap between the model and the no-imagery baseline is the value that
  Sentinel-2 actually adds - with small datasets that is the number to watch.
* Empirical prediction intervals come from the out-of-fold residuals.
* Models are saved with ``joblib``; previous versions are archived on retraining.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from . import MODELS_DIR, __version__
from . import preprocessing as pp

ARCHIVE_DIR = MODELS_DIR / "archive"
REGISTRY_PATH = MODELS_DIR / "registry.json"
HISTORY_PATH = MODELS_DIR / "training_history.csv"
CONFIG_PATH = MODELS_DIR / "last_train_config.json"

DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=3, reg_lambda=5.0, reg_alpha=0.1, gamma=0.0,
)
PARAM_DIST = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(2, 5),
    "learning_rate": loguniform(0.02, 0.2),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 6),
    "reg_lambda": loguniform(1, 30),
    "reg_alpha": loguniform(1e-3, 5),
    "gamma": uniform(0, 2),
}
MIN_CYCLES = 15
MIN_FIELDS = 3

ProgressCB = Callable[[float, str], None]


@dataclass
class TrainConfig:
    horizons: list[int] = field(default_factory=lambda: [3, 5, 7, 9])
    n_splits: int = 5
    tune: bool = True
    n_iter: int = 20
    inner_splits: int = 3
    nested_cv: bool = True
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TrainResult:
    metrics: pd.DataFrame
    oof: pd.DataFrame
    skipped: list[tuple[int, str]]
    trained_at: str


# --------------------------------------------------------------------------- #
# Estimators and metrics
# --------------------------------------------------------------------------- #
def make_regressor(params: dict | None = None, random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_jobs=1,
        random_state=random_state, verbosity=0, **{**DEFAULT_PARAMS, **(params or {})},
    )


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "mape_pct": float(np.mean(np.abs((y - p) / y)) * 100),
        "bias": float(np.mean(p - y)),
    }


def _fit(X, y, groups, cfg: TrainConfig, tune: bool):
    n_groups = groups.nunique()
    inner = min(cfg.inner_splits, n_groups)
    if tune and inner >= 2:
        search = RandomizedSearchCV(
            make_regressor(random_state=cfg.random_state), PARAM_DIST, n_iter=cfg.n_iter,
            cv=GroupKFold(n_splits=inner), scoring="neg_root_mean_squared_error",
            random_state=cfg.random_state, n_jobs=cfg.n_jobs, refit=True,
        )
        search.fit(X, y, groups=groups)
        return search.best_estimator_, {k: _py(v) for k, v in search.best_params_.items()}
    model = make_regressor(random_state=cfg.random_state)
    model.fit(X, y)
    return model, dict(DEFAULT_PARAMS)


def non_imagery_columns(columns) -> list[str]:
    """Agronomic features only (no satellite-derived columns)."""
    return [c for c in columns if c in ("ratoon", "cycle_months") or c.startswith(("variety_", "irr_"))]


def _py(v):
    return v.item() if hasattr(v, "item") else v


def cross_validate_horizon(X, y, groups, cfg: TrainConfig):
    """Field-grouped (nested, if tuning) CV. Returns (oof_df, fold_metrics, summary)."""
    n_splits = min(cfg.n_splits, groups.nunique())
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct fields for grouped cross-validation.")
    oof = pd.Series(np.nan, index=X.index)
    base = pd.Series(np.nan, index=X.index)
    noimg = pd.Series(np.nan, index=X.index)
    agr = non_imagery_columns(X.columns)
    fold_rows = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        model, _ = _fit(X.iloc[tr], y.iloc[tr], groups.iloc[tr], cfg, tune=cfg.tune and cfg.nested_cv)
        oof.iloc[te] = model.predict(X.iloc[te])
        base.iloc[te] = y.iloc[tr].mean()
        agr_model = make_regressor(random_state=cfg.random_state)
        agr_model.fit(X[agr].iloc[tr], y.iloc[tr])
        noimg.iloc[te] = agr_model.predict(X[agr].iloc[te])
        m = regression_metrics(y.iloc[te], oof.iloc[te])
        fold_rows.append({"fold": k, "n_test": len(te), "fields_test": groups.iloc[te].nunique(), **m})
    out = pd.DataFrame({"cycle_id": X.index, "field": groups.to_numpy(), "y_true": y.to_numpy(),
                        "y_pred": oof.to_numpy(), "baseline": base.to_numpy(),
                        "no_imagery": noimg.to_numpy()})
    summary = {"model": regression_metrics(y, oof), "baseline": regression_metrics(y, base),
               "no_imagery": regression_metrics(y, noimg)}
    return out, pd.DataFrame(fold_rows), summary


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def train_all(long_clean: pd.DataFrame, cfg: TrainConfig | None = None,
              progress_cb: ProgressCB | None = None, save: bool = True) -> TrainResult:
    """Train and (optionally) persist one model per horizon from a QC-cleaned long table."""
    cfg = cfg or TrainConfig()
    cats = pp.get_categories(long_clean)
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_rows, oofs, skipped, bundles = [], [], [], {}
    horizons = sorted(set(cfg.horizons))

    for i, h in enumerate(horizons):
        if progress_cb:
            progress_cb(i / len(horizons), f"Horizon {h} months: cross-validating and tuning...")
        X, y, groups = pp.build_training_set(long_clean, h, cats)
        if len(y) < MIN_CYCLES or groups.nunique() < MIN_FIELDS:
            skipped.append((h, f"only {len(y)} usable cycles / {groups.nunique()} fields "
                               f"(need >= {MIN_CYCLES} / {MIN_FIELDS})"))
            continue
        oof, folds, summary = cross_validate_horizon(X, y, groups, cfg)
        model, params = _fit(X, y, groups, cfg, tune=cfg.tune)

        resid = oof["y_true"] - oof["y_pred"]
        q_lo, q_hi = (float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90))) if len(resid) >= 10 \
            else (-1.28 * summary["model"]["rmse"], 1.28 * summary["model"]["rmse"])
        bundles[h] = {
            "app_version": __version__, "horizon": h, "model": model, "params": params,
            "feature_names": list(X.columns), "categories": cats, "trained_at": trained_at,
            "n_cycles": int(len(y)), "n_fields": int(groups.nunique()),
            "cv": summary, "interval_residual_q": (q_lo, q_hi), "X_background": X.copy(),
            "cfg": cfg.to_dict(),
        }
        oof.insert(0, "horizon", h)
        oofs.append(oof)
        m, b, a = summary["model"], summary["baseline"], summary["no_imagery"]
        metric_rows.append({
            "horizon": h, "n_cycles": len(y), "n_fields": groups.nunique(),
            "RMSE": m["rmse"], "MAE": m["mae"], "R2": m["r2"], "MAPE %": m["mape_pct"], "bias": m["bias"],
            "mean-baseline RMSE": b["rmse"], "no-imagery RMSE": a["rmse"],
            "imagery gain %": 100 * (1 - m["rmse"] / a["rmse"]) if a["rmse"] else np.nan,
        })

    if progress_cb:
        progress_cb(1.0, "Done")
    metrics = pd.DataFrame(metric_rows)
    result = TrainResult(metrics, pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame(), skipped, trained_at)
    if save and bundles:
        _persist(bundles, result, cfg)
    return result


def _persist(bundles: dict, result: TrainResult, cfg: TrainConfig) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    for h, b in bundles.items():
        path = MODELS_DIR / f"yield_h{h}.joblib"
        if path.exists():
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            shutil.move(str(path), ARCHIVE_DIR / f"yield_h{h}_{stamp}.joblib")
        joblib.dump(b, path)
    # drop stale horizons that were not retrained so the registry stays consistent
    for old in list_horizons():
        if old not in bundles:
            shutil.move(str(MODELS_DIR / f"yield_h{old}.joblib"), ARCHIVE_DIR / f"yield_h{old}_superseded.joblib")
    REGISTRY_PATH.write_text(json.dumps({
        "app_version": __version__, "trained_at": result.trained_at,
        "categories": next(iter(bundles.values()))["categories"],
        "horizons": {str(h): {"n_cycles": b["n_cycles"], "n_fields": b["n_fields"],
                              "rmse": b["cv"]["model"]["rmse"], "r2": b["cv"]["model"]["r2"]} for h, b in bundles.items()},
    }, indent=2))
    hist = result.metrics.assign(trained_at=result.trained_at)[
        ["trained_at", "horizon", "n_cycles", "n_fields", "RMSE", "MAE", "R2", "no-imagery RMSE"]]
    hist.to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)
    CONFIG_PATH.write_text(json.dumps(cfg.to_dict(), indent=2))


# --------------------------------------------------------------------------- #
# Loading / predicting
# --------------------------------------------------------------------------- #
def list_horizons() -> list[int]:
    return sorted(int(p.stem.split("_h")[1]) for p in MODELS_DIR.glob("yield_h*.joblib"))


def load_bundle(horizon: int) -> dict:
    path = MODELS_DIR / f"yield_h{horizon}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model for horizon {horizon}. Train models first.")
    return joblib.load(path)


def load_registry() -> dict | None:
    return json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else None


def load_history() -> pd.DataFrame:
    return pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()


def load_last_config() -> TrainConfig:
    if CONFIG_PATH.exists():
        try:
            d = json.loads(CONFIG_PATH.read_text())
            return TrainConfig(**{k: v for k, v in d.items() if k in TrainConfig().__dict__})
        except Exception:  # noqa: BLE001
            pass
    return TrainConfig()


def pick_horizon(available: list[int], last_complete_age: int) -> int | None:
    """Largest trained horizon that the observed imagery already covers."""
    ok = [h for h in available if h <= last_complete_age]
    return max(ok) if ok else None


def predict_yield(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Point forecast plus an empirical 80% interval from out-of-fold residuals."""
    X = X.reindex(columns=bundle["feature_names"]).astype(float)
    pred = bundle["model"].predict(X)
    q_lo, q_hi = bundle["interval_residual_q"]
    return pd.DataFrame({"pred": pred, "lo": np.maximum(pred + q_lo, 0), "hi": pred + q_hi}, index=X.index)


def forecast_cycle(long: pd.DataFrame, planting, harvest, as_of, qc_cfg: pp.QCConfig | None = None,
                   horizon="Auto") -> dict:
    """In-season forecast for one cycle from its (partial) tidy time series.

    ``long`` must hold one cycle (see ``pp.extraction_to_long``). The same QC used for
    training is applied (without dropping the cycle). With ``horizon="Auto"`` the longest
    trained horizon already covered by *complete* months is used.
    """
    planned = pp.months_between(planting, harvest)
    clean = pp.apply_qc(long, qc_cfg, for_training=False).clean
    available = list_horizons()
    if not available:
        raise RuntimeError("No trained models found - train models first.")
    last_complete = pp.months_between(planting, as_of)
    if pd.Timestamp(as_of) != pd.Timestamp(as_of) + pd.offsets.MonthEnd(0):
        last_complete -= 1  # the as-of month is still in progress
    hz = pick_horizon(available, last_complete) if horizon == "Auto" else int(horizon)
    if hz is None:
        raise ValueError(f"Not enough imagery yet: {last_complete} complete month(s) after planting, "
                         f"earliest trained horizon is {min(available)}.")
    if int(clean["age"].max()) < hz:
        raise ValueError(f"Imagery only reaches month {int(clean['age'].max())}; horizon {hz} needs more.")
    bundle = load_bundle(hz)
    X, _ = pp.build_features(clean, hz, bundle["categories"])
    first = clean.iloc[0]
    notes = []
    if planned <= hz:
        notes.append(f"Planned cycle ({planned} months) is not longer than the horizon ({hz}); forecast may be unreliable.")
    if str(first["variety"]) not in bundle["categories"]["variety"]:
        notes.append(f"Variety '{first['variety']}' was not in the training data; its variety flags are all zero.")
    if str(first["irrigation"]) not in bundle["categories"]["irrigation"]:
        notes.append(f"Irrigation '{first['irrigation']}' was not in the training data.")
    if X.iloc[0].isna().mean() > 0.3:
        notes.append("Many features are missing (cloudy months?) - treat this forecast with caution.")
    return {"pred": predict_yield(bundle, X).iloc[0], "X": X, "bundle": bundle, "clean": clean,
            "horizon": hz, "notes": notes, "cycle_id": first["cycle_id"]}


def plot_observed_vs_predicted(oof: pd.DataFrame, horizon: int):
    import matplotlib.pyplot as plt

    d = oof[oof["horizon"] == horizon]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    lim = [min(d.y_true.min(), d.y_pred.min()) - 5, max(d.y_true.max(), d.y_pred.max()) + 5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.scatter(d.y_true, d.y_pred, c="#2E7D32", alpha=0.75, edgecolor="white")
    ax.set(xlim=lim, ylim=lim, xlabel="Observed yield (t/ha)", ylabel="Out-of-fold predicted (t/ha)",
           title=f"Horizon {horizon} months - grouped CV")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig
