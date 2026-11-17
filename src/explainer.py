"""SHAP explanations (global and per-forecast) for the XGBoost yield models.

SHAP values are exact TreeSHAP values. ``shap.TreeExplainer`` is tried first; if the
installed shap/xgboost pair is incompatible (it has happened between major releases)
the same values are obtained from XGBoost's native ``pred_contribs`` - identical
algorithm, no dependency on shap's xgboost parser.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

_STAT_NAMES = {
    "mean": "mean", "max": "peak", "last": "latest", "auc": "cumulative (AUC)",
    "slope3": "trend, last 3 mo", "peak_age": "age at peak (months)",
}


def pretty(name: str) -> str:
    """Human-readable feature label, e.g. ``ndvi_auc`` -> ``NDVI cumulative (AUC)``."""
    for idx in ("ndvi", "savi", "evi", "ndre"):
        if name.startswith(idx + "_"):
            return f"{idx.upper()} {_STAT_NAMES.get(name[len(idx) + 1:], name[len(idx) + 1:])}"
    if name.startswith("variety_"):
        return f"Variety {name[8:]}"
    if name.startswith("irr_"):
        return f"Irrigation {name[4:].title()}"
    return {"cloud_mean": "Mean cloud cover (%)", "valid_frac": "Share of clear months",
            "ratoon": "Ratoon number", "cycle_months": "Cycle length (months)"}.get(name, name)


def _explanation(bundle: dict, X: pd.DataFrame):
    import shap

    X = X.reindex(columns=bundle["feature_names"]).astype(float)
    model = bundle["model"]
    try:
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X))
        base = float(np.ravel(explainer.expected_value)[0])
    except Exception:  # noqa: BLE001 - fall back to native TreeSHAP
        contribs = model.get_booster().predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
        values, base = contribs[:, :-1], float(contribs[0, -1])
    return shap.Explanation(values=values, base_values=np.full(len(X), base), data=X.to_numpy(),
                            feature_names=[pretty(c) for c in X.columns]), X


@dataclass
class GlobalExplanation:
    importance: pd.DataFrame
    bar_fig: plt.Figure
    beeswarm_fig: plt.Figure


@dataclass
class LocalExplanation:
    prediction: float
    base_value: float
    contributions: pd.DataFrame
    waterfall_fig: plt.Figure
    summary: str


def _grab(max_w: float = 8.0) -> plt.Figure:
    fig = plt.gcf()
    fig.set_size_inches(max_w, max(4.0, fig.get_size_inches()[1]))
    fig.tight_layout()
    plt.close(fig)
    return fig


def global_explanation(bundle: dict, max_display: int = 15) -> GlobalExplanation:
    """Mean |SHAP| ranking and bar / beeswarm plots over the model's training features."""
    import shap

    expl, X = _explanation(bundle, bundle["X_background"])
    imp = (pd.DataFrame({"feature": X.columns, "label": [pretty(c) for c in X.columns],
                         "mean_abs_shap": np.abs(expl.values).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    plt.figure()
    shap.plots.bar(expl, max_display=max_display, show=False)
    bar = _grab()
    plt.figure()
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    swarm = _grab()
    return GlobalExplanation(imp, bar, swarm)


def local_explanation(bundle: dict, x_row: pd.DataFrame, max_display: int = 12) -> LocalExplanation:
    """Waterfall + ranked contributions for a single forecast (``x_row`` has one row)."""
    import shap

    expl, X = _explanation(bundle, x_row.iloc[[0]])
    contrib = (pd.DataFrame({"feature": X.columns, "label": [pretty(c) for c in X.columns],
                             "value": X.iloc[0].to_numpy(), "shap_t_per_ha": expl.values[0]})
               .assign(abs_shap=lambda d: d["shap_t_per_ha"].abs())
               .sort_values("abs_shap", ascending=False).drop(columns="abs_shap").reset_index(drop=True))
    base = float(expl.base_values[0])
    pred = base + float(expl.values[0].sum())
    plt.figure()
    shap.plots.waterfall(expl[0], max_display=max_display, show=False)
    fig = _grab()

    up = contrib[contrib["shap_t_per_ha"] > 0].head(2)
    dn = contrib[contrib["shap_t_per_ha"] < 0].head(2)
    fmt = lambda df: ", ".join(f"{r.label} ({r.shap_t_per_ha:+.1f})" for r in df.itertuples()) or "none"  # noqa: E731
    summary = (f"Model baseline is {base:.1f} t/ha. Biggest upward drivers: {fmt(up)}. "
               f"Biggest downward drivers: {fmt(dn)}. (Contributions in t/ha.)")
    return LocalExplanation(pred, base, contrib, fig, summary)
