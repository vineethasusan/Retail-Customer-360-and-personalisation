"""Churn and customer lifetime value (CLV) models, validated out-of-time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import Data, future_spend, model_columns, snapshot

CHURN_WINDOW = 90   # churned = no purchase in the next 90 days (among customers active in the last 90)
CLV_WINDOW = 180    # CLV target = spend over the next 180 days


def churn_dataset(d: Data, as_of):
    f = snapshot(d, as_of)
    f = f[f["baskets_90d"] > 0]
    fut = future_spend(d, as_of, CHURN_WINDOW)
    y = (fut.reindex(f.index) == 0).astype(int)
    return f, y


def train_churn(d: Data, train_as_of="2025-09-30", test_as_of="2025-12-31"):
    ftr, ytr = churn_dataset(d, train_as_of)
    fte, yte = churn_dataset(d, test_as_of)
    cols = model_columns(ftr)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                       random_state=42).fit(ftr[cols], ytr)
    p = m.predict_proba(fte[cols])[:, 1]
    top = np.argsort(-p)[: int(0.2 * len(p))]
    metrics = {"roc_auc": round(roc_auc_score(yte, p), 3), "pr_auc": round(average_precision_score(yte, p), 3),
               "churn_rate": round(float(yte.mean()), 3),
               "churners_caught_in_top_20pct": round(float(yte.iloc[top].sum() / max(yte.sum(), 1)), 3),
               "train_snapshot": train_as_of, "test_snapshot": test_as_of}
    # simple global importance: drop in AUC when a feature is shuffled
    rng = np.random.default_rng(0)
    base = metrics["roc_auc"]
    imp = {}
    for c in cols:
        x = fte[cols].copy()
        x[c] = rng.permutation(x[c].values)
        imp[c] = base - roc_auc_score(yte, m.predict_proba(x)[:, 1])
    drivers = pd.Series(imp).sort_values(ascending=False).head(8).round(4)
    return m, cols, metrics, drivers


def train_clv(d: Data, train_as_of="2025-06-30", test_as_of="2025-12-31"):
    ftr = snapshot(d, train_as_of)
    fte = snapshot(d, test_as_of)
    ytr = future_spend(d, train_as_of, CLV_WINDOW).reindex(ftr.index)
    yte = future_spend(d, test_as_of, CLV_WINDOW).reindex(fte.index)
    cols = model_columns(ftr)
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      min_samples_leaf=40, random_state=42).fit(ftr[cols], ytr)
    p = np.clip(m.predict(fte[cols]), 0, None)
    baseline = fte["spend_180d"].to_numpy()                # "next 6 months = last 6 months"
    top = np.argsort(-p)[: int(0.1 * len(p))]
    top_bl = np.argsort(-baseline)[: int(0.1 * len(p))]
    metrics = {
        "mae_aed": round(float(np.abs(yte - p).mean()), 1),
        "baseline_mae_aed": round(float(np.abs(yte - baseline).mean()), 1),
        "spearman": round(float(spearmanr(yte, p)[0]), 3),
        "top10pct_value_captured": round(float(yte.iloc[top].sum() / yte.sum()), 3),
        "baseline_top10pct_value_captured": round(float(yte.iloc[top_bl].sum() / yte.sum()), 3),
        "train_snapshot": train_as_of, "test_snapshot": test_as_of,
    }
    return m, cols, metrics
