"""Campaign uplift modelling: who should get the coupon?

April 2026 coupon campaign was randomised (half of members got a 15% coupon), so the
true effect can be estimated. A "T-learner" trains one model on treated and one on
control customers; the difference of their predictions is each customer's expected
uplift. Customers are then split into:

  * Persuadables  - buy more only because of the coupon      -> target
  * Sure things   - buy anyway; a coupon just gives away margin -> don't target
  * Sleeping dogs - buy *less* when contacted                 -> never target

Evaluation uses 2-fold cross-fitting and a Qini-style curve on held-out customers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .features import Data, model_columns, snapshot

MARGIN = 0.25
DISCOUNT = 0.15


def campaign_frame(d: Data, as_of="2026-03-31", days=30):
    f = snapshot(d, as_of)
    f = f[f["ever_purchased"] == 1]
    t = pd.Timestamp(as_of)
    b = d.baskets[(d.baskets["date"] > t) & (d.baskets["date"] <= t + pd.Timedelta(days=days))]
    paid = b.groupby("customer_id")["basket_value_aed"].sum().reindex(f.index).fillna(0)
    discount_given = (b[b["coupon_used"] == 1].groupby("customer_id")["basket_value_aed"].sum()
                      * DISCOUNT / (1 - DISCOUNT)).reindex(f.index).fillna(0)
    # Outcome = spend at full price, so the coupon's cost is counted once (as discount_given), not twice
    spend = paid + discount_given
    treated = d.customers.set_index("customer_id")["campaign_treated"].reindex(f.index)
    return f, spend, treated, discount_given


def _model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80,
                                         random_state=42)


def fit_predict_uplift(f: pd.DataFrame, y: pd.Series, w: pd.Series, n_splits: int = 2) -> pd.Series:
    cols = model_columns(f)
    out = pd.Series(np.nan, index=f.index)
    for tr, te in KFold(n_splits, shuffle=True, random_state=0).split(f):
        ftr, ytr, wtr = f.iloc[tr], y.iloc[tr], w.iloc[tr]
        m1 = _model().fit(ftr.loc[wtr == 1, cols], ytr[wtr == 1])
        m0 = _model().fit(ftr.loc[wtr == 0, cols], ytr[wtr == 0])
        fte = f.iloc[te]
        out.iloc[te] = m1.predict(fte[cols]) - m0.predict(fte[cols])
    return out


def qini_table(uplift: pd.Series, y: pd.Series, w: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Observed treated-minus-control spend per decile of predicted uplift (decile 1 = highest)."""
    dec = pd.qcut(uplift.rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    rows = []
    for k, g in pd.DataFrame({"y": y, "w": w, "dec": dec}).groupby("dec", observed=True):
        rows.append({"decile": int(k), "customers": len(g),
                     "observed_uplift_aed": g.loc[g.w == 1, "y"].mean() - g.loc[g.w == 0, "y"].mean()})
    return pd.DataFrame(rows)


def policy_value(uplift: pd.Series, y: pd.Series, w: pd.Series, discount: pd.Series, share: float) -> dict:
    """Incremental profit per 1,000 members if we target the top `share` by predicted uplift."""
    n_target = int(share * len(uplift))
    target = uplift.rank(ascending=False, method="first") <= n_target
    t, c = target & (w == 1), target & (w == 0)
    inc_spend = y[t].mean() - y[c].mean() if share > 0 else 0.0
    cost = discount[t].mean() if share > 0 else 0.0
    per_customer = inc_spend * MARGIN - cost
    return {"share_targeted": share, "incremental_spend_per_target_aed": round(float(inc_spend), 2),
            "coupon_cost_per_target_aed": round(float(cost), 2),
            "incremental_profit_per_1000_members_aed": round(float(per_customer * n_target / len(uplift) * 1000), 0)}


def segment_label(u: float, threshold: float) -> str:
    if u < -threshold:
        return "Sleeping dog"
    if u > threshold:
        return "Persuadable"
    return "Sure thing / Lost cause"
