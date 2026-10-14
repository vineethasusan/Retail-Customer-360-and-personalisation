"""Customer features as of a snapshot date (only data before the snapshot is used)."""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_CODES = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}


class Data:
    """Loads the raw tables once and joins basket lines to categories."""

    def __init__(self, folder: str = "data", tables=None):
        if tables is not None:
            cust, bask, line, prod = tables
        else:
            cust = pd.read_csv(f"{folder}/customers.csv")
            bask = pd.read_csv(f"{folder}/baskets.csv.gz", parse_dates=["date"])
            line = pd.read_csv(f"{folder}/basket_lines.csv.gz")
            prod = pd.read_csv(f"{folder}/products.csv")
        bask = bask.copy()
        bask["date"] = pd.to_datetime(bask["date"])
        self.customers = cust
        self.baskets = bask
        self.products = prod
        line = line.merge(prod[["product_id", "category"]], on="product_id")
        line["line_value"] = line["qty"] * line["unit_price_aed"]
        self.lines = line.merge(bask[["basket_id", "customer_id", "date"]], on="basket_id")
        self.categories = sorted(prod["category"].unique())


def snapshot(d: Data, as_of: str | pd.Timestamp) -> pd.DataFrame:
    t = pd.Timestamp(as_of)
    b = d.baskets[d.baskets["date"] <= t]
    f = d.customers[["customer_id", "emirate", "age_band", "member_since", "app_user"]].copy().set_index("customer_id")
    g = b.groupby("customer_id")
    f["last_purchase"] = g["date"].max()
    f["recency_days"] = (t - f["last_purchase"]).dt.days.fillna(9999)
    for days in (30, 90, 180, 365):
        w = b[b["date"] > t - pd.Timedelta(days=days)].groupby("customer_id")
        f[f"baskets_{days}d"] = w.size()
        f[f"spend_{days}d"] = w["basket_value_aed"].sum()
    f["spend_prev_90d"] = b[(b["date"] > t - pd.Timedelta(days=180)) & (b["date"] <= t - pd.Timedelta(days=90))] \
        .groupby("customer_id")["basket_value_aed"].sum()
    f["avg_basket_aed"] = g["basket_value_aed"].mean()
    f["app_share"] = b.assign(app=(b["channel"] == "app").astype(int)).groupby("customer_id")["app"].mean()
    f["coupon_rate"] = g["coupon_used"].mean()
    f["active_months_365d"] = b[b["date"] > t - pd.Timedelta(days=365)].groupby("customer_id")["date"] \
        .apply(lambda s: s.dt.to_period("M").nunique())
    f = f.fillna({c: 0 for c in f.columns if c.startswith(("baskets_", "spend_", "app_share", "coupon_rate",
                                                             "active_months", "avg_basket"))})
    f["spend_trend"] = (f["spend_90d"] - f["spend_prev_90d"]) / (f["spend_prev_90d"] + 50)
    f["tenure_days"] = (t - pd.to_datetime(f["member_since"])).dt.days

    l = d.lines[(d.lines["date"] <= t) & (d.lines["date"] > t - pd.Timedelta(days=365))]
    share = l.pivot_table(index="customer_id", columns="category", values="line_value", aggfunc="sum", fill_value=0)
    share = share.div(share.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    share.columns = [f"share_{c.lower().replace(' & ', '_').replace(' ', '_')}" for c in share.columns]
    f = f.join(share).fillna(0)
    f["age_code"] = f["age_band"].map(AGE_CODES)
    f["emirate_code"] = f["emirate"].astype("category").cat.codes
    f["ever_purchased"] = (f["recency_days"] < 9999).astype(int)
    return f.drop(columns=["member_since", "last_purchase"])


def model_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in ("emirate", "age_band")]


def future_spend(d: Data, start, days: int) -> pd.Series:
    s = pd.Timestamp(start)
    b = d.baskets[(d.baskets["date"] > s) & (d.baskets["date"] <= s + pd.Timedelta(days=days))]
    return b.groupby("customer_id")["basket_value_aed"].sum().reindex(d.customers["customer_id"]).fillna(0)
