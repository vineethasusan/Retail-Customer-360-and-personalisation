"""Behavioural segmentation: RFM + shopping-mission clustering, with plain-English segment names."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["recency_days", "baskets_365d", "spend_365d", "avg_basket_aed", "coupon_rate", "app_share"]


def rfm_scores(f: pd.DataFrame) -> pd.DataFrame:
    act = f[f["ever_purchased"] == 1]
    r = pd.qcut(act["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    fq = pd.qcut(act["baskets_365d"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(act["spend_365d"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return pd.DataFrame({"R": r, "F": fq, "M": m}).reindex(f.index).fillna(0).astype(int)


def name_cluster(p: pd.Series) -> str:
    """Rule-based naming from the cluster profile, so business users get labels, not numbers."""
    if p["recency_days"] > 120:
        return "Lapsed"
    if p["share_perfume_gifts"] > 0.35:
        return "Premium Gifters"
    if p["coupon_rate"] > 0.04 or p["share_home"] + p["share_pantry"] > 0.45 and p["avg_basket_aed"] < 250:
        return "Deal Seekers"
    if p["share_ready_meals"] > 0.25:
        return "Convenience Regulars"
    if p["avg_basket_aed"] > 250 and p["baskets_365d"] > 25:
        return "Family Stock-up Shoppers"
    if p["baskets_365d"] < 12:
        return "Occasional Shoppers"
    return "Core Regulars"


def segment(f: pd.DataFrame, k: int = 7, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    act = f[f["ever_purchased"] == 1]
    share_cols = [c for c in f.columns if c.startswith("share_")]
    Xs = act[SEG_FEATURES].copy()
    Xs[["recency_days", "baskets_365d", "spend_365d", "avg_basket_aed"]] = np.log1p(
        Xs[["recency_days", "baskets_365d", "spend_365d", "avg_basket_aed"]])
    Xs = pd.concat([Xs, act[share_cols] * 2], axis=1)   # weight shopping mission
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(Xs))
    prof = act.assign(cluster=labels).groupby("cluster")[SEG_FEATURES + share_cols].mean()
    prof["segment"] = prof.apply(name_cluster, axis=1)
    seg = pd.Series(labels, index=act.index).map(prof["segment"])
    seg = seg.reindex(f.index).fillna("Never Purchased")
    summary = f.assign(segment=seg, recency_days=f["recency_days"].where(f["ever_purchased"] == 1)).groupby("segment").agg(
        customers=("recency_days", "size"), avg_spend_365d=("spend_365d", "mean"),
        avg_baskets_365d=("baskets_365d", "mean"), avg_recency_days=("recency_days", "mean"),
        coupon_rate=("coupon_rate", "mean"), app_share=("app_share", "mean"))
    summary["share_of_sales"] = f.assign(segment=seg).groupby("segment")["spend_365d"].sum() / f["spend_365d"].sum()
    return seg, summary.sort_values("share_of_sales", ascending=False)
