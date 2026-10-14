"""Next-best-product recommendations: item-to-item collaborative filtering.

"Customers who buy what you buy also buy…" Products are compared by who buys them
(cosine similarity on a TF-IDF weighted customer x product matrix, so very popular
items like water don't dominate). Evaluated out-of-time: train on purchases up to a
date, then check whether customers go on to buy a recommended product they had never
bought before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Data


class ItemCF:
    def fit(self, lines: pd.DataFrame):
        m = lines.groupby(["customer_id", "product_id"])["qty"].sum().unstack(fill_value=0)
        self.customers, self.products = m.index, m.columns
        x = np.log1p(m.to_numpy(float))
        idf = np.log(len(x) / (1 + (x > 0).sum(axis=0)))
        x = x * idf
        norms = np.linalg.norm(x, axis=0) + 1e-9
        self.sim = (x.T @ x) / np.outer(norms, norms)
        np.fill_diagonal(self.sim, 0)
        self.user = x
        self.bought = m.to_numpy() > 0
        self.popularity = (m.to_numpy() > 0).sum(axis=0)
        return self

    def recommend(self, customer_id: str, k: int = 5, new_only: bool = True) -> list[str]:
        if customer_id not in self.customers:
            return list(self.products[np.argsort(-self.popularity)[:k]])
        i = self.customers.get_loc(customer_id)
        scores = self.user[i] @ self.sim
        if new_only:
            scores = np.where(self.bought[i], -np.inf, scores)
        return list(self.products[np.argsort(-scores)[:k]])

    def recommend_popular(self, customer_id: str, k: int = 5) -> list[str]:
        scores = self.popularity.astype(float)
        if customer_id in self.customers:
            scores = np.where(self.bought[self.customers.get_loc(customer_id)], -np.inf, scores)
        return list(self.products[np.argsort(-scores)[:k]])


def evaluate(d: Data, cutoff="2026-03-31", horizon_days=90, k=5) -> dict:
    t = pd.Timestamp(cutoff)
    train = d.lines[d.lines["date"] <= t]
    test = d.lines[(d.lines["date"] > t) & (d.lines["date"] <= t + pd.Timedelta(days=horizon_days))]
    model = ItemCF().fit(train)
    bought_before = train.groupby("customer_id")["product_id"].agg(set)
    new_buys = test.groupby("customer_id")["product_id"].agg(set)
    hits_cf = hits_pop = n = 0
    for cid, prods in new_buys.items():
        new = prods - bought_before.get(cid, set())
        if not new or cid not in model.customers:
            continue
        n += 1
        hits_cf += bool(set(model.recommend(cid, k)) & new)
        hits_pop += bool(set(model.recommend_popular(cid, k)) & new)
    return {"customers_evaluated": n, f"hit_rate_at_{k}": round(hits_cf / max(n, 1), 3),
            f"popularity_hit_rate_at_{k}": round(hits_pop / max(n, 1), 3),
            "lift_vs_popularity": round((hits_cf / max(hits_pop, 1)) - 1, 3)}
