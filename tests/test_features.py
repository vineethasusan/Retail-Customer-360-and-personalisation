import pandas as pd

from cust360.features import future_spend, snapshot


def test_snapshot_ignores_future_purchases(d):
    f1 = snapshot(d, "2025-12-31")
    later = d.baskets[d.baskets["date"] > "2025-12-31"]
    assert later.size > 0
    cid = later["customer_id"].iloc[0]
    hist = d.baskets[(d.baskets.customer_id == cid) & (d.baskets.date <= "2025-12-31")]
    expected = hist[hist.date > pd.Timestamp("2025-12-31") - pd.Timedelta(days=365)]["basket_value_aed"].sum()
    assert abs(f1.loc[cid, "spend_365d"] - expected) < 1e-6


def test_category_shares_sum_to_one(d):
    f = snapshot(d, "2026-03-31")
    share = f[[c for c in f.columns if c.startswith("share_")]].sum(axis=1)
    assert ((share.round(6) == 1) | (share == 0)).all()


def test_future_spend_window(d):
    s = future_spend(d, "2026-01-31", 30)
    b = d.baskets[(d.baskets.date > "2026-01-31") & (d.baskets.date <= "2026-03-02")]
    assert abs(s.sum() - b.basket_value_aed.sum()) < 1e-6
