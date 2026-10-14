"""Synthetic loyalty-programme data for a UAE retailer.

20,000 loyalty members, 2 years of baskets (line items), and one randomised
single-use 15% coupon campaign (April 2026). Customers belong to hidden behavioural personas, churn over
time, and react differently to coupons, including "sleeping dogs" who are
annoyed by marketing. Hidden truth (persona, true campaign effect) is kept in
separate columns so models can be evaluated honestly; models never see it.

    python -m cust360.data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2024-07-01"), pd.Timestamp("2026-06-30")
CAMPAIGN_START, CAMPAIGN_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30")

CATEGORIES = {
    # category: (products, avg price AED)
    "Fresh & Dairy": (["Milk 2L", "Labneh", "Halloumi", "Eggs 30", "Bananas", "Tomatoes", "Chicken Breast", "Yogurt"], 14),
    "Pantry": (["Basmati Rice 5kg", "Sunflower Oil", "Pasta", "Chickpeas", "Sugar 2kg", "Tea 100 bags"], 22),
    "Dates & Sweets": (["Medjool Dates", "Khalas Dates", "Maamoul", "Chocolate Box"], 35),
    "Beverages": (["Water 6-pack", "Laban", "Orange Juice", "Energy Drink", "Arabic Coffee"], 12),
    "Baby & Kids": (["Diapers", "Baby Formula", "Kids Snacks", "Baby Wipes"], 65),
    "Personal Care": (["Shampoo", "Body Wash", "Toothpaste", "Skincare Serum", "Razor Pack"], 28),
    "Perfume & Gifts": (["Oud Perfume", "Bakhoor", "Gift Hamper", "Luxury Candle"], 160),
    "Electronics": (["Earbuds", "Power Bank", "Phone Case", "Smart Watch Strap"], 95),
    "Home": (["Detergent", "Cleaning Spray", "Storage Box", "Bedsheet Set"], 40),
    "Ready Meals": (["Shawarma Wrap", "Salad Bowl", "Sushi Box", "Sandwich"], 25),
}

PERSONAS = {
    # share, baskets/month, items/basket, category weights, coupon effect on visit rate, monthly churn hazard
    "Family Weekly Shopper": (0.28, 4.5, 6, {"Fresh & Dairy": 5, "Pantry": 4, "Baby & Kids": 3, "Beverages": 2, "Home": 2, "Dates & Sweets": 1}, 0.0, 0.010),
    "Young Professional":    (0.22, 6.0, 2, {"Ready Meals": 5, "Beverages": 4, "Fresh & Dairy": 2, "Personal Care": 2, "Electronics": 1}, 0.10, 0.030),
    "Premium Gifter":        (0.10, 1.5, 3, {"Perfume & Gifts": 5, "Dates & Sweets": 3, "Personal Care": 2, "Electronics": 1}, -0.06, 0.015),
    "Deal Seeker":           (0.20, 2.5, 4, {"Pantry": 4, "Home": 3, "Personal Care": 3, "Beverages": 2, "Fresh & Dairy": 2}, 0.60, 0.025),
    "Occasional Visitor":    (0.20, 0.8, 2, {"Beverages": 3, "Fresh & Dairy": 2, "Ready Meals": 2, "Pantry": 1, "Electronics": 1}, 0.12, 0.040),
}
EMIRATES = ["Dubai", "Dubai", "Dubai", "Sharjah", "Sharjah", "Abu Dhabi", "Abu Dhabi", "Ajman", "Ras Al Khaimah"]
RAMADAN = [("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19")]


def _catalogue():
    rows = []
    pid = 0
    for cat, (names, price) in CATEGORIES.items():
        for n in names:
            pid += 1
            rows.append((f"P{pid:03d}", n, cat, round(price * np.random.default_rng(pid).uniform(0.6, 1.5), 2)))
    return pd.DataFrame(rows, columns=["product_id", "product", "category", "price"])


def generate(n_customers: int = 20_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    cat_df = _catalogue()
    by_cat = {c: (g["product_id"].tolist(), g["price"].tolist()) for c, g in cat_df.groupby("category")}
    personas = list(PERSONAS)
    shares = np.array([PERSONAS[p][0] for p in personas])
    months = pd.date_range(START, END, freq="MS")

    customers, baskets, lines = [], [], []
    bid = 0
    for i in range(n_customers):
        cid = f"C{i + 1:05d}"
        persona = personas[rng.choice(len(personas), p=shares)]
        _, rate, items, weights, coupon_eff, hazard = PERSONAS[persona]
        rate *= rng.lognormal(0, 0.35)
        items = max(1, int(round(items * rng.uniform(0.7, 1.3))))
        joined = START - pd.Timedelta(days=int(rng.integers(0, 1500)))
        age = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                         p=[0.12, 0.34, 0.28, 0.16, 0.10] if persona != "Young Professional" else [0.3, 0.55, 0.12, 0.02, 0.01])
        app_user = rng.random() < (0.75 if persona in ("Young Professional", "Deal Seeker") else 0.4)
        treated = rng.random() < 0.5          # randomised campaign assignment (A/B test)
        cats = list(weights)
        w = np.array([weights[c] for c in cats], dtype=float)
        w /= w.sum()

        churned_at = None
        for m in months:
            if churned_at is not None:
                break
            # churn hazard rises for customers whose engagement is low
            if rng.random() < hazard * (1.5 if rate < 1 else 1):
                churned_at = m
                break
            lam = rate
            if any(pd.Timestamp(a) <= m <= pd.Timestamp(b) for a, b in RAMADAN):
                lam *= 1.25
            if m.month in (7, 8):
                lam *= 0.75                      # summer travel
            in_campaign = m == CAMPAIGN_START
            true_effect = 0.0
            if in_campaign and treated:
                true_effect = coupon_eff * rate
                lam = max(0.05, lam + true_effect)
            n_b = rng.poisson(lam)
            days = np.sort(rng.integers(0, m.days_in_month, size=n_b))
            coupon_left = in_campaign and treated and rng.random() < 0.6   # single-use coupon, 60% redeem
            for d in days:
                bid += 1
                date = m + pd.Timedelta(days=int(d))
                if date > END:
                    continue
                k = max(1, rng.poisson(items))
                chosen = rng.choice(cats, size=k, p=w)
                coupon_used = coupon_left
                coupon_left = False
                total = 0.0
                for c in chosen:
                    ids, prices = by_cat[c]
                    j = int(rng.integers(0, len(ids)))
                    qty = int(rng.integers(1, 3))
                    price = prices[j] * (0.85 if coupon_used else 1.0)
                    lines.append((bid, ids[j], qty, round(price, 2)))
                    total += qty * price
                baskets.append((bid, cid, date, "app" if (app_user and rng.random() < 0.5) else "store",
                                round(total, 2), int(coupon_used)))
        customers.append((cid, rng.choice(EMIRATES), age, joined.date(), int(app_user), int(treated), persona,
                          churned_at.date() if churned_at is not None else None))

    cust = pd.DataFrame(customers, columns=["customer_id", "emirate", "age_band", "member_since", "app_user",
                                            "campaign_treated", "_persona", "_churned_month"])
    bask = pd.DataFrame(baskets, columns=["basket_id", "customer_id", "date", "channel", "basket_value_aed",
                                          "coupon_used"])
    line = pd.DataFrame(lines, columns=["basket_id", "product_id", "qty", "unit_price_aed"])
    return cust, bask, line, cat_df


if __name__ == "__main__":
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)
    cust, bask, line, cat = generate()
    cust.to_csv("data/customers.csv", index=False)
    bask.to_csv("data/baskets.csv.gz", index=False)
    line.to_csv("data/basket_lines.csv.gz", index=False)
    cat.to_csv("data/products.csv", index=False)
    print(f"{len(cust):,} customers | {len(bask):,} baskets | {len(line):,} line items | "
          f"AED {bask.basket_value_aed.sum() / 1e6:,.1f}M sales")
