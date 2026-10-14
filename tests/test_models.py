from cust360 import models, recommend, segments, uplift
from cust360.features import snapshot


def test_segments_have_business_names(d):
    seg, summary = segments.segment(snapshot(d, "2026-06-30"))
    assert summary["customers"].sum() == len(seg)
    assert not any(str(s).isdigit() for s in seg.unique())


def test_churn_model_is_useful(d):
    _, _, metrics, drivers = models.train_churn(d)
    assert metrics["roc_auc"] > 0.7
    assert "recency_days" in drivers.index[:5]


def test_clv_beats_naive_baseline(d):
    _, _, m = models.train_clv(d)
    assert m["mae_aed"] < m["baseline_mae_aed"]


def test_recommender_beats_best_sellers(d):
    r = recommend.evaluate(d)
    assert r["hit_rate_at_5"] >= r["popularity_hit_rate_at_5"]


def test_recommendations_are_new_products(d):
    m = recommend.ItemCF().fit(d.lines)
    cid = d.lines["customer_id"].iloc[0]
    bought = set(d.lines.loc[d.lines.customer_id == cid, "product_id"])
    assert not set(m.recommend(cid, 5)) & bought


def test_uplift_ranks_deal_seekers_above_premium_gifters(d):
    f, y, w, _ = uplift.campaign_frame(d)
    u = uplift.fit_predict_uplift(f, y, w)
    persona = d.customers.set_index("customer_id")["_persona"].reindex(u.index)
    by = u.groupby(persona).mean()
    assert by["Deal Seeker"] > by["Premium Gifter"]
