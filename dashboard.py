"""Customer 360 dashboard: segments, retention priorities, campaign targeting and customer lookup.

    streamlit run dashboard.py      (run `make pipeline` first)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer 360", page_icon="👥", layout="wide")
OUT = Path("outputs")
if not (OUT / "customer_scores.csv").exists():
    st.error("Run `make pipeline` first.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(OUT / "customer_scores.csv"), json.loads((OUT / "metrics.json").read_text())


scores, m = load()
st.title("👥 Customer 360")
st.caption("Loyalty analytics for a UAE retailer · 20,000 members · as of 30 June 2026")

k = st.columns(5)
k[0].metric("Members", f"{len(scores):,}")
k[1].metric("Active (90 days)", f"{scores['churn_risk'].notna().sum():,}")
k[2].metric("High churn risk", f"{(scores['churn_risk'] > 0.5).sum():,}")
k[3].metric("6-month value at risk", f"AED {scores['value_at_risk_aed'].sum() / 1e6:,.2f}M")
k[4].metric("Churn model AUC", m["churn"]["roc_auc"])

t1, t2, t3, t4 = st.tabs(["Segments", "Retention priorities", "Campaign targeting", "Customer lookup"])

with t1:
    seg = pd.DataFrame(m["segments"]).set_index("segment")
    c1, c2 = st.columns(2)
    c1.subheader("Share of sales")
    c1.bar_chart(seg["share_of_sales"], horizontal=True, height=280)
    c2.subheader("Members")
    c2.bar_chart(seg["customers"], horizontal=True, height=280)
    st.dataframe(seg.style.format({"share_of_sales": "{:.0%}", "avg_spend_365d": "AED {:,.0f}",
                                   "avg_baskets_365d": "{:.0f}", "avg_recency_days": "{:.0f}",
                                   "coupon_rate": "{:.1%}", "app_share": "{:.0%}"}), use_container_width=True)

with t2:
    st.markdown("Active members ranked by **value at risk** = churn probability × predicted 6-month spend. "
                "Call the top of this list first.")
    top = scores.dropna(subset=["churn_risk"]).sort_values("value_at_risk_aed", ascending=False).head(200)
    st.dataframe(top[["customer_id", "segment", "churn_risk", "clv_180d_aed", "value_at_risk_aed",
                      "next_best_action"]].style.format({"churn_risk": "{:.0%}", "clv_180d_aed": "AED {:,.0f}",
                                                          "value_at_risk_aed": "AED {:,.0f}"}),
                 use_container_width=True, height=420, hide_index=True)
    st.download_button("Download call list (CSV)", top.to_csv(index=False).encode(), "retention_call_list.csv")

with t3:
    st.markdown(f"April coupon test: on average the coupon added **AED {m['campaign_average_effect_aed']}** of spend per "
                "member, but who you send it to decides whether it makes money.")
    pol = pd.DataFrame(m["uplift_policies"])
    pol["targeting"] = pol["share_targeted"].map(lambda s: "Everyone" if s == 1 else f"Top {s:.0%} by uplift")
    st.bar_chart(pol.set_index("targeting")["incremental_profit_per_1000_members_aed"], height=260)
    st.caption("Incremental profit per 1,000 members (AED), measured on held-out customers from the randomised test.")
    st.subheader("Who responds?")
    st.bar_chart(scores.groupby("campaign_action")["customer_id"].count(), horizontal=True, height=180)
    q = pd.DataFrame(m["uplift_qini"]).set_index("decile")["observed_uplift_aed"]
    st.markdown("**Observed extra spend (treated − control) by predicted-uplift decile** (1 = most persuadable)")
    st.bar_chart(q, height=220)

with t4:
    cid = st.selectbox("Customer", scores["customer_id"].head(3000))
    r = scores.set_index("customer_id").loc[cid]
    c = st.columns(4)
    c[0].metric("Segment", r["segment"])
    c[1].metric("Churn risk", "–" if pd.isna(r["churn_risk"]) else f"{r['churn_risk']:.0%}")
    c[2].metric("Predicted 6-month spend", f"AED {r['clv_180d_aed']:,.0f}")
    c[3].metric("Coupon response", r["campaign_action"])
    st.markdown(f"**RFM:** R{r['R']} F{r['F']} M{r['M']}  \n**Recommended next products:** {r['recommended_products']}  \n"
                f"**Next best action:** {r['next_best_action']}")
