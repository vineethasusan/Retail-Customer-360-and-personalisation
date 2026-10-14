"""End-to-end Customer 360 run.

    python -m cust360.pipeline
Outputs (in outputs/): customer_scores.csv (one row per member with segment, churn risk,
CLV, value at risk, campaign action and recommendations), metrics.json and report.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as datagen
from . import models, recommend, segments, uplift
from .features import Data, snapshot

OUT = Path("outputs")
AS_OF = "2026-06-30"


def ensure_data():
    if not Path("data/customers.csv").exists():
        Path("data").mkdir(exist_ok=True)
        cust, bask, line, cat = datagen.generate()
        cust.to_csv("data/customers.csv", index=False)
        bask.to_csv("data/baskets.csv.gz", index=False)
        line.to_csv("data/basket_lines.csv.gz", index=False)
        cat.to_csv("data/products.csv", index=False)


def run(d: Data, out: Path = OUT) -> dict:
    out.mkdir(exist_ok=True)
    t0 = time.time()
    print("1/5 Segmenting customers…")
    f_now = snapshot(d, AS_OF)
    seg, seg_summary = segments.segment(f_now)
    rfm = segments.rfm_scores(f_now)

    print("2/5 Churn model (out-of-time validation)…")
    churn_m, churn_cols, churn_metrics, churn_drivers = models.train_churn(d)
    print("3/5 CLV model (out-of-time validation)…")
    clv_m, clv_cols, clv_metrics = models.train_clv(d)

    print("4/5 Recommender evaluation…")
    rec_metrics = recommend.evaluate(d)
    rec_model = recommend.ItemCF().fit(d.lines[d.lines["date"] <= AS_OF])

    print("5/5 Campaign uplift model…")
    f_c, y_c, w_c, disc_c = uplift.campaign_frame(d)
    u = uplift.fit_predict_uplift(f_c, y_c, w_c)
    qini = uplift.qini_table(u, y_c, w_c)
    policies = [uplift.policy_value(u, y_c, w_c, disc_c, s) for s in (1.0, 0.5, 0.3, 0.2)]
    ate = y_c[w_c == 1].mean() - y_c[w_c == 0].mean()
    thr = float(np.nanstd(u)) * 0.5

    # ---- score every member today
    scores = pd.DataFrame(index=f_now.index)
    scores["segment"] = seg
    scores = scores.join(rfm)
    active = f_now["baskets_90d"] > 0
    scores["churn_risk"] = np.where(active, churn_m.predict_proba(f_now[churn_cols])[:, 1], np.nan)
    scores["clv_180d_aed"] = np.clip(clv_m.predict(f_now[clv_cols]), 0, None).round(0)
    scores["value_at_risk_aed"] = (scores["churn_risk"].fillna(0) * scores["clv_180d_aed"]).round(0)
    scores["campaign_uplift_aed"] = u.reindex(scores.index).round(1)
    scores["campaign_action"] = scores["campaign_uplift_aed"].map(
        lambda x: uplift.segment_label(x, thr) if pd.notna(x) else "No history")
    names = d.products.set_index("product_id")["product"]
    scores["recommended_products"] = [", ".join(names[p] for p in rec_model.recommend(c, 3)) for c in scores.index]
    scores["next_best_action"] = np.select(
        [scores["value_at_risk_aed"] > scores["value_at_risk_aed"].quantile(0.9),
         scores["campaign_action"] == "Persuadable",
         scores["campaign_action"] == "Sleeping dog"],
        ["Retention call / personal offer", "Send coupon", "Do not contact"], default="Personalised recommendations")
    scores.reset_index().to_csv(out / "customer_scores.csv", index=False)

    # ---- check against hidden truth (only possible because the data is simulated)
    persona = d.customers.set_index("customer_id")["_persona"]
    uplift_by_persona = u.groupby(persona.reindex(u.index)).mean().round(1).sort_values(ascending=False)

    metrics = {"churn": churn_metrics, "churn_top_drivers": churn_drivers.to_dict(), "clv": clv_metrics,
               "recommender": rec_metrics, "campaign_average_effect_aed": round(float(ate), 2),
               "uplift_policies": policies, "uplift_qini": qini.round(1).to_dict(orient="records"),
               "predicted_uplift_by_true_persona": uplift_by_persona.to_dict(),
               "segments": seg_summary.round(3).reset_index().to_dict(orient="records"),
               "runtime_s": round(time.time() - t0)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_report(metrics, seg_summary, out)
    return metrics


def write_report(m: dict, seg_summary: pd.DataFrame, out: Path):
    c, v, r = m["churn"], m["clv"], m["recommender"]
    L = ["# Customer 360: results", "", "## Segments (as of 30 June 2026)", "",
         "| Segment | Customers | Share of sales | Avg spend (12m) | Avg visits (12m) | Days since last visit |",
         "|---|---|---|---|---|---|"]
    L += [f"| {s} | {int(x.customers):,} | {x.share_of_sales:.0%} | AED {x.avg_spend_365d:,.0f} | "
          f"{x.avg_baskets_365d:.0f} | {'–' if pd.isna(x.avg_recency_days) else format(x.avg_recency_days, '.0f')} |"
          for s, x in seg_summary.iterrows()]
    L += ["", "## Churn (no purchase in the next 90 days)", "",
          f"- ROC-AUC **{c['roc_auc']}** out-of-time (trained on Sep 2025, tested on Dec 2025 snapshot)",
          f"- Contacting the riskiest 20% reaches **{c['churners_caught_in_top_20pct']:.0%}** of customers who churn "
          f"(base churn rate {c['churn_rate']:.0%})",
          f"- Top drivers: {', '.join(list(m['churn_top_drivers'])[:5])}", "",
          "## Customer lifetime value (spend over the next 6 months)", "",
          f"- Mean absolute error **AED {v['mae_aed']:,}** vs AED {v['baseline_mae_aed']:,} for "
          "\"next 6 months = last 6 months\"",
          f"- The top 10% predicted customers deliver **{v['top10pct_value_captured']:.0%}** of future value "
          f"(baseline {v['baseline_top10pct_value_captured']:.0%})", "",
          "## Recommendations (new products customers go on to buy)", "",
          f"- Hit rate @5: **{r['hit_rate_at_5']:.0%}** vs {r['popularity_hit_rate_at_5']:.0%} for best-sellers "
          f"(**{r['lift_vs_popularity']:+.0%}**)", "",
          "## April coupon campaign: who should get it?", "",
          f"Average effect of the coupon across everyone: AED {m['campaign_average_effect_aed']} extra spend per member.", "",
          "| Targeting | Extra spend per targeted member | Coupon cost per targeted member | Profit per 1,000 members |",
          "|---|---|---|---|"]
    L += [f"| {'Everyone' if p['share_targeted'] == 1 else 'Top ' + format(p['share_targeted'], '.0%') + ' by uplift'} | "
          f"AED {p['incremental_spend_per_target_aed']} | AED {p['coupon_cost_per_target_aed']} | "
          f"AED {p['incremental_profit_per_1000_members_aed']:,.0f} |" for p in m["uplift_policies"]]
    L += ["", "Sanity check against the simulator's hidden personas (average predicted uplift, AED):", ""]
    L += [f"- {k}: {v}" for k, v in m["predicted_uplift_by_true_persona"].items()]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main():
    ensure_data()
    d = Data("data")
    m = run(d)
    print(f"\nDone in {m['runtime_s']}s -> outputs/report.md")
    print(json.dumps({k: m[k] for k in ("churn", "clv", "recommender", "campaign_average_effect_aed")}, indent=1))
    for p in m["uplift_policies"]:
        print(p)
    print(m["predicted_uplift_by_true_persona"])


if __name__ == "__main__":
    main()
