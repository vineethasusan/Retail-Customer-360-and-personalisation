# Customer 360: results

## Segments (as of 30 June 2026)

| Segment | Customers | Share of sales | Avg spend (12m) | Avg visits (12m) | Days since last visit |
|---|---|---|---|---|---|
| Family Stock-up Shoppers | 4,897 | 63% | AED 14,885 | 53 | 27 |
| Convenience Regulars | 2,901 | 15% | AED 5,847 | 65 | 54 |
| Deal Seekers | 3,207 | 12% | AED 4,335 | 27 | 56 |
| Premium Gifters | 1,588 | 9% | AED 6,857 | 17 | 49 |
| Lapsed | 6,571 | 1% | AED 202 | 2 | 442 |
| Never Purchased | 836 | 0% | AED 0 | 0 | – |

## Churn (no purchase in the next 90 days)

- ROC-AUC **0.862** out-of-time (trained on Sep 2025, tested on Dec 2025 snapshot)
- Contacting the riskiest 20% reaches **75%** of customers who churn (base churn rate 8%)
- Top drivers: recency_days, baskets_365d, spend_prev_90d, spend_180d, spend_30d

## Customer lifetime value (spend over the next 6 months)

- Mean absolute error **AED 913.5** vs AED 1,022.3 for "next 6 months = last 6 months"
- The top 10% predicted customers deliver **37%** of future value (baseline 36%)

## Recommendations (new products customers go on to buy)

- Hit rate @5: **83%** vs 65% for best-sellers (**+28%**)

## April coupon campaign: who should get it?

Average effect of the coupon across everyone: AED 42.94 extra spend per member.

| Targeting | Extra spend per targeted member | Coupon cost per targeted member | Profit per 1,000 members |
|---|---|---|---|
| Everyone | AED 42.94 | AED 10.92 | AED -184 |
| Top 50% by uplift | AED 66.21 | AED 12.44 | AED 2,055 |
| Top 30% by uplift | AED 91.5 | AED 16.68 | AED 1,860 |
| Top 20% by uplift | AED 105.12 | AED 17.41 | AED 1,774 |

Sanity check against the simulator's hidden personas (average predicted uplift, AED):

- Deal Seeker: 148.8
- Young Professional: 37.9
- Occasional Visitor: 0.8
- Family Weekly Shopper: -8.0
- Premium Gifter: -45.9
