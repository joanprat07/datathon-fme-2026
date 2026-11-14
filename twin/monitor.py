"""How are we doing live? Reads our feedback and compares model versions.

    python -m twin.monitor          # last 60 minutes
    python -m twin.monitor 180      # last 180 minutes

`baseline-holdout` rows are the share of requests we answer with the best sellers on purpose:
the difference with `twin-*` on the same minutes is the lift of the model.
"""
from __future__ import annotations

import sys

import pandas as pd

from twin.rt import TEAM, q

pd.set_option("display.width", 200)


def main(minutes: int) -> None:
    recent = f"request_id::String != '__init__' AND ts::DateTime64(3) > now() - INTERVAL {minutes} MINUTE"
    print(f"{TEAM}, last {minutes} minutes\n")

    print("Reliability (the status light needs more than 90% answered by us)")
    print(q(f"""
        SELECT if(served_by::String = 'team', 'answered by us', concat('fallback: ', fallback_reason::String)) AS outcome,
               count() AS requests, round(100 * count() / sum(count()) OVER (), 1) AS pct,
               round(quantile(0.5)(latency_ms::Float64)) AS latency_p50_ms, round(quantile(0.95)(latency_ms::Float64)) AS latency_p95_ms
        FROM {TEAM}.feedback WHERE {recent} GROUP BY outcome ORDER BY requests DESC""").to_string(index=False))

    print("\nBy model version (requests we answered in time)")
    print(q(f"""
        SELECT s.model_version AS model_version, count() AS requests,
               round(100 * countIf(f.clicked != '') / count(), 2) AS click_pct,
               round(100 * countIf(f.purchased != '') / count(), 2) AS purchase_pct,
               round(sum(f.revenue) / count(), 3) AS revenue_per_request
        FROM (SELECT request_id::String AS rid, clicked::String AS clicked, purchased::String AS purchased,
                     revenue_eur::Float64 AS revenue FROM {TEAM}.feedback WHERE {recent} AND served_by::String = 'team') AS f
        INNER JOIN (SELECT request_id::String AS rid, any(model_version::String) AS model_version
                    FROM {TEAM}.submissions GROUP BY rid) AS s ON s.rid = f.rid
        GROUP BY model_version ORDER BY requests DESC""").to_string(index=False))

    print("\nBy model version and kind of request (cold = new shopper)")
    print(q(f"""
        SELECT s.model_version AS model_version, r.page AS page, r.is_new AS new_shopper, count() AS requests,
               round(100 * countIf(f.purchased != '') / count(), 2) AS purchase_pct
        FROM (SELECT request_id::String AS rid, purchased::String AS purchased FROM {TEAM}.feedback
              WHERE {recent} AND served_by::String = 'team') AS f
        INNER JOIN (SELECT request_id::String AS rid, any(model_version::String) AS model_version
                    FROM {TEAM}.submissions GROUP BY rid) AS s ON s.rid = f.rid
        INNER JOIN (SELECT request_id::String AS rid, `context.page`::String AS page, is_new_user::Bool AS is_new
                    FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL {minutes + 5} MINUTE) AS r ON r.rid = f.rid
        GROUP BY model_version, page, new_shopper ORDER BY model_version, page, new_shopper""").to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
