"""Download everything the model needs from RawTree: aggregates computed in SQL, saved as parquet.

    python -m twin.build train     # features from the first 40 days + a sample of later carousel sessions
    python -m twin.build live      # features from all the history + the live stream so far

Heavy lifting (co-visits, user profiles, popularity) is done by RawTree in SQL; we only download
the small aggregated results. Two snapshots keep the offline evaluation honest: the `train`
snapshot only sees events from before the sessions the ranker is trained and validated on.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from twin.rt import ARTIFACTS, q, q_csv

HISTORY_START = "2026-09-15 07:00:00"
HISTORY_END = "2026-11-14 07:00:00"
TRAIN_CUTOFF = "2026-10-25 07:00:00"   # features before, training sessions after
VALID_START = "2026-11-11 07:00:00"    # last 3 days of history: validation

CATALOG = """(SELECT product_id::String AS p, any(subcategory::String) AS sub FROM (
    SELECT product_id, subcategory FROM shop.catalog UNION ALL SELECT product_id, subcategory FROM shop.catalog_updates
) GROUP BY p)"""


def events(cutoff: str, live: bool) -> str:
    """Organic events before `cutoff` as a typed subquery (history, plus the live stream if `live`)."""
    cols = ("ts::DateTime64(3) AS ts, event_type::String AS et, user_id::String AS u, session_id::String AS s, "
            "product_id::String AS p, price_eur::Float64 AS price, device::String AS device, region::String AS region")
    sql = f"SELECT {cols} FROM shop.history_events WHERE ts::DateTime64(3) < '{cutoff}'"
    if live:
        sql += f" UNION ALL SELECT {cols} FROM shop.events WHERE ts::DateTime64(3) < '{cutoff}'"
    return f"({sql})"


def pairs(inner: str, top: int) -> str:
    return f"SELECT a, b, n FROM ({inner}) ORDER BY a, n DESC LIMIT {top} BY a SETTINGS max_execution_time = 170"


def queries(cutoff: str, live: bool) -> dict[str, str]:
    ev = events(cutoff, live)
    # Tables that carry their own timestamps are downloaded in full even for the `train` snapshot:
    # the model only ever looks at rows from before the session it is scoring.
    ev_all = events(cutoff if live else HISTORY_END, live)
    return {
        "item_stats": f"""
            SELECT p, countIf(et = 'view') AS views, countIf(et = 'add_to_cart') AS carts, countIf(et = 'purchase') AS purchases,
                   uniqIf(u, et = 'view') AS viewers,
                   countIf(et = 'view' AND ts >= toDateTime64('{cutoff}', 3) - INTERVAL 7 DAY) AS views_7d,
                   countIf(et = 'purchase' AND ts >= toDateTime64('{cutoff}', 3) - INTERVAL 7 DAY) AS purch_7d
            FROM {ev} GROUP BY p""",
        # viewed together in the same session
        "vv": pairs(f"""
            SELECT a, b, count() AS n FROM (
                SELECT arrayJoin(ps) AS a, ps FROM (
                    SELECT groupUniqArray(p) AS ps FROM {ev} WHERE et = 'view' GROUP BY s HAVING length(ps) BETWEEN 2 AND 40)
            ) ARRAY JOIN ps AS b WHERE a != b GROUP BY a, b""", 60),
        # viewed a, bought b in the same session
        "vp": pairs(f"""
            SELECT a, b, count() AS n FROM (
                SELECT arrayJoin(vs) AS a, bs FROM (
                    SELECT groupUniqArrayIf(p, et = 'view') AS vs, groupUniqArrayIf(p, et = 'purchase') AS bs
                    FROM {ev} GROUP BY s HAVING length(bs) > 0)
            ) ARRAY JOIN bs AS b WHERE a != b GROUP BY a, b""", 40),
        # bought by the same user (any time)
        "bb": pairs(f"""
            SELECT a, b, count() AS n FROM (
                SELECT arrayJoin(ps) AS a, ps FROM (
                    SELECT groupUniqArray(p) AS ps FROM {ev} WHERE et = 'purchase' GROUP BY u HAVING length(ps) BETWEEN 2 AND 100)
            ) ARRAY JOIN ps AS b WHERE a != b GROUP BY a, b""", 40),
        "user_sub": f"""
            SELECT e.u AS u, c.sub AS sub, countIf(e.et = 'view') AS v, countIf(e.et = 'add_to_cart') AS c, countIf(e.et = 'purchase') AS b
            FROM {ev} AS e INNER JOIN {CATALOG} AS c ON c.p = e.p GROUP BY u, sub""",
        "user_stats": f"""
            SELECT u, count() AS n_events, countIf(et = 'view') AS n_views, countIf(et = 'purchase') AS n_purch, uniq(s) AS n_sessions,
                   avgIf(price, et = 'view') AS view_price, avgIf(price, et = 'purchase') AS purch_price
            FROM {ev} GROUP BY u""",
        # the shopper's own history with each product: the strongest signal in the data
        "user_product": f"""
            SELECT u, p, countIf(et = 'view') AS v, countIf(et = 'add_to_cart') AS c FROM {ev} GROUP BY u, p""",
        "user_purchases": f"""
            SELECT u, p, toUnixTimestamp64Milli(ts) AS t FROM {ev_all} WHERE et = 'purchase'""",
        "seg_pop": f"""
            SELECT device, region, p, countIf(et = 'purchase') AS purchases, countIf(et = 'view') AS views
            FROM {ev} GROUP BY device, region, p""",
        # views per subcategory in 15-minute buckets: the trend signal
        "sub_trend": f"""
            SELECT c.sub AS sub, toUnixTimestamp(toStartOfFifteenMinutes(e.ts)) AS bucket, countIf(e.et = 'view') AS views
            FROM {ev_all} AS e INNER JOIN {CATALOG} AS c ON c.p = e.p GROUP BY sub, bucket""",
        # how each product did in the old carousel, by page and slot (to remove position bias later)
        "car_stats": f"""
            SELECT t.1 AS p, page, t.2 AS slot, count() AS imp, countIf(c = t.1) AS clicks, countIf(b = t.1) AS purch
            FROM (SELECT arrayJoin(arrayZip(shown::Array(String), range(1, 11))) AS t, `context.page`::String AS page,
                         ifNull(clicked::String, '') AS c, ifNull(purchased::String, '') AS b
                  FROM shop.history_sessions WHERE ts::DateTime64(3) < '{min(cutoff, HISTORY_END)}')
            GROUP BY p, page, slot""",
        # carousel purchases (not in the organic events): needed to never recommend something already bought
        "car_purchases": """
            SELECT user_id::String AS u, purchased::String AS p, toUnixTimestamp64Milli(ts::DateTime64(3)) AS t
            FROM shop.history_sessions WHERE purchased IS NOT NULL""",
    }


SESSIONS = """
SELECT request_id::String AS request_id, toUnixTimestamp64Milli(ts::DateTime64(3)) AS ts_ms, user_id::String AS user_id,
       `context.device`::String AS device, `context.page`::String AS page, `context.region`::String AS region,
       arrayStringConcat(arrayMap(x -> x.product_id::String, recent_views::Array(JSON)), ' ') AS rv,
       arrayStringConcat(shown::Array(String), ' ') AS shown,
       arrayStringConcat(arrayMap(x -> toString(x), propensity::Array(Float64)), ' ') AS prop,
       ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased, revenue_eur::Float64 AS revenue
FROM shop.history_sessions
WHERE ts::DateTime64(3) >= '{a}' AND ts::DateTime64(3) < '{b}'
  AND (clicked IS NOT NULL OR cityHash64(request_id::String) % {keep} = 0)"""
SESSION_KEEP = 4  # keep every session with a click and 1 in 4 of the others (weighted back in training)


PARTS = {"user_sub": 6, "user_purchases": 3, "user_stats": 2, "user_product": 12}   # big per-user tables: download by slices of users


def fetch(name: str, sql: str, out) -> None:
    t = time.time()
    k = PARTS.get(name, 1)
    if k == 1:
        df = q_csv(sql, timeout=240, retries=2)
    else:  # RawTree caps one result at 100 MiB
        df = pd.concat([q_csv(f"SELECT * FROM ({sql}) WHERE cityHash64(u) % {k} = {j} SETTINGS max_execution_time = 170",
                              timeout=240, retries=2) for j in range(k)], ignore_index=True)
    if name == "user_product":  # millions of rows: store ids as dictionary codes so loading stays fast
        df["u"], df["p"] = df.u.astype("category"), df.p.astype("category")
    df.to_parquet(out / f"{name}.parquet", index=False)
    print(f"  {name:15s} {len(df):>9,} rows  {time.time() - t:5.1f}s", flush=True)


def main(snapshot: str, only: list[str]) -> None:
    live = snapshot == "live"
    out = ARTIFACTS / snapshot
    out.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if live else TRAIN_CUTOFF
    print(f"snapshot '{snapshot}', events before {cutoff} UTC")

    for name, sql in queries(cutoff, live).items():
        if not only or name in only:
            fetch(name, sql, out)
    if only and "sessions" not in only:  # refreshing a few tables only
        return

    q("SELECT * FROM shop.catalog").to_parquet(out / "catalog.parquet", index=False)
    print("  catalog         done")
    if live:  # price changes, stock and new products so far; the live loop continues from here
        upd = q("SELECT *, toUnixTimestamp64Milli(ts::DateTime64(3)) AS ts_ms FROM shop.catalog_updates ORDER BY ts")
        upd.drop(columns=["ts"]).to_parquet(out / "catalog_updates.parquet", index=False)
        print(f"  catalog_updates {len(upd):>9,} rows")

    if not live:
        day, end, parts = datetime.fromisoformat(TRAIN_CUTOFF), datetime.fromisoformat(HISTORY_END), []
        while day < end:
            nxt = min(day + timedelta(days=2), end)
            parts.append(q_csv(SESSIONS.format(a=day, b=nxt, keep=SESSION_KEEP), timeout=240, retries=2,
                               dtype={"rv": str, "clicked": str, "purchased": str}, keep_default_na=False))
            print(f"  sessions {day:%m-%d} -> {nxt:%m-%d}: {len(parts[-1]):,}", flush=True)
            day = nxt
        sessions = pd.concat(parts, ignore_index=True).rename(columns={"ts_ms": "ts"})
        sessions.to_parquet(out / "sessions.parquet", index=False)

    (out / "meta.txt").write_text(f"cutoff={cutoff}\nlive={live}\n")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train", sys.argv[2:])
