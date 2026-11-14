"""Snapshot sintètic amb els mateixos esquemes que `twin.build`, per provar el twin sense RawTree.

    from tests.twin_synth import make_snapshot
    make_snapshot(carpeta / "live", live=True)      # i, per a train, live=False (afegeix sessions.parquet)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = "2026-11-14 16:00:00"
CUT_MS = 1794672000000 - 0  # s'ajusta a sota amb to_ms
REGIONS = ["Andalucia", "Catalunya", "Madrid", "Valencia"]
DEVICES = ["web", "ios", "android"]
PAGES = ["home", "product", "category"]


def make_snapshot(out: Path, live: bool = True, n_products: int = 400, n_users: int = 300, seed: int = 3) -> dict:
    from twin.state import to_ms
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    cut = to_ms(CUTOFF)
    pid = [f"p_{i:04d}" for i in range(n_products)]
    cats = [f"cat{i % 10}" for i in range(n_products)]
    subs = [f"sub{i % 30:02d}" for i in range(n_products)]
    catalog = pd.DataFrame({
        "product_id": pid, "category": cats, "subcategory": subs, "brand": [f"b{i % 17}" for i in range(n_products)],
        "price_eur": np.round(rng.uniform(5, 200, n_products), 2),
        "description": [f"nice {subs[i]} item number {i % 7} quality" for i in range(n_products)],
        "created_at": "2026-08-01 00:00:00", "is_active": rng.random(n_products) > 0.02})
    catalog.to_parquet(out / "catalog.parquet", index=False)
    if live:
        upd = pd.DataFrame([
            {**catalog.iloc[5].to_dict(), "change": "out_of_stock", "ts_ms": cut - 5000},
            {**catalog.iloc[9].to_dict(), "change": "price_change", "price_eur": 3.0, "ts_ms": cut - 4000},
            {"product_id": "p_new1", "category": "cat1", "subcategory": "sub01", "brand": "b1", "price_eur": 20.0,
             "description": "nice sub01 item number 1 quality", "created_at": CUTOFF, "is_active": True,
             "change": "new_product", "ts_ms": cut - 3000}])
        upd.to_parquet(out / "catalog_updates.parquet", index=False)
    views = rng.poisson(300, n_products) * (1 + 5 * (rng.random(n_products) < 0.1))
    pd.DataFrame({"p": pid, "views": views, "carts": views // 8, "purchases": views // 25, "viewers": views // 2,
                  "views_7d": views // 9, "purch_7d": views // 200}).to_parquet(out / "item_stats.parquet", index=False)
    for name in ("vv", "vp", "bb"):
        a = rng.integers(0, n_products, 6000)
        b = (a + rng.integers(1, 30, 6000) * (1 if name != "bb" else 7)) % n_products
        pd.DataFrame({"a": np.array(pid)[a], "b": np.array(pid)[b], "n": rng.integers(1, 40, 6000)}) \
            .drop_duplicates(["a", "b"]).to_parquet(out / f"{name}.parquet", index=False)
    rows = [(p, page, slot, 200, int(rng.integers(0, 6)), int(rng.integers(0, 2)))
            for p in pid[:120] for page in PAGES for slot in range(1, 11)]
    pd.DataFrame(rows, columns=["p", "page", "slot", "imp", "clicks", "purch"]).to_parquet(out / "car_stats.parquet", index=False)
    seg = [(d, r, p, int(rng.integers(0, 9)), int(rng.integers(0, 90))) for d in DEVICES for r in REGIONS for p in pid[::3]]
    pd.DataFrame(seg, columns=["device", "region", "p", "purchases", "views"]).to_parquet(out / "seg_pop.parquet", index=False)
    buckets = (cut // 1000 // 900 - np.arange(1, 200)) * 900
    pd.DataFrame([(s, int(b), int(rng.integers(0, 50))) for s in sorted(set(subs)) for b in buckets],
                 columns=["sub", "bucket", "views"]).to_parquet(out / "sub_trend.parquet", index=False)
    users = [f"u_{i:04d}" for i in range(n_users)]
    nv = rng.integers(1, 200, n_users)
    pd.DataFrame({"u": users, "n_events": nv + 5, "n_views": nv, "n_purch": nv // 20, "n_sessions": nv // 4 + 1,
                  "view_price": rng.uniform(10, 100, n_users), "purch_price": rng.uniform(10, 100, n_users)}) \
        .to_parquet(out / "user_stats.parquet", index=False)
    us = [(u, sorted(set(subs))[int(s)], int(rng.integers(1, 30)), int(rng.integers(0, 4)), int(rng.integers(0, 2)))
          for u in users for s in rng.choice(30, 4, replace=False)]
    pd.DataFrame(us, columns=["u", "sub", "v", "c", "b"]).to_parquet(out / "user_sub.parquet", index=False)
    buys = [(u, pid[int(p)], int(cut - rng.integers(1, 40) * 86_400_000)) for u in users[::2] for p in rng.choice(n_products, 3, replace=False)]
    pd.DataFrame(buys, columns=["u", "p", "t"]).to_parquet(out / "user_purchases.parquet", index=False)
    pd.DataFrame(buys[:40], columns=["u", "p", "t"]).to_parquet(out / "car_purchases.parquet", index=False)
    up = pd.DataFrame([(u, pid[int(p)], int(rng.integers(1, 9)), int(rng.integers(0, 2)))
                       for u in users for p in rng.choice(n_products, 12, replace=False)], columns=["u", "p", "v", "c"])
    up["u"], up["p"] = up.u.astype("category"), up.p.astype("category")
    up.to_parquet(out / "user_product.parquet", index=False)
    if not live:   # sessions del carrusel antic: la meitat abans i la meitat després de VALID_START
        from twin.build import VALID_START
        v0, sess = to_ms(VALID_START), []
        bought_of = {}
        for u, p, t in buys:
            bought_of.setdefault(u, []).append(p)
        for k in range(1500):
            u = users[int(rng.integers(0, n_users))]
            shown = list(rng.choice(pid[:150], 10, replace=False))
            if u in bought_of and rng.random() < 0.5:
                shown[int(rng.integers(0, 10))] = bought_of[u][0]      # a vegades es mostra un producte ja comprat
            shown = list(dict.fromkeys(shown))
            while len(shown) < 10:
                c = pid[int(rng.integers(150, 300))]
                if c not in shown:
                    shown.append(c)
            click = ""
            for p in shown:                                              # els ja comprats es cliquen molt més
                if rng.random() < (0.5 if p in bought_of.get(u, ()) else 0.04):
                    click = p
                    break
            sess.append({"request_id": f"r{k}", "ts": int(v0 + (rng.random() - 0.5) * 4 * 86_400_000), "user_id": u,
                         "device": DEVICES[k % 3], "page": PAGES[k % 3], "region": REGIONS[k % 4],
                         "rv": " ".join(rng.choice(pid, int(rng.integers(0, 4)), replace=False)), "shown": " ".join(shown),
                         "prop": " ".join(["0.1"] * 10), "clicked": click,
                         "purchased": click if click and rng.random() < 0.5 else "", "revenue": 0.0})
        pd.DataFrame(sess).to_parquet(out / "sessions.parquet", index=False)
    (out / "meta.txt").write_text(f"cutoff={CUTOFF}\nlive={live}\n")
    return {"pid": pid, "users": users, "cut": cut, "buys": buys}
