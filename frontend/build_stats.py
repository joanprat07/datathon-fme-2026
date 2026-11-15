"""Summarise the real snapshots (data/, model/, logs/) into frontend_assets/real_stats.json
and a real catalogue sample (frontend_assets/catalog_sample.json) for the live simulation.

    python frontend/build_stats.py      # from the repo root, after conv.build and conv.train

The presentation only reads the JSON, so it runs anywhere without the parquet files.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from conv import config as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "frontend_assets" / "real_stats.json"
SAMPLE = OUT.with_name("catalog_sample.json")


def q(x: pd.Series) -> list[float]:
    """Box: p5, q1, median, q3, p95."""
    return [round(float(v), 2) for v in np.percentile(x, [5, 25, 50, 75, 95])]


def build_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = re.search(r"\s(\w+)\s+(\d+) rows\s+([\d.]+)s", line)
        if m:
            rows.append({"table": m[1], "rows": int(m[2]), "secs": float(m[3])})
    return rows


def main() -> None:
    cat = pd.read_parquet(ROOT / "data/live/catalog.parquet")
    upd = pd.read_parquet(ROOT / "data/live/catalog_updates.parquet")
    items = pd.read_parquet(ROOT / "data/live/item_init.parquet")
    ses = pd.read_parquet(ROOT / "data/train/sessions.parquet")
    model = json.loads((ROOT / "model/model.json").read_text())

    # ---- catalogue
    by_cat = []
    for c, g in cat.groupby("category"):
        by_cat.append({"cat": c, "n": len(g), "subs": g.subcategory.nunique(), "box": q(g.price_eur)})
    by_cat.sort(key=lambda d: -d["box"][2])

    # ---- popularity long tail (15-day decayed purchases)
    b = np.sort(items.b2.to_numpy())[::-1]
    share = np.cumsum(b) / b.sum()
    lorenz = [round(float(share[int(len(b) * f) - 1]), 4) for f in np.linspace(0.01, 1, 100)]

    # ---- sessions: what the old carousel showed and what happened (history_sessions)
    shown = ses.shown.str.split()
    clicked = ses.clicked.fillna("")
    bought = ses.purchased.fillna("")
    slot_click = np.zeros(10)
    slot_buy = np.zeros(10)
    for s, c, p in zip(shown, clicked, bought):
        if c:
            k = s.index(c) if c in s else -1
            if 0 <= k < 10:
                slot_click[k] += 1
                if p:
                    slot_buy[k] += 1
    # the sample keeps every clicked session + 1 in 5 of the rest -> reweight to the true rate
    n_click = (clicked != "").sum()
    n_true = n_click + (len(ses) - n_click) * 5
    has_rv = ses.rv.fillna("").str.len() > 0
    page = []
    for pg, g in ses.groupby("page"):
        nc = (g.clicked.fillna("") != "").sum()
        nt = nc + (len(g) - nc) * 5
        nb = (g.purchased.fillna("") != "").sum()
        page.append({"page": pg, "ctr": round(nc / nt * 100, 2), "cvr": round(nb / nt * 100, 2),
                     "share": round(nt / n_true * 100, 1)})
    hours = pd.to_datetime(ses.ts_ms, unit="ms", utc=True).dt.hour
    w = np.where(clicked != "", 1, 5)
    traffic = np.bincount(hours, weights=w, minlength=24)
    prop = ses.prop.str.split().explode().astype(float)

    # ---- model
    names = model["names"]

    def w_of(z: list[float], n: list[float]) -> list[float]:
        a, beta, l1, l2 = 0.4, 1.0, 0.0, 1.0
        out = []
        for zi, ni in zip(z, n):
            out.append(0.0 if abs(zi) <= l1 else -(zi - np.sign(zi) * l1) / ((beta + np.sqrt(ni)) / a + l2))
        return out

    wc, wb = w_of(model["click_z"], model["click_n"]), w_of(model["buy_z"], model["buy_n"])
    weights = sorted(
        [{"f": f, "click": round(c, 3), "buy": round(bb, 3)} for f, c, bb in zip(names, wc, wb)
         if f not in ("bias",)],
        key=lambda d: -abs(d["click"]) - abs(d["buy"]),
    )

    meta = json.loads((ROOT / "data/live/meta.json").read_text())
    dry = [json.loads(m[1]) for m in re.finditer(r"(\{.*\})", (ROOT / "logs/dry.log").read_text())]

    stats = {
        "catalog": {"n": len(cat), "cats": cat.category.nunique(), "subs": cat.subcategory.nunique(),
                    "brands": cat.brand.nunique(), "by_cat": by_cat,
                    "updates": upd.change.value_counts().to_dict()},
        "lorenz": lorenz,
        "regions": [[r, round(float(v), 4)] for r, v in ses.region.value_counts(normalize=True).items()],
        "top10_share": round(float(share[int(len(b) * 0.1) - 1]) * 100, 1),
        "sessions": {
            "n_sample": len(ses), "n_true": int(n_true),
            "ctr": round(n_click / n_true * 100, 2),
            "cvr": round((bought != "").sum() / n_true * 100, 2),
            "rv_share": round(float(has_rv.mean()) * 100, 1),
            "slot_click": (slot_click / slot_click.sum() * 100).round(2).tolist(),
            "slot_buy": (slot_buy / max(slot_buy.sum(), 1) * 100).round(2).tolist(),
            "page": page,
            "traffic": (traffic / traffic.sum() * 100).round(2).tolist(),
            "prop_box": [round(float(v), 4) for v in np.percentile(prop, [5, 25, 50, 75, 95])],
        },
        "exam_naive": [round(x, 3) for x in meta["exam_naive"]],
        "exam": [round(x, 3) for x in model["exam"]],
        "metrics": model["extra"]["metrics"],
        "weights": weights,
        "builds": {"train": build_log(ROOT / "logs/build_train.log"),
                   "live": build_log(ROOT / "logs/build_live.log") + build_log(ROOT / "logs/build_live2.log")},
        "dry": dry,
        # measured on the live stream during the practice round (team notes, not in the snapshots)
        "stream": {"requests_per_s": 3, "events_per_s": 50, "catalog_updates_per_s": 50, "new_user_share": 30,
                   "baseline_live_cvr": 0.68, "baseline_live_cold_cvr": 1.17},
        "users": int(re.search(r"(\d+) users", (ROOT / "logs/dry.log").read_text())[1]),
        "trained_until": model["extra"]["trained_until"],
        "config": {
            "quotas": {f"{'rv' if k[0] else 'no_rv'}|{'hist' if k[1] else 'no_hist'}": v for k, v in C.QUOTAS.items()},
            "hl_item_h": [h / 3600 for h in C.HL_ITEM], "hl_model_online_h": C.HL_MODEL_ONLINE / 3600,
            "hl_model_offline_d": C.HL_MODEL_OFFLINE / 86400, "explore_eps": C.EXPLORE_EPS,
            "explore_slots": [k + 1 for k in C.EXPLORE_SLOTS], "n_cand": C.N_CAND, "ips_cap": C.IPS_CAP,
            "prior_hours": C.PRIOR_HOURS, "poll_s": C.POLL_S, "bought_prior": C.BOUGHT_PRIOR_W,
            "train": [C.TRAIN_START, C.VALID_START, C.HISTORY_END], "n_features": len(model["names"]),
        },
    }
    OUT.write_text(json.dumps(stats, indent=1, ensure_ascii=False))

    # catalogue sample for the live simulation: real products (category, subcategory, brand, price,
    # 15-day purchases), 60 per category = the 30 best sellers + 30 at random
    pop = items.set_index("p").b2
    cat = cat[cat.is_active].assign(pop=lambda d: d.p.map(pop).fillna(0.0))
    rng = np.random.default_rng(42)
    sample = []
    for _, g in cat.groupby("category"):
        top = g.nlargest(30, "pop")
        rest = g.drop(top.index)
        sample.append(pd.concat([top, rest.iloc[rng.choice(len(rest), 30, replace=False)]]))
    s = pd.concat(sample)
    s = s.assign(pop=(s["pop"] / s["pop"].max() * 3).round(3))
    SAMPLE.write_text(json.dumps([
        {"id": r.p, "name": f"{r.brand} {r.subcategory}", "cat": r.category, "sub": f"{r.category}/{r.subcategory}",
         "brand": r.brand, "price": round(float(r.price_eur), 2), "pop": float(r.pop)} for r in s.itertuples()
    ], ensure_ascii=False))


if __name__ == "__main__":
    main()
