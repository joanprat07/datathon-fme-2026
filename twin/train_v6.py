"""V6: the V5 ranker (60 variables) retrained with history PLUS the simulator's feedback on our own lists.

    python -m twin.train_v6 features     # 1. decision log + feedback + shop.requests -> artifacts/train/feedback_v6.npz
    python -m twin.train_v6 train        # 2. matrix_hybrid.npz + feedback_v6.npz -> artifacts/model_v6 (validated, refitted)
    options for `train`: --fb-weight 3 · --holdout-hours 1 · --history-share 1.0 · --out model_v6

Why two phases: phase 1 needs the live State (user profiles) and the embeddings in memory, phase 2
needs the 1.1 GB training matrix. This laptop cannot hold both at once.

The decision log keeps only the 56 base variables of every carousel we served. The four hybrid
variables of V5 (twin/hybrid_live.py) are recomputed here from the request's recent views
(shop.requests) and the shopper's profile in the latest live snapshot. That profile is a snapshot,
not the exact profile at the moment of the request: up to ~1 h stale for the latest carousels, and
up to ~1 h "in the future" for the earliest ones. The session part (recent views) is exact.

Validation is the last hour of feedback, never used for training: V6 and the current V5 are scored
on the same rows, at the slot each product was really shown in. No ranking metric: the order of
these carousels was chosen by our own models, so a "was the clicked one first" measure would be
biased. The live A/B decides.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time

import numpy as np

from twin.rt import ARTIFACTS, TEAM, q, q_csv
from twin.state import C0, CATEGORICAL, FEATURES, NF

EXTRA = ["content_session", "content_user", "collab_session", "collab_user"]
NAMES = FEATURES + EXTRA
BOUGHT = FEATURES.index("u_p_buys")
FB_FILE = ARTIFACTS / "train" / "feedback_v6.npz"
HISTORY = ARTIFACTS / "train" / "matrix_hybrid.npz"
V5 = ARTIFACTS / "model_v5"
P_CLICK = {"num_leaves": 63, "min_data_in_leaf": 300}
P_BUY = {"num_leaves": 15, "min_data_in_leaf": 100, "learning_rate": 0.03}

FEEDBACK = """
SELECT request_id::String AS rid, served_by::String AS served_by, clicked::String AS clicked, purchased::String AS purchased
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) > fromUnixTimestamp64Milli({since})"""
REQUESTS = """
SELECT request_id::String AS rid, user_id::String AS u,
       arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv
FROM shop.requests
WHERE ts::DateTime64(3) BETWEEN fromUnixTimestamp64Milli({lo}) AND fromUnixTimestamp64Milli({hi})"""


# ----------------------------------------------------------------------------------------- phase 1
def features() -> None:
    from twin import declog
    from twin.hybrid_live import EMB_FILE, Embeddings
    from twin.state import State

    log = declog.load()
    rid, t, pids, X = log["rid"], log["t"], log["pids"], log["X"]
    print(f"decision log: {len(rid):,} carousels, {X.shape[-1]} variables, "
          f"{np.datetime64(int(t.min()), 'ms')} .. {np.datetime64(int(t.max()), 'ms')} UTC")
    if X.shape[-1] != NF:
        raise SystemExit("the decision log does not have the 56 base variables")

    fb = q_csv(FEEDBACK.format(team=TEAM, since=int(t.min()) - 1000), timeout=240, retries=2, dtype=str, keep_default_na=False)
    fb = fb[fb.served_by == "team"].drop_duplicates("rid").set_index("rid")
    req = q(REQUESTS.format(lo=int(t.min()) - 5000, hi=int(t.max()) + 5000), timeout=240, retries=2)
    req = req.drop_duplicates("rid").set_index("rid")
    keep = np.isin(rid, fb.index.to_numpy()) & np.isin(rid, req.index.to_numpy())
    rid, t, pids, X, arm = rid[keep], t[keep], pids[keep], X[keep], log["arm"][keep]
    clicked, bought = fb.clicked.reindex(rid).to_numpy(), fb.purchased.reindex(rid).to_numpy()
    print(f"with feedback served by us and a request row: {len(rid):,} carousels "
          f"({(clicked != '').sum():,} clicks, {(bought != '').sum():,} purchases)")

    st = State("live")
    emb = Embeddings(V5 / EMB_FILE)
    users, rvs = req.u.reindex(rid).to_numpy(), req.rv.reindex(rid).to_numpy()
    H = np.zeros((len(rid), 10, len(EXTRA)), np.float32)
    ok = np.ones(len(rid), bool)
    t0 = time.time()
    for k in range(len(rid)):
        cand = np.array([st.idx.get(p, -1) for p in pids[k]])
        if (cand < 0).any():
            ok[k] = False          # a product the snapshot does not know: cannot recompute its variables
            continue
        rv = [st.idx[p] for p in (rvs[k] or []) if p in st.idx]
        ui = st.uidx.get(str(users[k]), -1)
        H[k] = emb.features(st, rv, ui, cand)
        if k % 5000 == 0 and k:
            print(f"  {k:,} carousels, {time.time() - t0:.0f}s", flush=True)
    print(f"hybrid variables recomputed for {ok.sum():,} carousels ({(~ok).sum()} skipped) in {time.time() - t0:.0f}s")

    n = int(ok.sum())
    X60 = np.concatenate([X[ok], H[ok]], axis=2).reshape(n * 10, NF + len(EXTRA))
    out = {"X": X60, "click": (pids[ok] == clicked[ok][:, None]).reshape(-1).astype(np.int8),
           "buy": (pids[ok] == bought[ok][:, None]).reshape(-1).astype(np.int8),
           "t": np.repeat(t[ok], 10), "arm": np.repeat(arm[ok], 10), "rid": np.repeat(rid[ok], 10)}
    np.savez(FB_FILE, **out)
    print(f"saved {n * 10:,} rows x {X60.shape[1]} variables to {FB_FILE}")


# ----------------------------------------------------------------------------------------- phase 2
def fit(X, y, w, params: dict, valid=None, rounds: int | None = None):
    import lightgbm as lgb
    base = {"objective": "binary", "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 6, "max_cat_to_onehot": 8,
            "cat_smooth": 50, "seed": 7, **params}
    d = lgb.Dataset(X, y, weight=w, feature_name=NAMES, categorical_feature=CATEGORICAL, free_raw_data=True)
    if rounds:
        return lgb.train(base, d, num_boost_round=rounds)
    Xv, yv, wv = valid
    dv = lgb.Dataset(Xv, yv, weight=wv, reference=d)
    return lgb.train(base, d, num_boost_round=3000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])


def score(name: str, click_model, buy_model, X, click, buy) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score
    pc, pb = click_model.predict(X), buy_model.predict(X)
    out = {"click_logloss": float(log_loss(click, pc, labels=[0, 1])),
           "click_calibration": float(pc.mean() / max(click.mean(), 1e-9)),
           "purchase_calibration": float((pc * pb).mean() / max(buy.mean(), 1e-9))}
    for key, y, p in (("click_auc", click, pc), ("purchase_auc", buy, pc * pb)):
        out[key] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    print(f"  {name:8s} " + " | ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)
    return out


def train(out: str, fb_weight: float, holdout_hours: float, share: float) -> None:
    import lightgbm as lgb
    if out in ("model", "model_v5"):
        raise SystemExit(f"refusing to overwrite artifacts/{out}")
    f = np.load(FB_FILE, allow_pickle=False)
    Xf, cf, bf, tf = f["X"], f["click"], f["buy"], f["t"]
    own = ~(Xf[:, BOUGHT] > 0)                       # rows of already-bought products (rebuy arms) left out, as in training
    late = tf >= tf.max() - holdout_hours * 3_600_000
    tr, va = own & ~late, own & late
    print(f"feedback: {own.sum():,} rows ({cf[own].sum():,} clicks, {bf[own].sum():,} purchases) | "
          f"validation = last {holdout_hours:g} h: {va.sum():,} rows ({cf[va].sum():,} clicks, {bf[va].sum():,} purchases)")
    if cf[va].sum() < 20 or cf[tr].sum() < 100:
        raise SystemExit("too few clicks to train and validate: raise --holdout-hours or wait for more feedback")

    m = np.load(HISTORY)
    keep = ~(m["X"][:, BOUGHT] > 0)
    if share < 1:
        keep &= np.repeat(np.random.default_rng(7).random(len(keep) // 10) < share, 10)
    Xh, ch, bh, wh = m["X"][keep], m["click"][keep], m["buy"][keep], m["w"][keep]
    del m
    gc.collect()
    if Xh.shape[1] != len(NAMES):
        raise SystemExit("matrix_hybrid.npz does not have 60 columns")
    print(f"history: {len(Xh):,} rows ({ch.sum():,} clicks, {bh.sum():,} purchases)", flush=True)

    def stack(rows):
        X = np.concatenate([Xh, Xf[rows]])
        y_c, y_b = np.concatenate([ch, cf[rows]]), np.concatenate([bh, bf[rows]])
        w = np.concatenate([wh, np.full(int(rows.sum()), fb_weight, np.float32)])
        return X, y_c, y_b, w

    X, click, buy, w = stack(tr)
    Xv, cv, bv, wv = Xf[va], cf[va], bf[va], np.ones(int(va.sum()), np.float32)
    print(f"training rows {len(X):,} (feedback {tr.sum():,} x weight {fb_weight:g})", flush=True)
    t0 = time.time()
    click_model = fit(X, click, w, P_CLICK, valid=(Xv, cv, wv))
    c, cva = click == 1, cv == 1
    if cva.sum() >= 20 and 0 < bv[cva].sum() < cva.sum():
        buy_model = fit(X[c], buy[c], w[c], P_BUY, valid=(Xv[cva], bv[cva], wv[cva]))
    else:
        buy_model = fit(X[c], buy[c], w[c], P_BUY, rounds=lgb.Booster(model_file=str(V5 / "buy.txt")).num_trees())
    print(f"trained in {time.time() - t0:.0f}s: click rounds {click_model.best_iteration}, buy rounds {buy_model.best_iteration}")
    del X, click, buy, w
    gc.collect()

    print("validation on the last hour of our own feedback (same rows, slot as shown):")
    v5c, v5b = lgb.Booster(model_file=str(V5 / "click.txt")), lgb.Booster(model_file=str(V5 / "buy.txt"))
    metrics = {"current_v5": score("v5", v5c, v5b, Xv, cv, bv), "new_v6": score("v6", click_model, buy_model, Xv, cv, bv),
               "feedback_rows_train": int(tr.sum()), "feedback_rows_valid": int(va.sum()),
               "feedback_clicks": int(cf[own].sum()), "feedback_purchases": int(bf[own].sum()),
               "history_rows": int(len(Xh)), "fb_weight": fb_weight, "holdout_hours": holdout_hours,
               "click_rounds": click_model.best_iteration or click_model.num_trees(),
               "buy_rounds": buy_model.best_iteration or buy_model.num_trees(),
               "feedback_from_utc": str(np.datetime64(int(tf.min()), "ms")), "feedback_to_utc": str(np.datetime64(int(tf.max()), "ms"))}
    better = metrics["new_v6"]["click_logloss"] < metrics["current_v5"]["click_logloss"]
    print(f"click log-loss {'improves' if better else 'DOES NOT improve'}: "
          f"{metrics['current_v5']['click_logloss']:.5f} -> {metrics['new_v6']['click_logloss']:.5f}")

    # the live model learns from the validation hour too, with the rounds found above (+10 %)
    X, click, buy, w = stack(own)
    click_model = fit(X, click, w, P_CLICK, rounds=max(int(metrics["click_rounds"] * 1.1), 20))
    buy_model = fit(X[click == 1], buy[click == 1], w[click == 1], P_BUY, rounds=max(int(metrics["buy_rounds"] * 1.1), 20))
    metrics["refit_on_all"] = True

    d = ARTIFACTS / out
    d.mkdir(parents=True, exist_ok=True)
    click_model.save_model(str(d / "click.txt"))
    buy_model.save_model(str(d / "buy.txt"))
    shutil.copy(V5 / "emb.joblib", d / "emb.joblib")     # same embedding space as V5 and as the feedback rows
    g = click_model.feature_importance("gain")
    (d / "meta.json").write_text(json.dumps({"features": NAMES, "metrics": metrics, "based_on": "model_v5 + feedback",
                                             "importance": {n: round(float(v / g.sum()), 4) for n, v in zip(NAMES, g)}}, indent=1))
    print(f"saved to {d}")


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "train"
    if phase == "features":
        features()
    else:
        train(_arg("--out", "model_v6"), float(_arg("--fb-weight", "3")), float(_arg("--holdout-hours", "1")),
              float(_arg("--history-share", "1")))
