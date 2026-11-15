"""Ranker v7: the v5 ranker, de-biased with the log's propensities and taught to rank new products.

    python -m twin.train_v7                    # matrix_hybrid.npz (60 variables) -> artifacts/model_v7
    python -m twin.train_v7 --feedback         # + our own feedback (artifacts/declog + team_NN.feedback), weight 3
    options: --matrix FILE (default matrix_hybrid.npz, else matrix_v3.npz) · --out NAME (default model_v7)
             --ips-cap 20 · --ips-power 1.0 (0 = no propensity correction) · --cold-share 0.2 (0 = no copies)

Needs, in artifacts/train/: the feature matrix that `python -m twin.hybrid` (or `twin.train`) already
built, and the sessions.parquet it was built from. Nothing is featurised again: it takes minutes.

What changes against v5, and why
1. PROPENSITY (inverse propensity weighting). The old carousel showed best sellers far more often
   than the rest, so most training rows are popular products and the ranker is precise on them and
   vague on everything else. `propensity` says how likely the old system was to put each product in
   each slot. Weighting every row by 1/propensity (capped at --ips-cap, normalised per slot) makes the
   training set look as if every product had been shown equally often: the model spends its capacity
   on the products we actually show now (long tail, new products), not on the ones the old system
   pushed. The cap keeps a few rare rows from dominating (variance). --ips-power 0.5 is a softer version.
2. NEW PRODUCTS (item cold-start copies). In the log a product with no views or sales was a dead
   product, so the model learnt "no history = do not show". During the final, new products arrive
   with no history and that is exactly the cold-start score. A share of the training rows is copied
   with the product's history erased (no views, sales, carousel stats, co-visits, past interactions),
   keeping its label, category, price, text similarity and the shopper's session and profile. The
   model learns to rank a product it has never seen from what it is and who is looking.
3. OUR FEEDBACK (optional, --feedback). Rows the live loop logged (features at the moment we served
   them) joined with what the simulator did with our lists: the closest data to the hidden model.

Validation (last 3 days of the log, never trained on) compares v7 with the current model_v5 on the
same rows: all shoppers, unknown shoppers, and "new product": the bought product's history erased.
"""
from __future__ import annotations

import json
import shutil
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from twin.build import VALID_START
from twin.rt import ARTIFACTS
from twin.state import C0, CATEGORICAL, FEATURES, NF, to_ms
from twin.train import ANON_SHARE, KEEP_NO_CLICK, mrr, top1

EXTRA = ["content_session", "content_user", "collab_session", "collab_user"]
P_CLICK = {"num_leaves": 63, "min_data_in_leaf": 300}
P_BUY = {"num_leaves": 15, "min_data_in_leaf": 100, "learning_rate": 0.03}
SLOT = C0 + 3
BOUGHT = FEATURES.index("u_p_buys")

# Columns that describe the product's own history, and the value each takes for a product that has none
# (exactly what State.refresh / State.featurize compute when views, sales and co-visits are zero).
I = {n: FEATURES.index(n) for n in FEATURES}
ZERO_HISTORY = {"i_views_pd": 0.0, "i_purch_pd": 0.0, "i_trend7": 1.0, "i_age": float(np.log1p(7.0)),
                "i_car_lift": 1.0, "i_car_evidence": 0.0, "i_seg_pop": 0.0, "i_seg_lift": 1.0,
                "s_vv_sum": 0.0, "s_vv_last": 0.0, "s_vv_max": 0.0, "s_vp_sum": 0.0, "s_vp_last": 0.0,
                "u_bb_sum": 0.0, "u_bb_max": 0.0, "u_p_views": 0.0, "u_p_carts": 0.0, "u_p_view_share": 0.0,
                "u_p_buys": 0.0, "u_p_buy_days": np.nan}
GLOBAL_PRIORS = ("i_conv", "i_cart_rate", "i_car_pconv")   # a product with no history gets the global rate


def session_order(s: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The sessions and their order exactly as twin.train.build_matrix put them in the matrix."""
    rng = np.random.default_rng(7)
    clicked = (s.clicked != "").to_numpy()
    valid = (s.ts >= to_ms(VALID_START)).to_numpy()
    keep = clicked | valid | (rng.random(len(s)) < KEEP_NO_CLICK)
    s = s[keep].reset_index(drop=True)
    n = len(s)
    is_valid = (s.ts >= to_ms(VALID_START)).to_numpy()
    again = np.flatnonzero(is_valid | (rng.random(n) < ANON_SHARE))
    return s, np.concatenate([np.arange(n), again])


def propensities(m: dict, sessions_file) -> np.ndarray:
    """Per-row propensity of the old system, aligned with the matrix rows.

    Exact path: the matrix was built from this same sessions.parquet, so the sessions are
    re-selected in the same order. If the file differs a little (downloaded at another moment),
    every matrix session is matched instead by its timestamp, and the match is checked with the
    slot of its click; sessions that cannot be matched keep weight 1 (no correction)."""
    s_all = pd.read_parquet(sessions_file)
    s, order = session_order(s_all)
    ts_rows = m["ts"]
    if len(order) * 10 == len(ts_rows) and np.array_equal(np.repeat(s.ts.to_numpy()[order], 10), ts_rows):
        P = _prop_matrix(s)
        return P[order].reshape(-1)

    print("sessions.parquet is not the exact file the matrix was built from: matching sessions by time", flush=True)
    n = len(ts_rows) // 10
    t_m = ts_rows.reshape(n, 10)[:, 0]
    clicks = m["click"].reshape(n, 10)
    has_click = clicks.any(1)
    click_slot = np.where(has_click, clicks.argmax(1), -1)
    by_ts = s_all.drop_duplicates("ts", keep=False).set_index("ts")     # ambiguous timestamps are not used
    hit = pd.Index(by_ts.index).get_indexer(t_m)
    found = hit >= 0
    if found.mean() < 0.95:
        raise SystemExit(f"only {found.mean():.1%} of the matrix sessions are in this sessions.parquet: "
                         "use the sessions.parquet the matrix was built from")
    P = np.full((n, 10), np.nan)
    P[found] = _prop_matrix(by_ts.iloc[hit[found]].reset_index())
    # check: the clicked product's slot in the file must be the clicked row in the matrix
    sub = by_ts.iloc[hit[found & has_click]]
    file_slot = np.array([r.shown.split(" ").index(r.clicked) if r.clicked in r.shown.split(" ") else -2
                          for r in sub.itertuples()])
    agree = float(np.mean(file_slot == click_slot[found & has_click])) if len(file_slot) else 0.0
    print(f"  matched {found.mean():.1%} of the matrix sessions; click slot agrees in {agree:.1%} of the matched clicked ones",
          flush=True)
    if found.mean() < 0.95 or agree < 0.98:
        raise SystemExit("too many sessions could not be matched: use the sessions.parquet the matrix was built from")
    return P.reshape(-1)


def _prop_matrix(s: pd.DataFrame) -> np.ndarray:
    P = np.array([[float(x) for x in p.split(" ")] if p else [np.nan] * 10 for p in s.prop], np.float64)
    if P.ndim != 2 or P.shape[1] != 10:
        raise SystemExit("unexpected propensity format")
    return P


def ips_weights(prop: np.ndarray, slot: np.ndarray, rows: np.ndarray, cap: float, power: float) -> np.ndarray:
    """1/propensity, capped, raised to `power`, normalised so each slot keeps mean weight 1."""
    p = np.where(np.isfinite(prop) & (prop > 0), prop, np.nan)
    w = np.minimum(1.0 / np.nan_to_num(p, nan=1.0), cap) ** power
    w[~np.isfinite(p)] = 1.0
    for k in range(1, 11):
        r = rows & (slot == k)
        if r.any():
            w[slot == k] /= w[r].mean()
    return w.astype(np.float32)


def priors_from(X: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    """Global rates for the cold copies: the products with the least evidence in the matrix are the
    closest thing to 'no history', and their smoothed rates converge to the global prior."""
    ev = X[rows, I["i_views_pd"]]
    low = rows.copy()
    low[rows] = ev <= np.quantile(ev, 0.02)
    return {c: float(np.nanmedian(X[low, I[c]])) for c in GLOBAL_PRIORS}


def erase_history(X: np.ndarray, priors: dict[str, float]) -> np.ndarray:
    Z = X.copy()
    for c, v in ZERO_HISTORY.items():
        Z[:, I[c]] = v
    for c, v in priors.items():
        Z[:, I[c]] = v
    return Z


def fit(X, y, w, names, params, valid=None, rounds=None):
    base = {"objective": "binary", "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 6, "max_cat_to_onehot": 8,
            "cat_smooth": 50, "seed": 7, **params}
    d = lgb.Dataset(X, y, weight=w, feature_name=names, categorical_feature=CATEGORICAL, free_raw_data=False)
    if rounds:
        return lgb.train(base, d, num_boost_round=rounds)
    Xv, yv, wv = valid
    dv = lgb.Dataset(Xv, yv, weight=wv, reference=d)
    return lgb.train(base, d, num_boost_round=3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    train_dir = ARTIFACTS / "train"
    default = train_dir / "matrix_hybrid.npz"
    mfile = train_dir / _arg("--matrix", "matrix_hybrid.npz" if default.exists() else "matrix_v3.npz")
    out = ARTIFACTS / _arg("--out", "model_v7")
    cap, power, cold_share = float(_arg("--ips-cap", "20")), float(_arg("--ips-power", "1")), float(_arg("--cold-share", "0.2"))
    if out.name in ("model", "model_v5"):
        raise SystemExit(f"refusing to overwrite artifacts/{out.name}: choose another --out")
    if not mfile.exists():
        raise SystemExit(f"{mfile} not found: it is built by `python -m twin.hybrid` (60 variables) or `python -m twin.train` (56)")

    m = dict(np.load(mfile))
    X, click, buy, w, ts, anon = m["X"], m["click"], m["buy"], m["w"], m["ts"], m["anon"]
    names = FEATURES + EXTRA if X.shape[1] == NF + len(EXTRA) else FEATURES
    if X.shape[1] != len(names):
        raise SystemExit(f"{mfile.name} has {X.shape[1]} columns; expected {NF} or {NF + len(EXTRA)}")
    print(f"{mfile.name}: {len(X):,} rows, {X.shape[1]} variables", flush=True)

    bought = X[:, BOUGHT] > 0                      # the strict rule stays: never an already-bought product
    click, buy = np.where(bought, 0, click).astype(np.int8), np.where(bought, 0, buy).astype(np.int8)
    late = ts >= to_ms(VALID_START)
    tr, va = ~late & ~bought, late & ~anon & ~bought
    slot = X[:, SLOT].astype(int)

    # 1. propensity
    prop = propensities(m, train_dir / "sessions.parquet")
    wi = ips_weights(prop, slot, tr, cap, power) if power > 0 else np.ones(len(X), np.float32)
    p_ok = prop[tr][np.isfinite(prop[tr])]
    print(f"propensity: median {np.median(p_ok):.3f}, 1% {np.quantile(p_ok, 0.01):.4f}; weights capped at {cap:g} "
          f"(power {power:g}): {100 * (np.minimum(1 / np.maximum(p_ok, 1e-9), 1e9) > cap).mean():.1f}% of rows at the cap", flush=True)
    w_train = w * wi

    # 2. new-product copies of a share of the training rows (positives at the same rate as negatives)
    priors = priors_from(X, tr)
    rng = np.random.default_rng(11)
    cold_rows = np.flatnonzero(tr & (rng.random(len(X)) < cold_share)) if cold_share > 0 else np.array([], int)
    Xc = erase_history(X[cold_rows], priors)
    print(f"new-product copies: {len(cold_rows):,} rows ({int(click[cold_rows].sum()):,} clicks); priors {priors}", flush=True)

    # 3. our own feedback
    Xf = np.zeros((0, X.shape[1]), np.float32)
    cf = bf = np.zeros(0, np.int8)
    if "--feedback" in sys.argv:
        from twin.train_feedback import feedback_rows
        fb = feedback_rows(0.0)
        Xf = fb["X"].astype(np.float32)
        if Xf.shape[1] < X.shape[1]:          # the decision log keeps the 56 base variables
            Xf = np.hstack([Xf, np.full((len(Xf), X.shape[1] - Xf.shape[1]), np.nan, np.float32)])
        cf, bf = fb["click"], fb["buy"]
        print(f"feedback: {len(Xf):,} rows ({int(cf.sum()):,} clicks, {int(bf.sum()):,} purchases), weight 3", flush=True)

    XT = np.concatenate([X[tr], Xc, Xf])
    cT = np.concatenate([click[tr], click[cold_rows], cf])
    bT = np.concatenate([buy[tr], buy[cold_rows], bf])
    wT = np.concatenate([w_train[tr], 0.5 * w_train[cold_rows], np.full(len(Xf), 3 * float(np.median(w[tr])), np.float32)])

    print("training (validation: last 3 days of the log, never trained on)...", flush=True)
    cm = fit(XT, cT, wT, names, P_CLICK, valid=(X[va], click[va], w[va]))
    c = cT == 1
    bm = fit(XT[c], bT[c], wT[c], names, P_BUY, valid=(X[va & (click == 1)], buy[va & (click == 1)], w[va & (click == 1)]))

    # ---- validation: v7 against the current model on the same rows -------------------------------
    def models(path):
        if not (path / "click.txt").exists():
            return None
        return lgb.Booster(model_file=str(path / "click.txt")), lgb.Booster(model_file=str(path / "buy.txt"))

    base = models(ARTIFACTS / ("model_v5" if len(names) > NF else "model"))

    def score(pair, Xr):
        flat = Xr.copy()
        flat[:, SLOT] = 1
        return pair[0].predict(flat) * pair[1].predict(flat)

    va_all, cold_all = late & ~anon, late & anon
    # "new product": in each validation carousel with a purchase, the bought product's history is erased
    sess_buy = buy.reshape(-1, 10).any(1) & va_all.reshape(-1, 10).all(1)
    rows_nb = np.flatnonzero(np.repeat(sess_buy, 10))
    Xnb = X[rows_nb].copy()
    target = buy[rows_nb] == 1
    Xnb[target] = erase_history(Xnb[target], priors)
    sets = {"all shoppers": (X[va_all], buy[va_all], click[va_all]),
            "unknown shoppers": (X[cold_all], buy[cold_all], click[cold_all]),
            "bought product is new": (Xnb, buy[rows_nb], click[rows_nb])}
    metrics: dict = {"matrix": mfile.name, "ips_cap": cap, "ips_power": power, "cold_share": cold_share,
                     "feedback_rows": int(len(Xf)), "click_rounds": cm.best_iteration, "buy_rounds": bm.best_iteration}
    print(f"\n{'':24s} {'v7 top1 bought':>15s} {'base':>8s} {'v7 mrr':>8s} {'base':>8s}")
    for name, (Xs, b, cl) in sets.items():
        s7 = score((cm, bm), Xs)
        row = {"top1_bought": top1(s7, b), "mrr_bought": mrr(s7, b)}
        if base:
            sb = score(base, Xs)
            row.update(base_top1_bought=top1(sb, b), base_mrr_bought=mrr(sb, b))
        metrics[name] = row
        print(f"{name:24s} {row['top1_bought']:15.4f} {row.get('base_top1_bought', float('nan')):8.4f} "
              f"{row['mrr_bought']:8.4f} {row.get('base_mrr_bought', float('nan')):8.4f}")
    pv = cm.predict(X[va]) * bm.predict(X[va])
    metrics["purchase_auc"] = float(roc_auc_score(buy[va], pv, sample_weight=w[va]))
    wv_ips = ips_weights(prop, slot, va, cap, 1.0)[va] * w[va]
    metrics["purchase_auc_ips"] = float(roc_auc_score(buy[va], pv, sample_weight=wv_ips))
    if base:
        pb = base[0].predict(X[va]) * base[1].predict(X[va])
        metrics["base_purchase_auc"] = float(roc_auc_score(buy[va], pb, sample_weight=w[va]))
        metrics["base_purchase_auc_ips"] = float(roc_auc_score(buy[va], pb, sample_weight=wv_ips))
    print(f"purchase AUC: v7 {metrics['purchase_auc']:.4f} (de-biased {metrics['purchase_auc_ips']:.4f}) | base "
          f"{metrics.get('base_purchase_auc', float('nan')):.4f} (de-biased {metrics.get('base_purchase_auc_ips', float('nan')):.4f})")

    # ---- live model: every day of the log + copies + feedback, same rounds + 10 % ------------------
    every = ~bought
    wi_all = ips_weights(prop, slot, every, cap, power) if power > 0 else np.ones(len(X), np.float32)
    cold_all_rows = np.flatnonzero(every & (rng.random(len(X)) < cold_share)) if cold_share > 0 else np.array([], int)
    XT = np.concatenate([X[every], erase_history(X[cold_all_rows], priors), Xf])
    cT = np.concatenate([click[every], click[cold_all_rows], cf])
    bT = np.concatenate([buy[every], buy[cold_all_rows], bf])
    wT = np.concatenate([(w * wi_all)[every], 0.5 * (w * wi_all)[cold_all_rows], np.full(len(Xf), 3 * float(np.median(w[every])), np.float32)])
    cm = fit(XT, cT, wT, names, P_CLICK, rounds=max(int((metrics["click_rounds"] or 100) * 1.1), 20))
    c = cT == 1
    bm = fit(XT[c], bT[c], wT[c], names, P_BUY, rounds=max(int((metrics["buy_rounds"] or 100) * 1.1), 20))

    out.mkdir(parents=True, exist_ok=True)
    cm.save_model(str(out / "click.txt"))
    bm.save_model(str(out / "buy.txt"))
    if len(names) > NF:
        emb = ARTIFACTS / "model_v5" / "emb.joblib"
        if not emb.exists():
            raise SystemExit("a 60-variable model needs artifacts/model_v5/emb.joblib (python -m twin.hybrid_live)")
        shutil.copy(emb, out / "emb.joblib")
    g = cm.feature_importance("gain")
    (out / "meta.json").write_text(json.dumps({"features": names, "metrics": metrics,
                                               "importance": {n: round(float(v / g.sum()), 4) for n, v in zip(names, g)}}, indent=1))
    print(f"saved to {out}. To try it live, add for example 'v7+m2-intent+m3-live: 50' to config/twin_arms.txt")


if __name__ == "__main__":
    main()
