"""Train the ranker on the old carousel's log and validate it on the last 3 days of history.

    python -m twin.train            # featurize (cached), train, validate, save artifacts/model/
    python -m twin.train --refit    # same, then refit on train + validation for the live model
    python -m twin.train --keep-bought --out model_rebuy --refit
                                    # variant m1-rebuy: already-bought products stay in training, so the
                                    # ranker learns from the log what a past purchase means; saved apart

Two LightGBM models, multiplied at serving time:
    p(click | product, slot, shopper, session)   x   p(purchase | click, ...)
Features come from the `train` snapshot (events before 25 Oct); sessions are from 25 Oct on,
so nothing the model sees was computed from the future of the session it scores.
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from twin.build import VALID_START
from twin.rt import ARTIFACTS
from twin.state import C0, CATEGORICAL, DEVICES, FEATURES, PAGES, State, to_ms

KEEP_NO_CLICK = 0.3   # share of sessions without a click kept for training (weighted back)
ANON_SHARE = 0.25     # share of training sessions repeated with the shopper's profile hidden
MODEL_DIR = ARTIFACTS / "model"
CACHE = ARTIFACTS / "train" / "matrix.npz"
SLOT = C0 + 3
BOUGHT = FEATURES.index("u_p_buys")


def build_matrix(st: State) -> dict[str, np.ndarray]:
    s = pd.read_parquet(ARTIFACTS / "train" / "sessions.parquet")
    rng = np.random.default_rng(7)
    clicked = (s.clicked != "").to_numpy()
    valid = (s.ts >= to_ms(VALID_START)).to_numpy()
    keep = clicked | valid | (rng.random(len(s)) < KEEP_NO_CLICK)   # validation is kept whole
    s = s[keep].reset_index(drop=True)
    weight = np.where((s.clicked != "") | (s.ts >= to_ms(VALID_START)), 1.0, 1.0 / KEEP_NO_CLICK)
    n = len(s)
    # Cold start: some sessions are repeated as if we knew nothing about the shopper, so the
    # ranker also learns to rank from the session and the products alone. In training that is
    # ANON_SHARE of the sessions; in validation every session, to measure cold-start quality.
    is_valid = (s.ts >= to_ms(VALID_START)).to_numpy()
    again = np.flatnonzero(is_valid | (rng.random(n) < ANON_SHARE))
    order = np.concatenate([np.arange(n), again])
    anon = np.concatenate([np.zeros(n, bool), np.ones(len(again), bool)])
    m = len(order)
    print(f"{n:,} sessions + {len(again):,} anonymous copies -> {m * 10:,} rows", flush=True)

    rows = list(s.itertuples(index=False))
    X = np.zeros((m * 10, len(FEATURES)), np.float32)
    click, buy = np.zeros(m * 10, np.int8), np.zeros(m * 10, np.int8)
    slots = np.arange(1, 11)
    t0 = time.time()
    for k in range(m):
        r = rows[order[k]]
        shown = np.array([st.idx[p] for p in r.shown.split(" ")])
        rv = [st.idx[p] for p in r.rv.split(" ") if p in st.idx] if r.rv else []
        ui = -1 if anon[k] else st.uidx.get(r.user_id, -1)
        X[k * 10:k * 10 + 10] = st.featurize(r.ts, PAGES.get(r.page, -1), DEVICES.get(r.device, -1),
                                             st.region_id.get(r.region, -1), ui, rv, shown, slots)
        if r.clicked:
            j = k * 10 + int(np.flatnonzero(shown == st.idx[r.clicked])[0])
            click[j] = 1
            buy[j] = r.purchased != ""
        if k % 50_000 == 0 and k:
            print(f"  {k:,} sessions, {time.time() - t0:.0f}s", flush=True)
    out = {"X": X, "click": click, "buy": buy, "w": np.repeat(weight[order], 10).astype(np.float32),
           "ts": np.repeat(s.ts.to_numpy()[order], 10), "anon": np.repeat(anon, 10)}
    np.savez(CACHE, **out)
    return out


def fit(X, y, w, Xv, yv, wv, params: dict, rounds: int | None = None) -> lgb.Booster:
    base = {"objective": "binary", "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 8, "max_cat_to_onehot": 8,
            "cat_smooth": 50, **params}
    train = lgb.Dataset(X, y, weight=w, feature_name=FEATURES, categorical_feature=CATEGORICAL, free_raw_data=False)
    if rounds:
        return lgb.train(base, train, num_boost_round=rounds)
    valid = lgb.Dataset(Xv, yv, weight=wv, reference=train)
    return lgb.train(base, train, num_boost_round=3000, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])


def mrr(score: np.ndarray, click: np.ndarray) -> float:
    """Mean reciprocal rank of the clicked product among the 10 shown (sessions with a click)."""
    s, c = score.reshape(-1, 10), click.reshape(-1, 10)
    has = c.sum(1) > 0
    rank = (s[has] > (s[has] * c[has]).sum(1, keepdims=True)).sum(1) + 1
    return float((1 / rank).mean())


def top1(score: np.ndarray, click: np.ndarray) -> float:
    """How often the model's favourite among the 10 shown is the one that was clicked/bought."""
    s, c = score.reshape(-1, 10), click.reshape(-1, 10)
    has = c.sum(1) > 0
    return float(c[has][np.arange(has.sum()), s[has].argmax(1)].mean())


def main(refit: bool, keep_bought: bool = False, out: str = "model") -> None:
    model_dir = ARTIFACTS / out
    if CACHE.exists():
        m = dict(np.load(CACHE))
        print("loaded cached feature matrix", m["X"].shape)
    else:
        m = build_matrix(State("train"))
    X, click, buy, w, ts, anon = m["X"], m["click"], m["buy"], m["w"], m["ts"], m["anon"]
    late = ts >= to_ms(VALID_START)
    va_all, cold_all = late & ~anon, late & anon        # whole carousels, for the ranking metrics
    # Business rule: we never recommend a product the shopper already bought. Those rows are
    # left out of training and validation, so the ranker does not learn the repurchase pattern
    # of the old carousel's log; they also never win in the ranking metrics below.
    was_bought = X[:, BOUGHT] > 0
    # How much of what the old carousel sold is a repurchase? This is what the rule gives up.
    repeat = {"repurchase_share_of_purchases": float(buy[va_all & was_bought].sum() / max(buy[va_all].sum(), 1)),
              "repurchase_share_of_clicks": float(click[va_all & was_bought].sum() / max(click[va_all].sum(), 1)),
              "already_bought_share_of_rows": float(was_bought[va_all].mean())}
    print("already-bought products in validation: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in repeat.items()))
    # --keep-bought: nothing is left out; the ranker sees u_p_buys / u_p_buy_days and decides.
    bought = was_bought & (not keep_bought)
    click, buy = np.where(bought, 0, click).astype(np.int8), np.where(bought, 0, buy).astype(np.int8)
    tr, va, cold = ~late & ~bought, va_all & ~bought, cold_all & ~bought   # cold: shopper hidden
    print(f"left out {bought.sum():,} rows of already-bought products ({100 * bought.mean():.1f}%)")
    print(f"train rows {tr.sum():,} (clicks {click[tr].sum():,}, purchases {buy[tr].sum():,}) | "
          f"valid rows {va.sum():,} (clicks {click[va].sum():,}, purchases {buy[va].sum():,})")

    p_click = {"num_leaves": 63, "min_data_in_leaf": 300}
    p_buy = {"num_leaves": 15, "min_data_in_leaf": 100, "learning_rate": 0.03}
    click_model = fit(X[tr], click[tr], w[tr], X[va], click[va], w[va], p_click)
    ctr, cva = tr & (click == 1), va & (click == 1)
    buy_model = fit(X[ctr], buy[ctr], w[ctr], X[cva], buy[cva], w[cva], p_buy)

    # ---- validation on the last 3 days --------------------------------------------------
    def rank_score(rows: np.ndarray) -> np.ndarray:
        flat = X[rows].copy()
        flat[:, SLOT] = 1                           # same slot for everybody: pure ranking score
        return np.where(bought[rows], -1.0, click_model.predict(flat) * buy_model.predict(flat))

    Xv = X[va]
    pc, pb = click_model.predict(Xv), buy_model.predict(Xv)
    score, score_cold = rank_score(va_all), rank_score(cold_all)
    popularity = np.where(bought[va_all], -1.0, X[va_all][:, FEATURES.index("i_purch_pd")])
    slot_rate = pd.Series(click[tr]).groupby(X[tr][:, SLOT]).mean()   # position-only reference
    ref = pd.Series(Xv[:, SLOT]).map(slot_rate).to_numpy()
    metrics = {
        "click_auc": roc_auc_score(click[va], pc),
        "click_auc_position_only": roc_auc_score(click[va], ref),
        "click_logloss": log_loss(click[va], pc),
        "click_logloss_position_only": log_loss(click[va], ref),
        "buy_given_click_auc": roc_auc_score(buy[cva], buy_model.predict(X[cva])),
        "purchase_auc": roc_auc_score(buy[va], pc * pb),
        "mrr_clicked_model": mrr(score, click[va_all]),
        "mrr_clicked_model_cold": mrr(score_cold, click[cold_all]),
        "mrr_clicked_popularity": mrr(popularity, click[va_all]),
        "mrr_clicked_random": float(np.mean(1 / np.arange(1, 11))),
        "mrr_bought_model": mrr(score, buy[va_all]),
        "mrr_bought_model_cold": mrr(score_cold, buy[cold_all]),
        "mrr_bought_popularity": mrr(popularity, buy[va_all]),
        "top1_bought_model": top1(score, buy[va_all]),
        "top1_bought_model_cold": top1(score_cold, buy[cold_all]),
        "top1_bought_popularity": top1(popularity, buy[va_all]),
        "rows_left_out_already_bought": int(bought.sum()),
        "keep_bought": keep_bought, **repeat,
        "calibration_click": float(pc.mean() / click[va].mean()),
        "click_rounds": click_model.best_iteration, "buy_rounds": buy_model.best_iteration,
        "valid_from": VALID_START,
    }
    if keep_bought:   # the same ranker with the blocking rule applied on top: what the rule costs offline
        blocked = np.where(was_bought[va_all], -1.0, score)
        metrics["top1_bought_model_if_blocked"] = top1(blocked, buy[va_all])
        metrics["mrr_bought_model_if_blocked"] = mrr(blocked, buy[va_all])
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    gain = pd.Series(click_model.feature_importance("gain"), FEATURES).sort_values(ascending=False)
    print("top features (click model, share of gain):")
    print((gain / gain.sum()).head(20).round(3).to_string())
    gain_b = pd.Series(buy_model.feature_importance("gain"), FEATURES).sort_values(ascending=False)
    print("top features (purchase model):")
    print((gain_b / gain_b.sum()).head(10).round(3).to_string())

    if refit:  # the live model uses every day we have, with the number of rounds found above
        every = ~bought
        click_model = fit(X[every], click[every], w[every], None, None, None, p_click,
                          rounds=int(click_model.best_iteration * 1.1))
        buy_model = fit(X[click == 1], buy[click == 1], w[click == 1], None, None, None, p_buy,
                        rounds=int(buy_model.best_iteration * 1.1))
        metrics["refit_on_all_days"] = True

    model_dir.mkdir(parents=True, exist_ok=True)
    click_model.save_model(str(model_dir / "click.txt"))
    buy_model.save_model(str(model_dir / "buy.txt"))
    (model_dir / "meta.json").write_text(json.dumps({"features": FEATURES, "metrics": metrics,
                                                     "importance": (gain / gain.sum()).round(4).to_dict()}, indent=1))
    print(f"saved to {model_dir}")


if __name__ == "__main__":
    main("--refit" in sys.argv, "--keep-bought" in sys.argv,
         sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "model")
