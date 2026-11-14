"""Retrain the ranker with what the hidden shopper model did with OUR lists.

    python -m twin.train_feedback                  # history + our feedback -> artifacts/model_fb, validated
    python -m twin.train_feedback --refit          # same, then refit on everything for the live model
    python -m twin.train_feedback --no-history     # feedback only (if artifacts/train/matrix.npz is not on this machine)
    options: --out NAME (default model_fb) · --fb-weight 3 · --holdout-hours 1 · --history-share 1.0

Run it on the machine that runs the live loop: it reads artifacts/declog/ (the features of every
carousel we served, written by the loop) and joins them with team_NN.feedback (what was clicked
and bought). The old carousel's log taught the base model; this adds rows that are answers of the
same simulator to our own candidates and our own ordering, which the log never contained.

Validation is the most recent hour(s) of feedback, never used for training: the new model and the
current one (artifacts/model) are scored on the same rows. Put the result live as an arm
("fb: 20" in config/twin_arms.txt) and let the leaderboard decide; this script never touches
artifacts/model.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from twin import declog
from twin.recommend import Ranker
from twin.rt import ARTIFACTS, TEAM, q_csv
from twin.state import FEATURES
from twin.train import BOUGHT, fit

HISTORY = ARTIFACTS / "train" / "matrix.npz"
P_CLICK = {"num_leaves": 63, "min_data_in_leaf": 300}
P_BUY = {"num_leaves": 15, "min_data_in_leaf": 100, "learning_rate": 0.03}

FEEDBACK = """
SELECT request_id::String AS rid, served_by::String AS served_by, clicked::String AS clicked, purchased::String AS purchased
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) > fromUnixTimestamp64Milli({since})"""


def feedback_rows(holdout_hours: float) -> dict[str, np.ndarray]:
    """Decision log joined with feedback: one row per shown product, with click and purchase labels."""
    log = declog.load()
    fb = q_csv(FEEDBACK.format(team=TEAM, since=int(log["t"].min()) - 1000), timeout=240, retries=2,
               dtype=str, keep_default_na=False)
    fb = fb[fb.served_by == "team"].drop_duplicates("rid").set_index("rid")
    known = np.isin(log["rid"], fb.index.to_numpy())
    rid, t, pids, X = log["rid"][known], log["t"][known], log["pids"][known], log["X"][known]
    clicked, bought = fb.clicked.reindex(rid).to_numpy(), fb.purchased.reindex(rid).to_numpy()
    n = len(rid)
    print(f"decision log: {len(log['rid']):,} carousels, {n:,} with feedback served by us "
          f"({(clicked != '').sum():,} clicks, {(bought != '').sum():,} purchases)")
    if n < 500:
        raise SystemExit("too little feedback to learn from yet: let the live loop run longer")
    late = t >= t.max() - holdout_hours * 3_600_000
    return {"X": X.reshape(n * 10, -1), "click": (pids == clicked[:, None]).reshape(-1).astype(np.int8),
            "buy": (pids == bought[:, None]).reshape(-1).astype(np.int8), "late": np.repeat(late, 10),
            "arm": np.repeat(log["arm"][known], 10)}


def history_rows(share: float) -> dict[str, np.ndarray] | None:
    """The old carousel's log, exactly as twin.train uses it (already-bought rows left out)."""
    if not HISTORY.exists():
        return None
    m = np.load(HISTORY)
    X, click, buy, w = m["X"], m["click"], m["buy"], m["w"]
    keep = ~(X[:, BOUGHT] > 0)
    if share < 1:   # whole carousels, so the 10 rows of a session stay together
        keep &= np.repeat(np.random.default_rng(7).random(len(X) // 10) < share, 10)
    print(f"history: {keep.sum():,} rows ({click[keep].sum():,} clicks, {buy[keep].sum():,} purchases)")
    return {"X": X[keep], "click": click[keep], "buy": buy[keep], "w": w[keep]}


def score(name: str, click_model, buy_model, X, click, buy) -> dict[str, float]:
    """Predictions at the slot each product was really shown in. (No ranking metric here: the
    order of these carousels was chosen by the current model, so any 'was the clicked one ranked
    first' measure would favour it by construction.)"""
    pc, pb = click_model.predict(X), buy_model.predict(X)
    out = {"click_logloss": log_loss(click, pc, labels=[0, 1]), "click_calibration": float(pc.mean() / max(click.mean(), 1e-9)),
           "purchase_calibration": float((pc * pb).mean() / max(buy.mean(), 1e-9))}
    for key, y, p in (("click_auc", click, pc), ("purchase_auc", buy, pc * pb)):
        out[key] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    print(f"  {name:10s} " + " | ".join(f"{k} {v:.4f}" for k, v in out.items()))
    return out


def main(out: str, fb_weight: float, holdout_hours: float, refit: bool, use_history: bool, share: float) -> None:
    if out == "model":
        raise SystemExit("refusing to overwrite artifacts/model: choose another --out")
    fb = feedback_rows(holdout_hours)
    hist = history_rows(share) if use_history else None
    if use_history and hist is None:
        raise SystemExit(f"{HISTORY} is not on this machine: run `python -m twin.build train` and `python -m twin.train`, "
                         "or pass --no-history to learn from the feedback alone")
    tr, va = ~fb["late"], fb["late"]
    if fb["click"][va].sum() < 20 or fb["click"][tr].sum() < 100:
        raise SystemExit(f"too few clicks to train and validate (train {fb['click'][tr].sum()}, validation "
                         f"{fb['click'][va].sum()}): let the loop run longer or raise --holdout-hours")

    def stack(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, click, buy = fb["X"][rows], fb["click"][rows], fb["buy"][rows]
        w = np.full(len(X), fb_weight, np.float32)
        if hist is not None:
            X, click, buy = np.concatenate([hist["X"], X]), np.concatenate([hist["click"], click]), np.concatenate([hist["buy"], buy])
            w = np.concatenate([hist["w"], w])
        return X, click, buy, w

    X, click, buy, w = stack(tr)
    Xv, cv, bv = fb["X"][va], fb["click"][va], fb["buy"][va]
    wv = np.ones(len(Xv), np.float32)
    print(f"training rows {len(X):,} (feedback {tr.sum():,}, weight {fb_weight:g}) | validation: last {holdout_hours:g} h of feedback, {va.sum():,} rows")
    click_model = fit(X, click, w, Xv, cv, wv, P_CLICK)
    c, cva = click == 1, cv == 1
    if cva.sum() >= 20 and 0 < bv[cva].sum() < cva.sum():
        buy_model = fit(X[c], buy[c], w[c], Xv[cva], bv[cva], wv[cva], P_BUY)
    else:   # too few validation clicks to stop early on: keep the number of rounds of the current model
        buy_model = fit(X[c], buy[c], w[c], None, None, None, P_BUY, rounds=Ranker(ARTIFACTS / "model").buy.num_trees())

    print("validation on our own most recent feedback (same rows for both):")
    base = Ranker(ARTIFACTS / "model")
    metrics = {"current": score("current", base.click, base.buy, Xv, cv, bv),
               "new": score("new", click_model, buy_model, Xv, cv, bv),
               "feedback_rows_train": int(tr.sum()), "feedback_rows_valid": int(va.sum()),
               "feedback_clicks": int(fb["click"].sum()), "feedback_purchases": int(fb["buy"].sum()),
               "history_rows": 0 if hist is None else int(len(hist["X"])), "fb_weight": fb_weight,
               "click_rounds": click_model.best_iteration or click_model.num_trees(),
               "buy_rounds": buy_model.best_iteration or buy_model.num_trees()}
    better = metrics["new"]["click_logloss"] < metrics["current"]["click_logloss"]
    print(f"click log-loss {'improves' if better else 'DOES NOT improve'}: "
          f"{metrics['current']['click_logloss']:.5f} -> {metrics['new']['click_logloss']:.5f}")

    if refit:   # the live model also learns from the validation hour, with the rounds found above
        X, click, buy, w = stack(np.ones(len(fb["late"]), bool))
        click_model = fit(X, click, w, None, None, None, P_CLICK, rounds=max(int(metrics["click_rounds"] * 1.1), 20))
        buy_model = fit(X[click == 1], buy[click == 1], w[click == 1], None, None, None, P_BUY,
                        rounds=max(int(metrics["buy_rounds"] * 1.1), 20))
        metrics["refit_on_all"] = True

    model_dir = ARTIFACTS / out
    model_dir.mkdir(parents=True, exist_ok=True)
    click_model.save_model(str(model_dir / "click.txt"))
    buy_model.save_model(str(model_dir / "buy.txt"))
    (model_dir / "meta.json").write_text(json.dumps({"features": FEATURES, "metrics": metrics}, indent=1))
    print(f"saved to {model_dir}. To try it live, add the line 'fb: 20' to config/twin_arms.txt"
          + ("" if out == "model_fb" else f" (with model '{out}': see VARIANTS in twin/variants.py)"))


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    main(_arg("--out", "model_fb"), float(_arg("--fb-weight", "3")), float(_arg("--holdout-hours", "1")),
         "--refit" in sys.argv, "--no-history" not in sys.argv, float(_arg("--history-share", "1")))
