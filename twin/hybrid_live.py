"""The four hybrid variables of twin/hybrid.py, served live inside the Twin (version v5-hybrid).

    python -m twin.hybrid_live      # export the embeddings + train the 60-variable ranker -> artifacts/model_v5

The offline check (twin-v1 trained with and without these four variables on the same rows) gave
+2.5 points of first-pick-bought, +1.3 for unknown shoppers and +0.010 purchase AUC.

    content_session   text similarity (TF-IDF + SVD) between the product and the last 5 products viewed
    content_user      the same against the shopper's profile (views + 3 x carts)
    collab_session    collaborative similarity (SVD of shopper x product) against the last 5 views
    collab_user       the same against the shopper's profile

The transforms are the ones twin.hybrid fitted on the training snapshot (events before 25 Oct), so
training and serving see the same space. Products the training snapshot did not have (added during
the event) get their text vector from the same transform, and their collaborative vector from the
5 known products with the closest text: they are never left without features.

Serving: Ranker sees 60 features in the model's meta.json, loads emb.joblib from the same folder
and appends the four columns. A 56-variable model keeps working unchanged in the same process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import normalize

from twin.rt import ARTIFACTS

EXTRA = ["content_session", "content_user", "collab_session", "collab_user"]
EMB_FILE = "emb.joblib"
OUT = ARTIFACTS / "model_v5"


class Embeddings:
    """Content and collaborative vectors for every product of a live State, kept up to date."""

    def __init__(self, path: Path):
        d = joblib.load(path)                      # our own file, written by export() below
        self.pids, self.vectorizer, self.text_svd = d["pids"], d["vectorizer"], d["text_svd"]
        self.known_content = np.asarray(d["content"], np.float32)
        self.known_collab = np.asarray(d["collab"], np.float32)
        self.known = {p: i for i, p in enumerate(self.pids)}
        self.content = self.collab = None
        self.n = 0                                 # products of the State already covered

    def _extend(self, st) -> None:
        k = self.known_content.shape[1]
        if self.content is None or len(self.content) < st.cap:
            grow = lambda a: np.vstack([a, np.zeros((st.cap - len(a), k), np.float32)]) if a is not None else np.zeros((st.cap, k), np.float32)
            self.content, self.collab = grow(self.content), grow(self.collab)
        missing = []
        for i in range(self.n, st.P):
            j = self.known.get(st.pid[i])
            if j is None:
                missing.append(i)
            else:
                self.content[i], self.collab[i] = self.known_content[j], self.known_collab[j]
        if missing:
            texts = [" ".join(sorted(st.tokens[i])) + " " + st.sub_names[st.sub[i]] for i in missing]
            z = normalize(self.text_svd.transform(self.vectorizer.transform(texts))).astype(np.float32)
            self.content[missing] = z
            near = np.argsort(-(z @ self.known_content.T), axis=1)[:, :5]
            self.collab[missing] = normalize(self.known_collab[near].mean(1)).astype(np.float32)
        self.n = st.P

    def features(self, st, rv: list[int], ui: int, cand: np.ndarray) -> np.ndarray:
        """The four columns for candidates `cand`, exactly as twin.hybrid.HybridState computes them."""
        if st.P > self.n or self.content is None:
            self._extend(st)
        ix, v, c = st.user_products(ui)
        cols = []
        for E in (self.content, self.collab):
            session = E[rv[-5:]].mean(0) if rv else np.zeros(E.shape[1], np.float32)
            profile = np.average(E[ix], axis=0, weights=v + 3 * c + 1e-8) if len(ix) else np.zeros(E.shape[1], np.float32)
            session = session / max(float(np.linalg.norm(session)), 1e-8)
            profile = profile / max(float(np.linalg.norm(profile)), 1e-8)
            cols.extend([E[cand] @ session, E[cand] @ profile])
        return np.column_stack(cols).astype(np.float32)


def export(out: Path = OUT) -> None:
    """Keep from twin.hybrid's saved training state only what serving needs."""
    st = joblib.load(ARTIFACTS / "model_hybrid" / "state.joblib")   # written by `python -m twin.hybrid`
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pids": list(st.pid[:st.P]), "content": st.content[:st.P], "collab": st.collab[:st.P],
                 "vectorizer": st.vectorizer, "text_svd": st.text_svd}, out / EMB_FILE, compress=3)
    print(f"embeddings for {st.P} products saved to {out / EMB_FILE}")


def train(out: Path = OUT) -> None:
    """twin-v1's ranker (p(click) x p(purchase | click), already-bought rows left out) on 60 variables,
    validated on the last 3 days and refitted on every day, from twin.hybrid's feature matrix."""
    from sklearn.metrics import roc_auc_score

    from twin.build import VALID_START
    from twin.state import C0, CATEGORICAL, FEATURES, NF, to_ms
    from twin.train import mrr, top1

    m = np.load(ARTIFACTS / "train" / "matrix_hybrid.npz")
    X, click, buy, w, ts, anon = m["X"], m["click"], m["buy"], m["w"], m["ts"], m["anon"]
    if X.shape[1] != NF + len(EXTRA):
        raise SystemExit("matrix_hybrid.npz does not have 56 + 4 columns: run `python -m twin.hybrid` again")
    names, slot, bought = FEATURES + EXTRA, C0 + 3, X[:, FEATURES.index("u_p_buys")] > 0
    late = ts >= to_ms(VALID_START)
    click, buy = np.where(bought, 0, click).astype(np.int8), np.where(bought, 0, buy).astype(np.int8)
    tr, va, va_all, cold_all = ~late & ~bought, late & ~anon & ~bought, late & ~anon, late & anon

    def fit(rows, y, params, valid=None, rounds=None):
        base = {"objective": "binary", "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 6, "max_cat_to_onehot": 8,
                "cat_smooth": 50, "seed": 7, **params}
        d = lgb.Dataset(X[rows], y[rows], weight=w[rows], feature_name=names, categorical_feature=CATEGORICAL)
        if rounds:
            return lgb.train(base, d, num_boost_round=rounds)
        dv = lgb.Dataset(X[valid], y[valid], weight=w[valid], reference=d)
        return lgb.train(base, d, num_boost_round=3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])

    p_click = {"num_leaves": 63, "min_data_in_leaf": 300}
    p_buy = {"num_leaves": 15, "min_data_in_leaf": 100, "learning_rate": 0.03}
    cm = fit(tr, click, p_click, valid=va)
    bm = fit(tr & (click == 1), buy, p_buy, valid=va & (click == 1))

    def rank(rows):
        flat = X[rows].copy()
        flat[:, slot] = 1
        return np.where(bought[rows], -1.0, cm.predict(flat) * bm.predict(flat))

    pc = cm.predict(X[va])
    metrics = {"top1_bought_model": top1(rank(va_all), buy[va_all]), "top1_bought_model_cold": top1(rank(cold_all), buy[cold_all]),
               "mrr_bought_model": mrr(rank(va_all), buy[va_all]), "click_auc": roc_auc_score(click[va], pc),
               "purchase_auc": roc_auc_score(buy[va], pc * bm.predict(X[va])),
               "click_rounds": cm.best_iteration, "buy_rounds": bm.best_iteration, "valid_from": VALID_START}
    print(json.dumps(metrics, indent=1), flush=True)

    every = ~bought                                # live model: every day, same number of rounds + 10 %
    cm = fit(every, click, p_click, rounds=int(cm.best_iteration * 1.1))
    bm = fit(every & (click == 1), buy, p_buy, rounds=int(bm.best_iteration * 1.1))
    out.mkdir(parents=True, exist_ok=True)
    cm.save_model(str(out / "click.txt"))
    bm.save_model(str(out / "buy.txt"))
    g = cm.feature_importance("gain")
    (out / "meta.json").write_text(json.dumps({"features": names, "metrics": {**metrics, "refit_on_all_days": True},
                                               "importance": {n: round(float(v / g.sum()), 4) for n, v in zip(names, g)}}, indent=1))
    print(f"saved to {out}")


if __name__ == "__main__":
    if "--train-only" not in sys.argv:
        export()
    train()
