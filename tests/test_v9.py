"""V9 (cold8): the main ranker for products with history, model_v7s for products without.

    python -m pytest -q tests/test_v9.py        (needs artifacts/model_v5 and artifacts/model_v7s)
"""
from __future__ import annotations

import numpy as np

from tests import test_twin_variants as T
import twin.recommend as R
import twin.variants as V

MAIN = R.Ranker(T.REAL_MODEL.parent / "model_v5")
COLD = R.Ranker(T.REAL_MODEL.parent / "model_v7s")


def _mark_new(k: int) -> np.ndarray:
    st = T.ST
    keep = st.is_new.copy()
    idx = np.flatnonzero(st.ok[:st.P])[:k]
    st.is_new[idx] = True
    return keep


def test_v9_recipes_parse():
    v = V.get_variant("v9")
    assert (v.model, v.slate, v.live_layer, v.exploit_slots, v.cold_model) == ("model_v6", "intent", True, 10, "model_v7s")
    w = V.get_variant("v5-hybrid+m2-intent+m3-live+noexp+cold8")
    assert (w.model, w.slate, w.live_layer, w.exploit_slots, w.cold_model) == ("model_v5", "intent", True, 10, "model_v7s")
    assert V.arm_models([(w, 1.0)]) == {"model_v5", "model_v7s"}


def test_cold_blend_only_moves_new_products_and_keeps_scale():
    st, rng = T.ST, np.random.default_rng(3)
    keep = _mark_new(60)
    try:
        cand = np.flatnonzero(st.ok[:st.P])[:300]
        X = st.featurize(T.INFO["cut"] + 5000, 0, 0, 0, st.user(T.INFO["users"][3]), [], cand, 1)
        emb = MAIN.emb.features(st, [], st.user(T.INFO["users"][3]), cand)
        pc, pb = MAIN.predict(np.column_stack([X, emb]))
        qc, qb = R.cold_blend(st, COLD, cand, X, [], st.user(T.INFO["users"][3]), emb, pc, pb)
        new = st.is_new[cand]
        assert new.sum() == 60
        assert np.array_equal(qc[~new], pc[~new]) and np.array_equal(qb[~new], pb[~new])
        # new products now sit inside the known products' range (same scale)
        assert qc[new].min() >= pc[~new].min() - 1e-12 and qc[new].max() <= pc[~new].max() + 1e-12
        # and in the cold model's order
        cc, _ = COLD.predict(np.column_stack([X, emb]))
        o = np.argsort(cc[new], kind="stable")
        assert (np.diff(qc[new][o]) >= -1e-12).all()
    finally:
        st.is_new[:] = keep


def test_almost_all_new_uses_the_cold_model_alone():
    st = T.ST
    keep = _mark_new(st.P)
    try:
        cand = np.flatnonzero(st.ok[:st.P])[:200]
        X = st.featurize(T.INFO["cut"] + 5000, 0, 0, 0, -1, [], cand, 1)
        emb = MAIN.emb.features(st, [], -1, cand)
        pc, pb = MAIN.predict(np.column_stack([X, emb]))
        qc, qb = R.cold_blend(st, COLD, cand, X, [], -1, emb, pc, pb)
        cc, cb = COLD.predict(np.column_stack([X, emb]))
        assert np.allclose(qc, cc) and np.allclose(qb, cb)
    finally:
        st.is_new[:] = keep


def test_v9_slates_are_valid():
    st, rng = T.ST, np.random.default_rng(4)
    layer = V.LiveLayer(st)
    keep = _mark_new(40)
    try:
        v = V.get_variant("v5-hybrid+m2-intent+m3-live+noexp+cold8")
        shown_new = 0
        for k in range(60):
            req = T._req(k, user=f"nou_{k}" if k % 5 == 0 else None)
            ids, items, e_click, e_buy, Xs = R.recommend(st, MAIN, req, rng, v, layer, COLD)
            T._valid(ids, items)
            assert (e_click <= 0.95 + 1e-12).all() and (e_buy <= e_click + 1e-9).all() and (e_click >= 0).all()
            shown_new += int(st.is_new[items].sum())
        assert shown_new > 0, "the blend should let some new products into the slates"
    finally:
        st.is_new[:] = keep
