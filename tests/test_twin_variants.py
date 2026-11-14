"""Proves de les variants del twin amb un snapshot sintètic (sense RawTree) i el model real del repo.

    python -m tests.test_twin_variants
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TEAM_ID", "team_00")           # twin.rt els demana en importar; aquí no hi ha xarxa
os.environ.setdefault("RAWTREE_API_KEY", "test")

import numpy as np  # noqa: E402

import twin.recommend as R  # noqa: E402
import twin.rt  # noqa: E402
import twin.state  # noqa: E402
import twin.train  # noqa: E402
import twin.variants as V  # noqa: E402
from tests.twin_synth import make_snapshot  # noqa: E402

REAL_MODEL = twin.rt.ROOT / "artifacts" / "model"
TMP = Path(tempfile.mkdtemp(prefix="twin-test-"))
INFO = make_snapshot(TMP / "live", live=True)
twin.state.ARTIFACTS = TMP
ST = twin.state.State("live")
RK = R.Ranker(REAL_MODEL)


def _req(k: int, user: str | None = None, rv: list[str] | None = None) -> dict:
    return {"request_id": f"r{k}", "t": INFO["cut"] + 1000 * k, "user_id": user or INFO["users"][k % 80],
            "device": ["web", "ios", "android"][k % 3], "page": ["home", "product", "category"][k % 3],
            "region": ["Catalunya", "Madrid", "Nowhere"][k % 3],
            "rv": rv if rv is not None else [INFO["pid"][(7 * k + j) % 400] for j in range(k % 4)]}


def _valid(ids, items) -> None:
    assert len(ids) == 10 and len(set(ids)) == 10 and ST.ok[items].all()


def test_base_and_all_variants_return_valid_slates():
    rng = np.random.default_rng(1)
    layer = V.LiveLayer(ST)
    for name in (None, "m1-rebuy", "m2-intent", "m3-live"):
        v = V.VARIANTS[name] if name else R.BASE
        for k in range(60):
            req = _req(k, user=f"nou_{k}" if k % 5 == 0 else None, rv=["p_new1", "desconegut"] if k % 7 == 0 else None)
            ids, items, exp_click, exp_buy = R.recommend(ST, RK, req, rng, v, layer)
            _valid(ids, items)
            assert len(exp_click) == len(exp_buy) == 10 and (exp_buy <= exp_click + 1e-9).all()


def test_m1_rebuy_only_changes_the_blocking_rule():
    buyers = sorted({u for u, _, _ in INFO["buys"]})[:60]
    shown_base = shown_rebuy = 0
    for k, u in enumerate(buyers):
        ui = ST.user(u)
        bought = set(ST.bought_before(ui, INFO["cut"] + 1000 * k)[0])
        assert bought
        req = _req(k, user=u)
        _, items, _, _ = R.recommend(ST, RK, req, np.random.default_rng(k), R.BASE)
        shown_base += len(bought & set(items.tolist()))
        cand = R.candidates(ST, int(req["t"]), 0, 0, ui, [], np.random.default_rng(k), block_bought=False)
        assert bought & set(cand.tolist()) or not any(ST.ok[list(bought)])   # ara sí que són candidats
        _, items, _, _ = R.recommend(ST, RK, req, np.random.default_rng(k), V.VARIANTS["m1-rebuy"])
        shown_rebuy += len(bought & set(items.tolist()))
    assert shown_base == 0            # el model base no en mostra mai cap
    assert shown_rebuy > 0            # la variant sí (quan el ranker els puntua prou amunt)


def test_m2_intent_slate():
    rng = np.random.default_rng(0)
    n = 200
    r = rng.uniform(0.001, 0.03, n)
    group = rng.integers(0, 10, n)
    # lam = 0: cap efecte d'intenció → és exactament l'ordre per valor
    assert V.intent_slate(r, r, r, group, 7, lam=0.0) == list(np.argsort(-r)[:10])
    # intenció incerta (valor repartit entre categories) i probabilitats altes → la llista es reparteix
    r_hi = np.full(n, 0.2) + rng.uniform(0, 1e-3, n)
    assert len({int(group[i]) for i in V.intent_slate(r_hi, r_hi, r_hi, group, 10)}) >= 5
    # intenció clara (una categoria concentra el valor) → la llista s'hi concentra
    r_clear = np.where(group == 3, 0.03, 0.0005) + rng.uniform(0, 1e-5, n)
    assert sum(int(group[i]) == 3 for i in V.intent_slate(r_clear, r_clear, r_clear, group, 10)) >= 8
    # mai duplicats, i funciona amb menys de 10 candidats
    assert len(set(V.intent_slate(r, r, r, group, 7))) == 10
    assert sorted(V.intent_slate(r[:4], r[:4], r[:4], group[:4], 7)) == [0, 1, 2, 3]


def test_m3_live_layer():
    layer = V.LiveLayer(ST)
    rng = np.random.default_rng(0)
    cand = np.arange(50)
    now = float(ST.now_ms)
    mean, sample = layer.multipliers(ST, cand, rng, now)
    assert np.allclose(mean, 1.0, atol=1e-6) and sample.min() >= 0.2 and sample.max() <= 4.5
    items, exp = np.arange(10), np.full(10, 0.02)
    for k in range(400):                       # el producte 0 es clica molt més del que s'esperava; l'1, mai
        layer.feedback(ST, items, exp, exp * 0.1, 0 if k % 4 == 0 else -1, -1, now + k)
    mean, _ = layer.multipliers(ST, cand, rng, now + 400)
    assert mean[0] > 1.5 and mean[1] < 0.8 and 0.25 <= mean.min() and mean.max() <= 3.5
    before = layer.click_obs[0]
    layer.feedback(ST, items, exp, exp * 0.1, -1, -1, now + 400 + 2 * 3600 * 1000)      # dues hores després
    assert abs(layer.click_obs[0] / before - 0.5) < 0.01                                 # mitja vida de 2 h
    # tendència: una ràfega d'activitat sobre un producte el puja; la resta queda neutra
    i = 20
    rows = [{"t": int(now) + 1000 + j, "et": "view", "p": ST.pid[i]} for j in range(300)]
    layer.events(ST, rows + [{"t": int(now) + 2000, "et": "view", "p": "no-existeix"}])
    assert layer._trend[i] > 1.2 and 0.8 <= layer._trend[:ST.P].min() and layer._trend[:ST.P].max() <= 1.4
    assert abs(float(np.median(layer._trend[:ST.P])) - 1.0) < 0.05
    # rebaixa de preu > 10 % → impuls durant una hora
    layer.price_change(30, 100.0, 85.0, now)
    layer.price_change(31, 100.0, 95.0, now)
    m, _ = layer.multipliers(ST, np.array([30, 31]), rng, now + 60_000)
    m_later, _ = layer.multipliers(ST, np.array([30, 31]), rng, now + 3_700_000)
    assert m[0] / m[1] > 1.1 and abs(m_later[0] / m_later[1] - m[0] / m[1]) > 0.1
    # a l'inici de la final, l'evidència acumulada es redueix a un quart, un sol cop
    layer2 = V.LiveLayer(ST)
    layer2.feedback(ST, items, exp, exp, 0, 0, now)
    obs = layer2.click_obs[0]
    layer2.feedback(ST, items, exp * 0, exp * 0, -1, -1, float(V.FINAL_START_MS) + 1)
    decay = 0.5 ** ((V.FINAL_START_MS + 1 - now) / layer2.hl_ms)
    assert abs(layer2.click_obs[0] - obs * decay * 0.25) < 1e-9 and layer2.final_done


def test_arms_parsing_and_stable_split():
    twin.variants.ARTIFACTS = TMP
    (TMP / "model").mkdir(exist_ok=True)
    (TMP / "model" / "meta.json").write_text("{}")
    arms = V.parse_arms("twin-v1:60,m1-rebuy:20,m2-intent:10,m3-live:10", "twin-v1")
    assert [v.name for v, _ in arms] == ["twin-v1", "m2-intent", "m3-live"]        # m1 cau: no hi ha model_rebuy
    assert abs(sum(w for _, w in arms) - 1) < 1e-9
    ids = [f"r_{i}" for i in range(20000)]
    first = [V.pick_arm(arms, r).name for r in ids]
    assert first == [V.pick_arm(arms, r).name for r in ids]                         # estable
    assert abs(first.count("twin-v1") / len(ids) - 0.75) < 0.02 and abs(first.count("m3-live") / len(ids) - 0.125) < 0.02
    assert [v.name for v, _ in V.parse_arms("", "x")] == ["x"]
    combo = V.get_variant("m2-intent+m3-live")                                      # combinació de dues variants
    assert combo.name == "m2-intent+m3-live" and combo.slate == "intent" and combo.live_layer and combo.block_bought
    assert V.get_variant("m1-rebuy+m3-live").model == "model_rebuy" and not V.get_variant("m1-rebuy+m3-live").block_bought
    assert V.get_variant("qualsevol") == V.Variant("qualsevol")
    assert [v.name for v, _ in V.parse_arms("m1-rebuy:100", "twin-v9")] == ["twin-v9"]   # res vàlid → model base


def test_train_keep_bought_runs_and_reports_repurchases():
    make_snapshot(TMP / "train", live=False)
    twin.train.ARTIFACTS, twin.train.CACHE = TMP, TMP / "train" / "matrix.npz"
    twin.train.main(False, keep_bought=True, out="model_rebuy")
    import json
    meta = json.loads((TMP / "model_rebuy" / "meta.json").read_text())
    m = meta["metrics"]
    assert m["keep_bought"] and m["rows_left_out_already_bought"] == 0
    assert m["repurchase_share_of_purchases"] > 0.2            # al sintètic els ja comprats es compren molt
    assert m["top1_bought_model"] >= m["top1_bought_model_if_blocked"]
    rk = R.Ranker(TMP / "model_rebuy")                         # el model nou es carrega i serveix
    ids, items, _, _ = R.recommend(ST, rk, _req(3), np.random.default_rng(0), V.VARIANTS["m1-rebuy"])
    _valid(ids, items)
    twin.train.main(False, keep_bought=False, out="model_check")   # el camí original segueix funcionant
    m0 = json.loads((TMP / "model_check" / "meta.json").read_text())["metrics"]
    assert not m0["keep_bought"] and m0["rows_left_out_already_bought"] > 0


if __name__ == "__main__":
    import time
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"OK    {name}  ({time.time() - t0:.1f} s)")
            except Exception:
                failed += 1
                print(f"FALLA {name}")
                traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    raise SystemExit(1 if failed else 0)
