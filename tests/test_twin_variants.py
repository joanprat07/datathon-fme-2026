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
import twin.declog  # noqa: E402
import twin.train  # noqa: E402
import twin.train_feedback as TF  # noqa: E402
import twin.variants as V  # noqa: E402
from tests.twin_synth import make_snapshot  # noqa: E402

REAL_MODEL = twin.rt.ROOT / "artifacts" / "model"
TMP = Path(tempfile.mkdtemp(prefix="twin-test-"))
INFO = make_snapshot(TMP / "live", live=True)
twin.state.ARTIFACTS = TMP
shutil.copytree(REAL_MODEL, TMP / "model")           # el model real, a la carpeta temporal de les proves
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
    for name in [None, *V.VARIANTS, "noexp+price0+cap3", "twin-v2=noexp+price0"]:
        v = V.get_variant(name) if name else R.BASE
        for k in range(40):
            req = _req(k, user=f"nou_{k}" if k % 5 == 0 else None, rv=["p_new1", "desconegut"] if k % 7 == 0 else None)
            ids, items, exp_click, exp_buy, Xs = R.recommend(ST, RK, req, rng, v, layer)
            _valid(ids, items)
            assert len(exp_click) == len(exp_buy) == 10 and (exp_buy <= exp_click + 1e-9).all()
            assert Xs.shape == (10, twin.state.NF) and list(Xs[:, R.SLOT]) == list(range(1, 11))


def test_single_setting_variants_do_what_they_say():
    subs_of = lambda items: np.bincount(ST.sub[items]).max()
    same = more_than_3 = 0
    for k in range(60):
        req = _req(k, rv=[INFO["pid"][(30 * j + k) % 400] for j in range(4)])     # sessió concentrada en una subcategoria
        a = R.recommend(ST, RK, req, np.random.default_rng(1), V.get_variant("noexp"))[1]
        b = R.recommend(ST, RK, req, np.random.default_rng(2), V.get_variant("noexp"))[1]
        assert list(a) == list(b)                                   # sense exploració no hi ha atzar
        base = R.recommend(ST, RK, req, np.random.default_rng(1), R.BASE)[1]
        assert list(a[:7]) == list(base[:7])                        # els 7 primers són els mateixos que el model base
        same += list(a) == list(base)
        more_than_3 += subs_of(base) > 3
        assert subs_of(R.recommend(ST, RK, req, np.random.default_rng(1), V.get_variant("cap3"))[1]) <= 3
        assert subs_of(base) <= 5
    assert same < 60                                                # i els 3 últims sí que canvien alguna vegada
    assert more_than_3 > 0                                          # el límit de 3 és una restricció real en aquestes dades
    v = V.get_variant("twin-v2=noexp+price0")
    assert v.name == "twin-v2" and v.exploit_slots == 10 and v.price_eps == 0 and v.max_per_sub == 5 and v.bandit
    assert V.get_variant("fb+noexp").model == "model_fb" and V.get_variant("nobandit").bandit is False


def test_arms_file_is_read_without_restart():
    f = TMP / "twin_arms.txt"
    f.write_text("# prova\ntwin-v1: 50\n\nnoexp: 25   # sense exploració\ntwin-v2=noexp+price0: 25\nholdout: 5\n", encoding="utf-8")
    spec, holdout = V.read_arms_file(f)
    assert holdout == 0.05 and spec == "twin-v1:50,noexp:25,twin-v2=noexp+price0:25"
    twin.variants.ARTIFACTS = TMP
    arms = V.parse_arms(spec, "twin-v1")
    assert [(v.name, round(w, 2)) for v, w in arms] == [("twin-v1", 0.5), ("noexp", 0.25), ("twin-v2", 0.25)]
    assert V.read_arms_file(TMP / "no-hi-es.txt") is None
    f.write_text("fb: 30, twin-v1: 70", encoding="utf-8")            # també en una sola línia; fb cau si no hi ha model_fb
    spec, holdout = V.read_arms_file(f)
    assert holdout is None and [v.name for v, _ in V.parse_arms(spec, "x")] == ["twin-v1"]
    assert [v.name for v, _ in V.parse_arms("brossa:abc,twin-v1:1", "x")] == ["twin-v1"]   # un pes il·legible no trenca res


def test_m1_rebuy_only_changes_the_blocking_rule():
    buyers = sorted({u for u, _, _ in INFO["buys"]})[:60]
    shown_base = shown_rebuy = 0
    for k, u in enumerate(buyers):
        ui = ST.user(u)
        bought = set(ST.bought_before(ui, INFO["cut"] + 1000 * k)[0])
        assert bought
        req = _req(k, user=u)
        items = R.recommend(ST, RK, req, np.random.default_rng(k), R.BASE)[1]
        shown_base += len(bought & set(items.tolist()))
        cand = R.candidates(ST, int(req["t"]), 0, 0, ui, [], np.random.default_rng(k), block_bought=False)
        assert bought & set(cand.tolist()) or not any(ST.ok[list(bought)])   # ara sí que són candidats
        items = R.recommend(ST, RK, req, np.random.default_rng(k), V.VARIANTS["m1-rebuy"])[1]
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
    ids, items = R.recommend(ST, rk, _req(3), np.random.default_rng(0), V.VARIANTS["m1-rebuy"])[:2]
    _valid(ids, items)
    twin.train.main(False, keep_bought=False, out="model_check")   # el camí original segueix funcionant
    m0 = json.loads((TMP / "model_check" / "meta.json").read_text())["metrics"]
    assert not m0["keep_bought"] and m0["rows_left_out_already_bought"] > 0


def test_decision_log_and_feedback_training():
    # 1) el bucle desa les features de cada resposta; aquí en simulem 3 hores
    log = twin.declog.DecLog(twin.state.NF, directory=TMP / "declog")
    rng, served = np.random.default_rng(5), {}
    for k in range(2400):
        req = _req(k)
        req["request_id"], req["t"] = f"fb{k}", INFO["cut"] + k * 4500
        ids, items, ec, eb, Xs = R.recommend(ST, RK, req, rng, R.BASE)
        log.add(req["request_id"], req["t"], "twin-v1", ids, Xs, ec, eb)
        served[req["request_id"]] = ids
    log.add("fb0", INFO["cut"], "twin-v1", served["fb0"], Xs, ec, eb)          # una petició repetida no es duplica
    log.add("curta", INFO["cut"], "twin-v1", ["a"], Xs, ec, eb)                # una resposta incompleta s'ignora
    log.flush()
    data = twin.declog.load(TMP / "declog")
    assert len(data["rid"]) == 2400 and data["X"].shape == (2400, 10, twin.state.NF) and data["pids"].shape == (2400, 10)
    assert len(list((TMP / "declog").glob("*.npz"))) >= 3 and not list((TMP / "declog").glob("*.tmp*"))

    # 2) feedback simulat: es clica sobretot el slot 1 i els productes barats; un 15 % ho serveix el fallback
    price = dict(zip(INFO["pid"], ST.price[:len(INFO["pid"])]))
    rows = []
    for k, (rid, ids) in enumerate(served.items()):
        p = np.array([0.25 / (j + 1) * (2.0 if price.get(i, 99) < 60 else 0.5) for j, i in enumerate(ids)])
        j = int(np.argmax(rng.random(10) < p)) if (rng.random(10) < p).any() else -1
        click = ids[j] if j >= 0 else ""
        rows.append({"rid": rid, "served_by": "fallback" if k % 7 == 0 else "team", "clicked": click,
                     "purchased": click if click and rng.random() < 0.4 else ""})
    import pandas as pd
    TF.q_csv = lambda sql, **kw: pd.DataFrame(rows)
    TF.declog.DIR = TMP / "declog"
    TF.ARTIFACTS = TMP
    TF.HISTORY = TMP / "train" / "matrix.npz"
    if not TF.HISTORY.exists():
        make_snapshot(TMP / "train", live=False)
        twin.train.ARTIFACTS, twin.train.CACHE = TMP, TF.HISTORY
        twin.train.main(False, out="model_tmp")

    # 3) entrenament amb historial + feedback, validat amb l'última hora de feedback
    TF.main("model_fb", 3.0, 1.0, True, True, 1.0)
    import json
    m = json.loads((TMP / "model_fb" / "meta.json").read_text())["metrics"]
    assert m["feedback_rows_train"] + m["feedback_rows_valid"] == 10 * sum(r["served_by"] == "team" for r in rows)
    assert m["history_rows"] > 0 and m["refit_on_all"] and 0.3 < m["new"]["click_calibration"] < 3
    assert m["new"]["click_logloss"] < m["current"]["click_logloss"]            # aprèn el patró del feedback simulat
    rk = R.Ranker(TMP / "model_fb")
    ids, items = R.recommend(ST, rk, _req(1), np.random.default_rng(0), V.get_variant("fb"))[:2]
    _valid(ids, items)
    twin.variants.ARTIFACTS = TMP
    assert [v.name for v, _ in V.parse_arms("fb:1", "x")] == ["fb"]             # ara el braç fb ja és vàlid
    TF.main("model_fb2", 3.0, 1.0, False, False, 1.0)                           # només feedback, sense historial
    try:
        TF.main("model", 3.0, 1.0, False, False, 1.0)
        raise AssertionError("no hauria de deixar sobreescriure artifacts/model")
    except SystemExit:
        pass


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
