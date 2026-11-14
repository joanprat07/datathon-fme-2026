"""One request in, ten products out: candidates -> LightGBM ranker -> bandit -> slate.

1. Candidates (~300-500): the shopper's own products (bought or viewed before), products linked
   to what they are looking at now (co-views, view->buy, same subcategory), products bought by
   people with the same purchases, popular in their device x region, popular overall, trending
   subcategories, and products added during the event.
2. Ranker: p(click) x p(purchase | click) for each candidate, as if shown in slot 1.
3. Bandit: each product's score is corrected by how it really performed in our own carousels
   (observed / expected clicks). Slots 1-7 use the corrected score; slots 8-10 use a Thompson
   sample of it, so uncertain products (new ones above all) get tried where it costs least.
4. Filters: in stock and active, never a product the shopper already bought (organically or
   through our carousel; the README says they tend not to want it again), at most MAX_PER_SUB
   products of one subcategory. Missing slots are filled with best sellers.

The ranker is trained without the already-bought rows as well (see twin/train.py), so it does
not learn the repurchase pattern of the old carousel's log. BLOCK_BOUGHT switches the rule.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from twin.rt import ARTIFACTS
from twin.state import C0, DEVICES, FEATURES, PAGES, State

SLOT = C0 + 3
EXPLOIT_SLOTS = 7      # slots filled by expected value; the rest explore
MAX_PER_SUB = 5        # diversity: at most this many products of one subcategory
PRICE_EPS = 0.0005     # tiny preference for pricier products at equal purchase probability (tie-breaker: revenue)
BLOCK_BOUGHT = True    # business rule: never recommend a product the shopper already bought


class Ranker:
    def __init__(self, path=ARTIFACTS / "model"):
        meta = json.loads((path / "meta.json").read_text())
        if meta["features"] != FEATURES:
            raise RuntimeError("the saved model was trained with a different feature list: run `python -m twin.train --refit`")
        self.click = lgb.Booster(model_file=str(path / "click.txt"))
        self.buy = lgb.Booster(model_file=str(path / "buy.txt"))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.click.predict(X, num_threads=4), self.buy.predict(X, num_threads=4)


def top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores) - 1)
    return np.argpartition(-scores, k)[:k]


def candidates(st: State, ts: int, dev: int, reg: int, ui: int, rv: list[int], rng: np.random.Generator) -> np.ndarray:
    n = st.P
    picks: list[np.ndarray] = [st.pop_order[:40], top(st.car_clicks[:n], 20)]
    subs: list[int] = []
    past: list[int] = []

    if rv:
        recent = list(dict.fromkeys(reversed(rv)))[:5]          # newest first, no repeats
        picks.append(np.asarray(recent))
        for j, a in enumerate(recent):
            picks.append(top(st.VV[a, :n] / st.nv[:n], 60 if j == 0 else 25))
            picks.append(top(st.VP[a, :n] / st.nb[:n], 30 if j == 0 else 10))
        subs += list(dict.fromkeys(int(s) for s in st.sub[recent]))[:2]

    if ui >= 0:
        # the shopper's own products: what they bought and what they looked at most
        ix, v, c = st.user_products(ui)
        if len(ix):
            picks.append(ix[np.argsort(-(v + 3 * c))[:150]])
        affinity = st.us_v[ui] + 3 * st.us_c[ui] + 6 * st.us_b[ui]
        if affinity.any():
            subs += [int(s) for s in np.argsort(-affinity)[:3] if affinity[s] > 0]
        past = st.bought_before(ui, ts)[0]
        if past:   # products bought by people who bought the same things
            pa = np.asarray(past[-30:])
            picks.append(top((st.BB[pa, :n] / st.nb[pa][:, None]).sum(0) / st.nb[:n], 80))
    for j, s in enumerate(dict.fromkeys(subs)):
        if s in st.sub_top:
            picks.append(st.sub_top[s][:40 if j == 0 else 25])

    if dev >= 0 and reg >= 0:
        picks.append(top(st.seg[dev, reg, :n], 40))
    trend = st.sub_trend(ts)
    for s in np.argsort(-trend)[:3]:
        if int(s) in st.sub_top:
            picks.append(st.sub_top[int(s)][:10])

    new = np.flatnonzero(st.is_new[:n])
    if len(new):
        picks.append(new[np.isin(st.sub[new], subs)][:40])
        picks.append(rng.choice(new, min(15, len(new)), replace=False))

    cand = np.unique(np.concatenate(picks))
    cand = cand[st.ok[cand]]
    if BLOCK_BOUGHT and past:
        cand = cand[~np.isin(cand, past)]
    return cand


def fallback(st: State, exclude: set[int] = frozenset(), k: int = 10) -> list[int]:
    """All-time best sellers that are active and in stock: what the organisers' fallback shows."""
    out = []
    for i in np.argsort(-st.purch[:st.P]):
        if st.ok[i] and i not in exclude:
            out.append(int(i))
            if len(out) == k:
                break
    return out


def recommend(st: State, rk: Ranker, req: dict, rng: np.random.Generator):
    """Returns (10 product ids, their indices, expected click probability of each slot)."""
    ts = int(req["t"])
    page, dev = PAGES.get(req.get("page"), -1), DEVICES.get(req.get("device"), -1)
    reg = st.region_id.get(req.get("region"), -1)
    ui = st.user(str(req["user_id"]))
    rv = [st.idx[p] for p in (req.get("rv") or []) if p in st.idx]

    cand = candidates(st, ts, dev, reg, ui, rv, rng)
    if len(cand) < 10:
        bought = set(st.bought_before(ui, ts)[0]) if BLOCK_BOUGHT else set()
        cand = np.unique(np.concatenate([cand, np.asarray(fallback(st, bought, 20), dtype=int)]))
    X = st.featurize(ts, page, dev, reg, ui, rv, cand, 1)
    p_click, p_buy = rk.predict(X)
    value = p_click * p_buy * (1 + PRICE_EPS * st.price[cand])
    mean, sample = st.bandit(cand, rng)
    exploit, explore = value * mean, value * sample

    chosen: list[int] = []
    taken: set[int] = set()
    per_sub: dict[int, int] = {}

    def take(order: np.ndarray, upto: int, cap: int) -> None:
        for j in order:
            if len(chosen) >= upto:
                return
            s = int(st.sub[cand[j]])
            if j in taken or per_sub.get(s, 0) >= cap:
                continue
            chosen.append(int(j))
            taken.add(int(j))
            per_sub[s] = per_sub.get(s, 0) + 1

    best = np.argsort(-exploit)
    take(best, EXPLOIT_SLOTS, MAX_PER_SUB)
    take(np.argsort(-explore), 10, MAX_PER_SUB)
    take(best, 10, 10)                                  # not enough variety: relax the cap

    items = cand[chosen]
    Xs = X[chosen].copy()
    Xs[:, SLOT] = np.arange(1, len(chosen) + 1)
    expected = rk.click.predict(Xs, num_threads=1)
    if len(items) < 10:                                  # should not happen; keep the answer valid anyway
        extra = np.asarray(fallback(st, set(items.tolist()), 10 - len(items)), dtype=int)
        items = np.concatenate([items, extra])
        expected = np.concatenate([expected, np.zeros(len(extra))])
    return [st.pid[i] for i in items], items, expected
