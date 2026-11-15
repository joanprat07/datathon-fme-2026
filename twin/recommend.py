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
from twin.variants import Variant, intent_slate
from twin.embeddings import semantic_candidates
from twin.hybrid_live import EMB_FILE, EXTRA, Embeddings

SLOT = C0 + 3
EXPLOIT_SLOTS = 7      # slots filled by expected value; the rest explore
MAX_PER_SUB = 5        # diversity: at most this many products of one subcategory
PRICE_EPS = 0.0005     # tiny preference for pricier products at equal purchase probability (tie-breaker: revenue)
BLOCK_BOUGHT = True    # business rule: never recommend a product the shopper already bought


class Ranker:
    def __init__(self, path=ARTIFACTS / "model"):
        meta = json.loads((path / "meta.json").read_text())
        self.emb = None
        if meta["features"] == FEATURES + EXTRA:       # v5-hybrid: four more columns from twin/hybrid_live.py
            self.emb = Embeddings(path / EMB_FILE)
        elif meta["features"] != FEATURES:
            raise RuntimeError("the saved model was trained with a different feature list: run `python -m twin.train --refit`")
        self.click = lgb.Booster(model_file=str(path / "click.txt"))
        self.buy = lgb.Booster(model_file=str(path / "buy.txt"))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.click.predict(X, num_threads=4), self.buy.predict(X, num_threads=4)


def top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores) - 1)
    return np.argpartition(-scores, k)[:k]


def candidates(st: State, ts: int, dev: int, reg: int, ui: int, rv: list[int], rng: np.random.Generator,
               block_bought: bool = BLOCK_BOUGHT, wide: bool = False) -> np.ndarray:
    """`wide` (variant "wide") adds, without changing anything else:
    - a shopper we know little about (fewer than 5 events; live, st.user() gives every shopper an
      index, so "ui < 0" never happens) gets the top 300 by popularity instead of 40, and the top
      100 of their device x region instead of 40. Measured on 406 validation purchases
      (proves_numeriques/recall.py, 15/11 00:34): the cold pool held the bought product 82.0% of
      the time, top-300 popularity alone 96.8%; known shoppers were at 98.5%.
    - the last 10 views (not 5) vote together, each weighing 0.7x the one after it: products linked
      to several recent views, above all the newest, enter the pool even if no single view ranks them."""
    n = st.P
    thin = wide and (ui < 0 or st.u_events[ui] < 5)
    picks: list[np.ndarray] = [st.pop_order[:300 if thin else 40], top(st.car_clicks[:n], 20)]
    subs: list[int] = []
    past: list[int] = []

    if rv:
        recent = list(dict.fromkeys(reversed(rv)))[:5]          # newest first, no repeats
        picks.append(np.asarray(recent))
        for j, a in enumerate(recent):
            picks.append(top(st.VV[a, :n] / st.nv[:n], 60 if j == 0 else 25))
            picks.append(top(st.VP[a, :n] / st.nb[:n], 30 if j == 0 else 10))
        subs += list(dict.fromkeys(int(s) for s in st.sub[recent]))[:2]
        if wide:
            last = np.asarray(list(dict.fromkeys(reversed(rv)))[:10])
            w = 0.7 ** np.arange(len(last))
            picks.append(last)
            # each view's row is scaled by its own popularity, so one heavily viewed product cannot outvote the rest
            picks.append(top((w[:, None] * st.VV[last, :n] / st.nv[last][:, None]).sum(0) / st.nv[:n], 80))
            picks.append(top((w[:, None] * st.VP[last, :n] / st.nv[last][:, None]).sum(0) / st.nb[:n], 40))

    if ui >= 0:
        # the shopper's own products: what they bought and what they looked at most
        ix, v, c = st.user_products(ui)
        if len(ix):
            picks.append(ix[np.argsort(-(v + 3 * c))[:150]])
        affinity = st.us_v[ui] + 3 * st.us_c[ui] + 6 * st.us_b[ui]
        if affinity.any():
            subs += [int(s) for s in np.argsort(-affinity)[:3] if affinity[s] > 0]
        past = st.bought_before(ui, ts)[0]
        if past and not block_bought:   # variant m1-rebuy: what they bought before can be recommended again
            picks.append(np.asarray(past[-60:]))
        if past:   # products bought by people who bought the same things
            pa = np.asarray(past[-30:])
            picks.append(top((st.BB[pa, :n] / st.nb[pa][:, None]).sum(0) / st.nb[:n], 80))
    for j, s in enumerate(dict.fromkeys(subs)):
        if s in st.sub_top:
            picks.append(st.sub_top[s][:40 if j == 0 else 25])

    if dev >= 0 and reg >= 0:
        picks.append(top(st.seg[dev, reg, :n], 100 if thin else 40))
    trend = st.sub_trend(ts)
    for s in np.argsort(-trend)[:3]:
        if int(s) in st.sub_top:
            picks.append(st.sub_top[int(s)][:10])

    new = np.flatnonzero(st.is_new[:n])
    if len(new):
        picks.append(new[np.isin(st.sub[new], subs)][:40])
        picks.append(rng.choice(new, min(15, len(new)), replace=False))

    if __import__("os").environ.get("TWIN_EMBEDDINGS") == "1":
        picks.append(semantic_candidates(st, rv, ui))
    cand = np.unique(np.concatenate(picks))
    cand = cand[st.ok[cand]]
    if block_bought and past:
        cand = cand[~np.isin(cand, past)]
    return cand


def widen(st: State, rk: Ranker, layer, cand: np.ndarray, ts: int, ui: int, rv: list[int], block_bought: bool) -> np.ndarray:
    """Variant "wide": two more sources of candidates that need the live layer or the embeddings.

    - Price drops: the live layer boosts a product for an hour after its price falls by more than
      10%, but only if something else had already made it a candidate. Here the 30 most popular
      discounted products enter the pool, so a discount can be shown to anyone.
    - New products in the session: a product added during the event has no co-views of its own
      (it borrows its neighbours'). When one of the last views is new, the 30 products with the
      closest description (content vectors of the v5 ranker) enter the pool too."""
    more: list[np.ndarray] = []
    if layer is not None:
        hot = np.flatnonzero(layer.boost_until[:st.P] > ts)
        if len(hot):
            more.append(hot[np.argsort(-st.pop[hot])[:30]])
    if rk.emb is not None and rv:
        fresh = [a for a in dict.fromkeys(reversed(rv)) if st.is_new[a]][:2]
        if fresh:
            if rk.emb.content is None or st.P > rk.emb.n:
                rk.emb._extend(st)
            E = rk.emb.content[:st.P]
            for a in fresh:
                if float(E[a] @ E[a]) > 0.5:          # no usable description vector: nothing to look up
                    more.append(top(E @ E[a], 30))
    if not more:
        return cand
    add = np.unique(np.concatenate(more))
    add = add[st.ok[add]]
    if block_bought and ui >= 0:
        past = st.bought_before(ui, ts)[0]
        if past:
            add = add[~np.isin(add, past)]
    return np.union1d(cand, add)


def fallback(st: State, exclude: set[int] = frozenset(), k: int = 10) -> list[int]:
    """All-time best sellers that are active and in stock: what the organisers' fallback shows."""
    out = []
    for i in np.argsort(-st.purch[:st.P]):
        if st.ok[i] and i not in exclude:
            out.append(int(i))
            if len(out) == k:
                break
    return out


BASE = Variant()


def recommend(st: State, rk: Ranker, req: dict, rng: np.random.Generator, v: Variant = BASE, layer=None):
    """Returns (10 product ids, their indices, expected click probability of each slot,
    expected purchase probability of each slot, the features of the chosen products at their
    slots). `v` selects the variant (twin/variants.py); with the default it behaves exactly as
    the original model."""
    ts = int(req["t"])
    page, dev = PAGES.get(req.get("page"), -1), DEVICES.get(req.get("device"), -1)
    reg = st.region_id.get(req.get("region"), -1)
    ui = st.user(str(req["user_id"]))
    rv = [st.idx[p] for p in (req.get("rv") or []) if p in st.idx]

    cand = candidates(st, ts, dev, reg, ui, rv, rng, v.block_bought, v.wide)
    if v.wide:
        cand = widen(st, rk, layer, cand, ts, ui, rv, v.block_bought)
    if len(cand) < 10:
        bought = set(st.bought_before(ui, ts)[0]) if v.block_bought else set()
        cand = np.unique(np.concatenate([cand, np.asarray(fallback(st, bought, 20), dtype=int)]))
    X = st.featurize(ts, page, dev, reg, ui, rv, cand, 1)
    if rk.emb is not None:                              # v5-hybrid ranker: append the four hybrid variables
        X = np.column_stack([X, rk.emb.features(st, rv, ui, cand)])
    p_click, p_buy = rk.predict(X)
    p_purchase = p_click * (p_buy if v.buy_power == 1.0 else p_buy ** v.buy_power)
    value = p_purchase * (1 + v.price_eps * st.price[cand])
    if not v.bandit:
        mean = sample = np.ones(len(cand))
    elif v.live_layer and layer is not None:
        mean, sample = layer.multipliers(st, cand, rng, ts)
    else:
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

    if v.slate == "intent":
        group = st.sub[cand] if v.intent_level == "sub" else st.cat[cand]
        chosen = intent_slate(exploit, explore, p_click * p_buy * mean, group, v.exploit_slots, v.lam)
    else:
        best = np.argsort(-exploit)
        take(best, v.exploit_slots, v.max_per_sub)
        if v.exploit_slots < 10:
            take(np.argsort(-explore), 10, v.max_per_sub)
        take(best, 10, 10)                              # not enough variety: relax the cap

    items = cand[chosen]
    Xs = X[chosen].copy()
    Xs[:, SLOT] = np.arange(1, len(chosen) + 1)
    expected = rk.click.predict(Xs, num_threads=1)
    expected_buy = expected * p_buy[chosen]
    if len(items) < 10:                                  # should not happen; keep the answer valid anyway
        extra = np.asarray(fallback(st, set(items.tolist()), 10 - len(items)), dtype=int)
        items = np.concatenate([items, extra])
        expected = np.concatenate([expected, np.zeros(len(extra))])
        expected_buy = np.concatenate([expected_buy, np.zeros(len(extra))])
    return [st.pid[i] for i in items], items, expected, expected_buy, Xs
