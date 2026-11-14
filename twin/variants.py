"""Model variants that can run side by side in the live loop, to compare them on real traffic.

    TWIN_ARMS="twin-v1:60,m1-rebuy:20,m2-intent:10,m3-live:10" python -m twin.live

Each arm is one `Variant`. A request is assigned to an arm by a stable hash of its request_id and
the arm's name is written as `model_version`, so `python -m twin.monitor` compares them directly.
Without TWIN_ARMS the loop runs exactly as before (one arm, the base model).

The three variants change one thing each, so the comparison says which idea works:

    m1-rebuy    what we are allowed to recommend: products the shopper already bought are no longer
                blocked, and the ranker is retrained WITH those rows (python -m twin.train
                --keep-bought --out model_rebuy --refit), so it learns from the log what a past
                purchase means (features u_p_buys, u_p_buy_days).
    m2-intent   how the 10 slots are filled: instead of "at most 5 per subcategory", each product
                picked lowers the value of the next ones of the same category in proportion to how
                likely it is to have already satisfied that intent.
    m3-live     how we learn from the stream: the bandit forgets (2 h half-life), also looks at
                purchases per category, and adds a per-product trend and a price-drop boost.
"""
from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, replace

import numpy as np

from twin.rt import ARTIFACTS
from twin.state import State, to_ms

FINAL_START_MS = to_ms("2026-11-15 09:00:00")     # Sunday 10:00 CET: the stream switches to new data
EXAM = (1.0 / (1.0 + 0.35 * np.arange(10))).astype(np.float64)   # how much each slot is looked at (slot 1 = 1)
LN2 = float(np.log(2.0))


@dataclass(frozen=True)
class Variant:
    name: str = "twin-v1"
    model: str = "model"          # folder under artifacts/ with click.txt, buy.txt, meta.json
    block_bought: bool = True     # never recommend a product the shopper already bought
    slate: str = "cap"            # "cap": at most MAX_PER_SUB per subcategory | "intent": diminishing returns
    intent_level: str = "cat"     # "cat" or "sub": what counts as one intent
    lam: float = 1.0              # strength of the intent effect (0 = plain sort)
    live_layer: bool = False      # use LiveLayer multipliers instead of the base bandit


VARIANTS = {
    "m1-rebuy": Variant("m1-rebuy", model="model_rebuy", block_bought=False),
    "m2-intent": Variant("m2-intent", slate="intent"),
    "m3-live": Variant("m3-live", live_layer=True),
}


def get_variant(name: str) -> Variant:
    """A registered variant, a combination joined with '+' ('m1-rebuy+m3-live': both changes at once),
    or, for any other name, the base model under that name."""
    v, base = Variant(name), Variant()
    for part in name.split("+"):
        known = VARIANTS.get(part)
        if known is not None:   # take every setting this variant changes
            changed = {f: getattr(known, f) for f in base.__dataclass_fields__ if f != "name" and getattr(known, f) != getattr(base, f)}
            v = replace(v, **changed)
    return v


def parse_arms(spec: str, default_name: str) -> list[tuple[Variant, float]]:
    """'twin-v1:60,m1-rebuy:20' -> [(Variant, share)]. Unknown names run the base model under that name.
    Arms whose model folder does not exist are dropped (their traffic goes to the others)."""
    arms: list[tuple[Variant, float]] = []
    for part in [p for p in spec.split(",") if p.strip()]:
        name, _, weight = part.strip().partition(":")
        v = get_variant(name)
        if not (ARTIFACTS / v.model / "meta.json").exists():
            print(f"arm '{name}' skipped: artifacts/{v.model}/ does not exist")
            continue
        if float(weight or 1) > 0:
            arms.append((v, float(weight or 1)))
    if not arms:
        arms = [(Variant(default_name), 1.0)]
    total = sum(w for _, w in arms)
    return [(v, w / total) for v, w in arms]


def pick_arm(arms: list[tuple[Variant, float]], request_id: str) -> Variant:
    """Stable assignment: the same request always lands in the same arm (independent of the holdout hash)."""
    if len(arms) == 1:
        return arms[0][0]
    x = (zlib.crc32(b"arm:" + request_id.encode()) % 10_000) / 10_000
    acc = 0.0
    for v, w in arms:
        acc += w
        if x < acc:
            return v
    return arms[-1][0]


def arms_from_env(default_name: str) -> list[tuple[Variant, float]]:
    spec = os.environ.get("TWIN_ARMS", "")
    return parse_arms(spec, default_name) if spec else [(replace(Variant(), name=default_name), 1.0)]


# ------------------------------------------------------------------------------------ m2: slate by intent
def intent_slate(key_exploit: np.ndarray, key_explore: np.ndarray, r: np.ndarray, group: np.ndarray,
                 exploit_slots: int, lam: float = 1.0, q_max: float = 0.9) -> list[int]:
    """Greedy slate that maximises the chance that at least one product is bought.

    r[i] is P(purchase | shown in slot 1). P(intent g) is taken as the share of r in group g among
    the candidates. If the shopper does want g, product i satisfies them with q = r[i] / P(g); once
    a product of g is in the slate, the next ones of g are worth less by (1 - exam[slot] * q).
    A clear intent (one group holds most of r) makes q small, so the slate concentrates; an unclear
    one makes it spread. No fixed cap per subcategory."""
    n = len(r)
    mass = np.bincount(group, weights=r, minlength=int(group.max()) + 1)
    p_group = np.clip(mass / max(float(mass.sum()), 1e-12), 0.02, 1.0)
    survive = np.ones(len(p_group))
    free = np.ones(n, bool)
    chosen: list[int] = []
    for k in range(min(10, n)):
        key = key_exploit if k < exploit_slots else key_explore
        gain = np.where(free, key * survive[group] ** lam, -np.inf)
        i = int(np.argmax(gain))
        chosen.append(i)
        free[i] = False
        q = min(float(r[i]) / float(p_group[group[i]]), q_max)
        survive[group[i]] *= 1.0 - EXAM[k] * q
    return chosen


# ------------------------------------------------------------------------------------ m3: live layer
class LiveLayer:
    """What the stream teaches us, with forgetting. Fed by every answer of every arm; used by m3-live.

    - clicks per product and per subcategory, observed against expected (as the base bandit), but
      decayed with a 2-hour half-life so yesterday's evidence does not outvote the last hour;
    - purchases per category, observed against expected (purchases are too rare for anything finer);
    - a trend per product from the organic events: activity of the last ~20 minutes against the
      last ~6 hours;
    - a boost for one hour when a product's price drops by more than 10%.
    When the final round starts the accumulated evidence is cut to a quarter once: trends change."""

    def __init__(self, st: State, half_life_s: float = 7200, fast_s: float = 1200, slow_s: float = 21600):
        cap = st.cap
        self.hl_ms, self.fast_ms, self.slow_ms = half_life_s * 1000, fast_s * 1000, slow_s * 1000
        f64 = lambda n: np.zeros(n, np.float64)
        self.click_obs, self.click_exp = f64(cap), f64(cap)
        self.sub_obs, self.sub_exp = f64(st.SCAP), f64(st.SCAP)
        self.cat_buy_obs, self.cat_buy_exp = f64(st.NC + 64), f64(st.NC + 64)
        self.t_bandit = float(st.now_ms)
        # trend: decayed event counts (view 1, purchase 5); start as if the last week's rate had been steady
        rate = (st.views_7d + 5 * st.purch_7d).astype(np.float64) / (7 * 86_400_000)
        self.fast, self.slow = rate * self.fast_ms / LN2, rate * self.slow_ms / LN2
        self.t_trend = float(st.now_ms)
        self.boost_until = f64(cap)
        self.final_done = st.now_ms >= FINAL_START_MS
        self._trend = np.ones(cap, np.float32)

    # ---- feedback ------------------------------------------------------------------------------
    def _decay_bandit(self, now_ms: float) -> None:
        dt = now_ms - self.t_bandit
        if dt > 1000:
            g = 0.5 ** (dt / self.hl_ms)
            for a in (self.click_obs, self.click_exp, self.sub_obs, self.sub_exp, self.cat_buy_obs, self.cat_buy_exp):
                a *= g
            self.t_bandit = now_ms
        if not self.final_done and now_ms >= FINAL_START_MS:
            for a in (self.click_obs, self.click_exp, self.sub_obs, self.sub_exp, self.cat_buy_obs, self.cat_buy_exp):
                a *= 0.25
            self.final_done = True

    def feedback(self, st: State, items: np.ndarray, exp_click: np.ndarray, exp_buy: np.ndarray,
                 clicked: int, bought: int, now_ms: float) -> None:
        """One carousel we served: expected clicks and purchases per shown item against what happened."""
        self._decay_bandit(now_ms)
        sub, cat = st.sub[items], np.minimum(st.cat[items], len(self.cat_buy_exp) - 1)
        self.click_exp[items] += exp_click
        np.add.at(self.sub_exp, sub, exp_click)
        np.add.at(self.cat_buy_exp, cat, exp_buy)
        if clicked >= 0:
            self.click_obs[clicked] += 1
            self.sub_obs[st.sub[clicked]] += 1
        if bought >= 0:
            self.cat_buy_obs[min(int(st.cat[bought]), len(self.cat_buy_exp) - 1)] += 1

    # ---- organic events and catalogue ------------------------------------------------------------
    def events(self, st: State, rows: list[dict]) -> None:
        if not rows:
            return
        now = float(max(int(r["t"]) for r in rows))
        dt = max(now - self.t_trend, 0.0)
        self.fast *= 0.5 ** (dt / self.fast_ms)
        self.slow *= 0.5 ** (dt / self.slow_ms)
        self.t_trend = max(now, self.t_trend)
        idx, w = [], []
        for r in rows:
            i = st.idx.get(r["p"])
            if i is not None and r["et"] != "add_to_cart":
                idx.append(i)
                w.append(5.0 if r["et"] == "purchase" else 1.0)
        if idx:
            np.add.at(self.fast, idx, w)
            np.add.at(self.slow, idx, w)
        n = st.P
        rate_f, rate_s = self.fast[:n] * LN2 / self.fast_ms, self.slow[:n] * LN2 / self.slow_ms
        a = max(float(rate_s.mean()), 1e-12)
        trend = np.ones(len(self._trend), np.float32)
        trend[:n] = np.clip(((rate_f + a) / (rate_s + a)) ** 0.3, 0.8, 1.4)
        self._trend = trend      # new array, swapped in one go

    def price_change(self, i: int, old_price: float, new_price: float, now_ms: float) -> None:
        if old_price > 0 and new_price < 0.9 * old_price:
            self.boost_until[i] = now_ms + 3_600_000

    # ---- multipliers -----------------------------------------------------------------------------
    def multipliers(self, st: State, cand: np.ndarray, rng: np.random.Generator, now_ms: float,
                    a_item: float = 4.0, a_sub: float = 8.0, a_cat: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """(posterior mean, Thompson sample) multipliers for the ranker's score of each candidate."""
        g = (self.sub_obs.sum() + 20) / (self.sub_exp.sum() + 20)
        sub = st.sub[cand]
        m_sub = (self.sub_obs[sub] + a_sub * g) / (self.sub_exp[sub] + a_sub)
        shape = self.click_obs[cand] + a_item * m_sub
        rate = self.click_exp[cand] + a_item
        gb = (self.cat_buy_obs.sum() + 5) / (self.cat_buy_exp.sum() + 5)
        cat = np.minimum(st.cat[cand], len(self.cat_buy_exp) - 1)
        m_buy = np.clip((self.cat_buy_obs[cat] + a_cat * gb) / (self.cat_buy_exp[cat] + a_cat) / gb, 0.8, 1.3)
        extra = m_buy * self._trend[cand] * np.where(self.boost_until[cand] > now_ms, 1.15, 1.0)
        mean = np.clip(np.clip(shape / rate / g, 0.3, 3.0) * extra, 0.25, 3.5)
        sample = np.clip(np.clip(rng.gamma(shape, 1.0 / rate) / g, 0.2, 4.0) * extra, 0.2, 4.5)
        return mean, sample
