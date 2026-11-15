"""Model variants that can run side by side in the live loop, to compare them on real traffic.

    TWIN_ARMS="twin-v1:60,noexp:20,price0:20" python -m twin.live

Each arm is one `Variant`. A request is assigned to an arm by a stable hash of its request_id and
the arm's name is written as `model_version`, so the leaderboard (Beta view, open the team row)
shows each one's conversion. Without arms the loop runs exactly as before (one arm, the base model).

Arms can be changed WITHOUT restarting: write them in config/twin_arms.txt (one "name: share" per
line); the loop re-reads the file every 15 seconds. A restart costs minutes of fallback answers.

Single-setting variants, to find out one by one what helps (join with '+' to combine winners, and
give the combination its own name with '=':  "twin-v2=noexp+price0: 50"):

    noexp       no exploration: all 10 slots by expected value (base: slots 8-10 are sampled)
    price0      no preference for expensive products (base tilts the score up to ~10% by price)
    cap3 cap4 cap7   at most N products per subcategory (base: 5)
    nobandit    no correction from our own feedback
    buy15 buy07 weight of p(purchase | click) in the score (base: 1)
    fb          the ranker retrained with our own feedback (python -m twin.train_feedback)

Tested live on Sat 14 Nov 19:05-19:20: "m2-intent+m3-live" converted 1.44% against 1.60% for the
base model on the same minutes (about 1,000 requests). Not recommended; kept for reference.

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
from pathlib import Path
import zlib
from dataclasses import dataclass, replace

import numpy as np

from twin.rt import ARTIFACTS
from twin.state import State, to_ms

FINAL_START_MS = to_ms("2026-11-15 09:00:00")     # Sunday 10:00 CET: the stream switches to new data
EXAM = (1.0 / (1.0 + 0.35 * np.arange(10))).astype(np.float64)   # how much each slot is looked at (slot 1 = 1)
LN2 = float(np.log(2.0))
REL_K = 10.0          # weighted events (view 1, purchase 5) in the fast window at which a product's trend counts half


@dataclass(frozen=True)
class Variant:
    name: str = "twin-v1"
    model: str = "model"          # folder under artifacts/ with click.txt, buy.txt, meta.json
    block_bought: bool = True     # never recommend a product the shopper already bought
    slate: str = "cap"            # "cap": at most MAX_PER_SUB per subcategory | "intent": diminishing returns
    intent_level: str = "cat"     # "cat" or "sub": what counts as one intent
    lam: float = 1.0              # strength of the intent effect (0 = plain sort)
    live_layer: bool = False      # use LiveLayer multipliers instead of the base bandit
    exploit_slots: int = 7        # slots filled by expected value; the rest are Thompson samples (10 = no exploration)
    max_per_sub: int = 5          # at most this many products of one subcategory
    bandit: bool = True           # correct the score with observed/expected clicks of our own carousels
    buy_power: float = 1.0        # score = p_click * p_buy ** buy_power
    price_eps: float = 0.0005     # score *= 1 + price_eps * price (revenue is only the tie-breaker)
    wide: bool = False            # wider candidate pool: see candidates() and widen() in twin/recommend.py
    cold_model: str = ""          # a second ranker used ONLY for products without history (see cold_blend in twin/recommend.py)
    new_mult: float = 1.0         # score *= new_mult for products added during the event (live: they took ~15% of our
                                  # slots with a third of the CTR of the rest, 02:30-03:30 UTC; see novetats.py)
    trend_power: float = 1.0      # weight of the newest information: live-layer trend (last ~20 min vs ~6 h) ** trend_power (0 = ignore it)
    trend_rel: bool = False       # powers above 1 only in proportion to the evidence behind each product's trend (see LiveLayer.multipliers)
    engine: str = "twin"          # "conv": requests of this arm are answered by the teammate's conv recommender (twin/conv_arm.py)
    trend_src: str = "live"       # "live": trend from the decayed counters of the loop | "sql": trend computed in RawTree (see TrendSQL in twin/live.py)
                                  # | "seed": "live" plus a prior for products added after start (see LiveLayer.events)


VARIANTS = {
    "m1-rebuy": Variant("m1-rebuy", model="model_rebuy", block_bought=False),
    "m2-intent": Variant("m2-intent", slate="intent"),
    "m3-live": Variant("m3-live", live_layer=True),
    "noexp": Variant("noexp", exploit_slots=10),
    "price0": Variant("price0", price_eps=0.0),
    "cap3": Variant("cap3", max_per_sub=3),
    "cap4": Variant("cap4", max_per_sub=4),
    "cap7": Variant("cap7", max_per_sub=7),
    "nobandit": Variant("nobandit", bandit=False),
    "buy15": Variant("buy15", buy_power=1.5),
    "buy07": Variant("buy07", buy_power=0.7),
    "fb": Variant("fb", model="model_fb"),
    "v5-hybrid": Variant("v5-hybrid", model="model_v5"),   # twin-v1 + the 4 hybrid variables (twin/hybrid_live.py)
    "wide": Variant("wide", wide=True),                    # more candidates: thin profiles, recency-weighted views, price drops, new products
    "v6-fb": Variant("v6-fb", model="model_v6"),          # V5 retrained with our own feedback (twin/train_v6.py)
    # V6 as served: the v6-fb ranker + intent slate + live layer + no exploration slots (all 10 by expected value)
    "v6": Variant("v6", model="model_v6", slate="intent", live_layer=True, exploit_slots=10),
    # v7 family (twin/train_v7.py): v5 + new-product copies (+ propensity weighting)
    "v7": Variant("v7", model="model_v7"),
    "v7c": Variant("v7c", model="model_v7c"),          # new-product copies, no propensity
    "v7s": Variant("v7s", model="model_v7s"),          # copies + soft propensity (power 0.5, cap 5)
    "v7base": Variant("v7base", model="model_v7base"),  # control: v5 recipe
    "v8": Variant("v8", model="model_v7s"),            # = v7s, served alone (worse than v5 live on known products)
    # cold8: keep the arm's ranker for every product with history; products added during the event
    # (no history) are scored by model_v7s, which learned them from the new-product copies
    "cold8": Variant("cold8", cold_model="model_v7s"),
    # V9 = the best of the three: V6 as served (feedback-retrained ranker, intent slate, live layer,
    # no exploration slots) + v8 for products without history only
    "v9": Variant("v9", model="model_v6", slate="intent", live_layer=True, exploit_slots=10, cold_model="model_v7s"),
    # strength of the intent slate (m2-intent): 0.5 = closer to a plain sort, 2 = more variety across categories
    "lam05": Variant("lam05", lam=0.5),
    "lam2": Variant("lam2", lam=2.0),
    # products added during the event: score halved / quartered (they reach the slate too often, see new_mult)
    "new05": Variant("new05", new_mult=0.5),
    "new025": Variant("new025", new_mult=0.25),
    # weight of the newest information (needs m3-live): the per-product trend multiplier is 0.8-1.4 by default;
    # trend2 squares it (0.64-1.96), so what is rising in the last ~20 minutes counts more; trend0 switches it off
    "trend2": Variant("trend2", trend_power=2.0),
    "trend0": Variant("trend0", trend_power=0.0),
    "trend3": Variant("trend3", trend_power=3.0),
    # the trend computed by RawTree every ~90 s over the real last 20 min / 6 h windows (twin/live.py TrendSQL): it
    # does not start from zero after a restart, nor for a product that appeared during the event (the decayed
    # counters of the loop start at 0 for those, so any activity gives them the maximum boost)
    "sqltrend": Variant("sqltrend", trend_src="sql"),
    # trend2/trend3 raise the trend of EVERY product to a power, also of products with a handful of events, where
    # the trend is mostly noise (offline, a noisy trend costs more the higher the power: twin/exp_final.py --fast-h).
    # trendrel keeps power 1 for those and reaches the full power only for products with enough events:
    # exponent = 1 + (power - 1) * e / (e + 10), e = weighted events expected in the fast window from the slow rate
    "trendrel": Variant("trendrel", trend_rel=True),
    # the teammate's conv recommender (branch conv-v1), served as one arm; model_v5 only passes the model-folder check
    "pissarra": Variant("pissarra", model="model_v5", engine="conv"),
    # the loop trend with a prior for products added after the loop started (LiveLayer.events): a new product is
    # not "rising" only because its counters start at zero. Same as "live" for every product that already existed.
    "seedtrend": Variant("seedtrend", trend_src="seed"),
}
# TWIN_ARMS_FILE points a dry run at its own arms file, so it can test new arms while the live loop keeps the real one
ARMS_FILE = Path(os.environ["TWIN_ARMS_FILE"]) if os.environ.get("TWIN_ARMS_FILE") else ARTIFACTS.parent / "config" / "twin_arms.txt"


def arm_models(arms) -> set[str]:
    """Every model folder the arms need: their ranker and, if any, their ranker for new products."""
    return {m for v, _ in arms for m in (v.model, v.cold_model) if m}


def get_variant(name: str) -> Variant:
    """A registered variant, a combination joined with '+' ('noexp+price0': both changes at once),
    'label=recipe' to run a recipe under a name of our choice ('twin-v2=noexp+price0'),
    or, for any other name, the base model under that name."""
    label, _, recipe = name.partition("=")
    v, base = Variant(label.strip()), Variant()
    for part in (recipe or label).split("+"):
        part = part.strip()
        known = VARIANTS.get(part)
        if known is not None:   # take every setting this variant changes
            changed = {f: getattr(known, f) for f in base.__dataclass_fields__ if f != "name" and getattr(known, f) != getattr(base, f)}
            v = replace(v, **changed)
    return v


def parse_arms(spec: str, default_name: str, strict: bool = False) -> list[tuple[Variant, float]]:
    """'twin-v1:60,m1-rebuy:20' -> [(Variant, share)]. Unknown names run the base model under that name.
    Arms whose model folder does not exist are dropped (their traffic goes to the others)."""
    arms: list[tuple[Variant, float]] = []
    for part in [p for p in spec.split(",") if p.strip()]:
        name, _, weight = part.strip().rpartition(":") if ":" in part else (part.strip(), "", "1")
        try:
            v, w = get_variant(name.strip()), float(weight)
        except Exception as e:
            print(f"arm '{part.strip()}' skipped: {type(e).__name__}: {e}")
            continue
        if not (ARTIFACTS / v.model / "meta.json").exists():
            print(f"arm '{v.name}' skipped: artifacts/{v.model}/ does not exist")
            continue
        if v.cold_model and not (ARTIFACTS / v.cold_model / "meta.json").exists():
            print(f"arm '{v.name}' skipped: artifacts/{v.cold_model}/ (its model for new products) does not exist")
            continue
        if w > 0:
            arms.append((v, w))
    if not arms:
        if strict:      # the caller keeps what it had
            return []
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


def read_arms_file(path=None) -> tuple[str, float | None] | None:
    """config/twin_arms.txt -> (arms spec, holdout share or None). None if there is no file.

        # comments and blank lines are ignored
        twin-v1: 60
        noexp: 20
        twin-v2=noexp+price0: 20
        holdout: 5          # percent of requests answered with the baseline on purpose (optional)
    """
    path = path or ARMS_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    parts, holdout = [], None
    for line in text.replace(",", "\n").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, weight = line.rpartition(":") if ":" in line else (line, "", "1")
        if name.strip().lower() == "holdout":
            try:
                holdout = min(max(float(weight) / 100.0, 0.0), 0.5)
            except ValueError:
                pass
        else:
            parts.append(f"{name.strip()}:{weight.strip()}")
    return ",".join(parts), holdout


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
        # "seeded" trend (trend_src="seed"): the same counters, plus a prior for every product that appears AFTER
        # this point (new_product during the event). Without it such a product starts with fast = slow = 0: the
        # fast counter fills in minutes, the slow one in hours, so for hours it looks "rising" only for being new
        # (~+17% after an hour at an average rate, up to the 1.4 cap if busier; squared by trend2), although new
        # products convert worse (see novetats.py). The prior is its inherited 7-day rate (State._inherit), the
        # same start __init__ gives every product that already exists. tests/test_trend.py checks both.
        self.seed_fast, self.seed_slow = np.zeros(cap, np.float64), np.zeros(cap, np.float64)
        self.n_seeded = st.P
        self._trend_seed = np.ones(cap, np.float32)
        self.boost_until = f64(cap)
        self.final_done = st.now_ms >= FINAL_START_MS
        self._trend = np.ones(cap, np.float32)
        self._trend_sql = np.ones(cap, np.float32)   # filled by sql_trend(); used by variants with trend_src="sql"
        self._rel = np.zeros(cap, np.float32)        # evidence behind _trend, 0..1 (variant trendrel)
        self._rel_sql = np.zeros(cap, np.float32)    # the same for _trend_sql
        self.sql_t = 0.0                             # when it was last filled (organisers' clock, ms)

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

    def restore(self, st: State, rows: list[dict], now_ms: float) -> int:
        """After a restart: the click evidence of our recent carousels, already decayed to `now_ms`
        (and cut to a quarter if it predates the final while the final is on) by LAYER_RESTORE in
        twin/live.py. Purchases per category are not restored: they start neutral (clipped to 0.8-1.3)."""
        n = 0
        for r in rows:
            i = st.idx.get(r["p"])
            if i is None:
                continue
            e, c = float(r["expected"]), float(r["clicks"])
            self.click_exp[i] += e
            self.click_obs[i] += c
            self.sub_exp[st.sub[i]] += e
            self.sub_obs[st.sub[i]] += c
            n += 1
        self.t_bandit = float(now_ms)            # the rows are decayed to now: do not decay them again
        self.final_done = now_ms >= FINAL_START_MS
        return n

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
        gf, gs = 0.5 ** (dt / self.fast_ms), 0.5 ** (dt / self.slow_ms)
        self.fast *= gf
        self.slow *= gs
        self.seed_fast *= gf
        self.seed_slow *= gs
        self.t_trend = max(now, self.t_trend)
        if st.P > self.n_seeded:          # products registered since the last batch: give them their prior
            new = np.arange(self.n_seeded, min(st.P, len(self.seed_fast)))
            rate = (st.views_7d[new] + 5 * st.purch_7d[new]).astype(np.float64) / (7 * 86_400_000)
            self.seed_fast[new] += rate * self.fast_ms / LN2
            self.seed_slow[new] += rate * self.slow_ms / LN2
            self.n_seeded = st.P
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
        sf = rate_f + self.seed_fast[:n] * LN2 / self.fast_ms
        ss = rate_s + self.seed_slow[:n] * LN2 / self.slow_ms
        seeded = np.ones(len(self._trend_seed), np.float32)
        seeded[:n] = np.clip(((sf + a) / (ss + a)) ** 0.3, 0.8, 1.4)
        self._trend_seed = seeded
        rel = np.zeros(len(self._rel), np.float32)
        e = rate_s * self.fast_ms                   # weighted events expected in the fast window
        rel[:n] = e / (e + REL_K)
        self._rel = rel

    def sql_trend(self, st: State, rows: list[dict], now_ms: float, fast_min: float = 20.0, slow_min: float = 360.0) -> dict:
        """Trend per product from RawTree (rows of TREND_SQL in twin/live.py: p, f = weighted events of the last
        `fast_min` minutes, s = of the last `slow_min`, first = time of its first event in that window, ms).
        Same formula as events(), but on real windows: rate = events per minute; the slow rate of a product added
        during the event is divided by the minutes since its first event (at least `fast_min`), not by 6 h, so
        a product that has just appeared is not 'trending' only because it is new. Products with no events -> 1."""
        n = st.P
        rf, rs = np.zeros(n), np.zeros(n)
        for r in rows:
            i = st.idx.get(r["p"])
            if i is None or i >= n:
                continue
            span = slow_min
            if st.is_new[i]:
                span = min(max((now_ms - float(r["first"])) / 60_000.0, fast_min), slow_min)
            rf[i] = float(r["f"]) / fast_min
            rs[i] = float(r["s"]) / span
        a = max(float(rs.mean()), 1e-12)
        trend = np.ones(len(self._trend_sql), np.float32)
        trend[:n] = np.clip(((rf + a) / (rs + a)) ** 0.3, 0.8, 1.4)
        self._trend_sql = trend      # new array, swapped in one go
        rel = np.zeros(len(self._rel_sql), np.float32)
        e = rs * fast_min
        rel[:n] = e / (e + REL_K)
        self._rel_sql = rel
        self.sql_t = float(now_ms)
        up = trend[:n]
        return {"products": len(rows), "rising": int((up > 1.05).sum()), "falling": int((up < 0.95).sum()),
                "max": float(up.max()) if n else 1.0}

    def sql_fresh(self, now_ms: float) -> bool:
        """The RawTree trend is usable: filled less than 5 minutes ago."""
        return self.sql_t > 0 and now_ms - self.sql_t < 300_000

    def price_change(self, i: int, old_price: float, new_price: float, now_ms: float) -> None:
        if old_price > 0 and new_price < 0.9 * old_price:
            self.boost_until[i] = now_ms + 3_600_000

    # ---- multipliers -----------------------------------------------------------------------------
    def multipliers(self, st: State, cand: np.ndarray, rng: np.random.Generator, now_ms: float,
                    a_item: float = 4.0, a_sub: float = 8.0, a_cat: float = 10.0,
                    trend_power: float = 1.0, trend_src: str = "live", trend_rel: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(posterior mean, Thompson sample) multipliers for the ranker's score of each candidate."""
        g = (self.sub_obs.sum() + 20) / (self.sub_exp.sum() + 20)
        sub = st.sub[cand]
        m_sub = (self.sub_obs[sub] + a_sub * g) / (self.sub_exp[sub] + a_sub)
        shape = self.click_obs[cand] + a_item * m_sub
        rate = self.click_exp[cand] + a_item
        gb = (self.cat_buy_obs.sum() + 5) / (self.cat_buy_exp.sum() + 5)
        cat = np.minimum(st.cat[cand], len(self.cat_buy_exp) - 1)
        m_buy = np.clip((self.cat_buy_obs[cat] + a_cat * gb) / (self.cat_buy_exp[cat] + a_cat) / gb, 0.8, 1.3)
        use_sql = trend_src == "sql" and self.sql_fresh(now_ms)                                    # stale SQL -> loop trend
        if trend_src == "seed":
            base = self._trend_seed                                   # "live" + prior for products added after start
        else:
            base = self._trend_sql if use_sql else self._trend
        if trend_power == 1.0:
            trend = base[cand]
        elif trend_rel and trend_power > 1.0:
            rel = (self._rel_sql if use_sql else self._rel)[cand].astype(np.float64)
            trend = base[cand].astype(np.float64) ** (1.0 + (trend_power - 1.0) * rel)
        else:
            trend = base[cand].astype(np.float64) ** trend_power
        extra = m_buy * trend * np.where(self.boost_until[cand] > now_ms, 1.15, 1.0)
        mean = np.clip(np.clip(shape / rate / g, 0.3, 3.0) * extra, 0.25, 3.5)
        sample = np.clip(np.clip(rng.gamma(shape, 1.0 / rate) / g, 0.2, 4.0) * extra, 0.2, 4.5)
        return mean, sample
