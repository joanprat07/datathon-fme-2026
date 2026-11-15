"""Everything the recommender knows, in memory as numpy arrays, plus the feature code.

`State` is built from a snapshot downloaded by `twin.build`, then kept fresh by the live loop:
organic events update popularity, user profiles, co-visits and trends; catalogue updates change
prices and stock and add new products; our own feedback updates the bandit.

`featurize()` is the single place where features are computed, so training and live serving
cannot drift apart.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from twin.rt import ARTIFACTS

PAGES = {"home": 0, "product": 1, "category": 2}
DEVICES = {"web": 0, "ios": 1, "android": 2}
BUCKET_MS = 900_000  # 15 minutes
DAY_MS = 86_400_000


def to_ms(text: str) -> int:
    return int(datetime.fromisoformat(str(text)[:19]).replace(tzinfo=timezone.utc).timestamp() * 1000)


HISTORY_START_MS = to_ms("2026-09-15 07:00:00")

ITEM_FEATURES = ["i_cat", "i_sub", "i_logprice", "i_views_pd", "i_purch_pd", "i_conv", "i_cart_rate", "i_trend7",
                 "i_age", "i_car_lift", "i_car_pconv", "i_car_evidence", "i_price_rel"]
DYN_FEATURES = ["i_seg_pop", "i_seg_lift", "i_sub_trend"]
SESSION_FEATURES = ["s_in_rv", "s_is_last", "s_same_sub_last", "s_same_cat_last", "s_sub_share", "s_cat_share",
                    "s_vv_sum", "s_vv_last", "s_vv_max", "s_vp_sum", "s_vp_last", "s_price_ratio",
                    "s_brand_last", "s_brand_share"]
USER_FEATURES = ["u_known", "u_events", "u_purchases", "u_sub_view_share", "u_sub_buy_share", "u_sub_cart_share",
                 "u_cat_view_share", "u_cat_buy_share", "u_sub_views", "u_sub_buys", "u_sub_lift",
                 "u_bb_sum", "u_bb_max", "u_price_vs_views", "u_price_vs_buys",
                 # the shopper's own history with this exact product
                 "u_p_views", "u_p_carts", "u_p_view_share", "u_p_buys", "u_p_buy_days"]
CONTEXT_FEATURES = ["f_page", "f_device", "f_region", "f_slot", "f_n_recent", "f_hour"]
FEATURES = ITEM_FEATURES + DYN_FEATURES + SESSION_FEATURES + USER_FEATURES + CONTEXT_FEATURES
CATEGORICAL = ["i_cat", "i_sub", "f_page", "f_device", "f_region"]
NI, ND, NS, NU = len(ITEM_FEATURES), len(DYN_FEATURES), len(SESSION_FEATURES), len(USER_FEATURES)
D0, S0, U0, C0 = NI, NI + ND, NI + ND + NS, NI + ND + NS + NU
NF = len(FEATURES)

WORD = re.compile(r"[a-z]+")


class State:
    def __init__(self, snapshot: str = "live", spare_products: int = 2000):
        d = ARTIFACTS / snapshot
        meta = dict(line.split("=", 1) for line in (d / "meta.txt").read_text().splitlines() if "=" in line)
        self.cutoff_ms = to_ms(meta["cutoff"])
        self.now_ms = self.cutoff_ms
        self.days = (self.cutoff_ms - HISTORY_START_MS) / DAY_MS

        # ---- catalogue -------------------------------------------------------------------
        cat = pd.read_parquet(d / "catalog.parquet").sort_values("product_id").reset_index(drop=True)
        self.cat_names = sorted(cat.category.unique())
        self.sub_names = sorted(cat.subcategory.unique())
        self.brand_names = sorted(cat.brand.unique())
        self.cat_id = {n: i for i, n in enumerate(self.cat_names)}
        self.sub_id = {n: i for i, n in enumerate(self.sub_names)}
        self.brand_id = {n: i for i, n in enumerate(self.brand_names)}
        self.NC, self.SCAP = len(self.cat_names) + 5, len(self.sub_names) + 30
        self.sub_cat = np.zeros(self.SCAP, np.int64)

        cap = self.cap = len(cat) + spare_products
        self.P = 0
        self.pid: list[str] = []
        self.idx: dict[str, int] = {}
        self.tokens: list[frozenset] = []
        self.cat = np.zeros(cap, np.int64)
        self.sub = np.zeros(cap, np.int64)
        self.brand = np.zeros(cap, np.int64)
        self.price = np.ones(cap, np.float32)
        self.created = np.zeros(cap, np.float64)
        self.active = np.zeros(cap, bool)
        self.stock = np.zeros(cap, bool)
        self.is_new = np.zeros(cap, bool)       # added during the event (cold start)
        self.inherited = np.zeros(cap, bool)
        for r in cat.to_dict("records"):
            self._register(r)
        # products added during the event must exist before their statistics are loaded
        upd = d / "catalog_updates.parquet"
        updates = pd.read_parquet(upd).sort_values("ts_ms") if upd.exists() else None
        self.catalog_cursor = 0
        if updates is not None:
            for r in updates[~updates.product_id.isin(self.idx)].drop_duplicates("product_id").to_dict("records"):
                self._register(r, new=True)

        # ---- per-item counters -----------------------------------------------------------
        f32 = lambda: np.zeros(cap, np.float32)
        self.views, self.carts, self.purch, self.viewers = f32(), f32(), f32(), f32()
        self.views_7d, self.purch_7d = f32(), f32()
        self.car_exp, self.car_clicks, self.car_purch = f32(), f32(), f32()
        self.VV = np.zeros((cap, cap), np.float32)   # viewed together in a session
        self.VP = np.zeros((cap, cap), np.float32)   # viewed a, bought b in the session
        self.BB = np.zeros((cap, cap), np.float32)   # bought by the same user
        self.bandit_exp, self.bandit_clicks = f32(), f32()
        self.bandit_sub_exp = np.zeros(self.SCAP, np.float32)
        self.bandit_sub_clicks = np.zeros(self.SCAP, np.float32)

        st = pd.read_parquet(d / "item_stats.parquet")
        i = st.p.map(self.idx)
        st, i = st[i.notna()], i.dropna().astype(int).to_numpy()
        for col, arr in [("views", self.views), ("carts", self.carts), ("purchases", self.purch), ("viewers", self.viewers),
                         ("views_7d", self.views_7d), ("purch_7d", self.purch_7d)]:
            arr[i] = st[col].to_numpy()

        for name, M in [("vv", self.VV), ("vp", self.VP), ("bb", self.BB)]:
            df = pd.read_parquet(d / f"{name}.parquet")
            a, b = df.a.map(self.idx), df.b.map(self.idx)
            ok = a.notna() & b.notna()
            M[a[ok].astype(int).to_numpy(), b[ok].astype(int).to_numpy()] = df.n[ok].to_numpy()

        # old carousel: clicks against what the slot alone would give (removes position bias)
        cs = pd.read_parquet(d / "car_stats.parquet")
        base = cs.groupby(["page", "slot"])[["imp", "clicks"]].sum()
        base["ctr"] = base.clicks / base.imp
        cs = cs.join(base.ctr, on=["page", "slot"])
        cs["exp"] = cs.imp * cs.ctr
        agg = cs.groupby("p")[["exp", "clicks", "purch"]].sum()
        i = agg.index.map(self.idx)
        ok = ~pd.isna(i)
        ii = np.asarray(i[ok], dtype=int)
        self.car_exp[ii], self.car_clicks[ii], self.car_purch[ii] = agg.exp[ok], agg.clicks[ok], agg.purch[ok]
        self.car_buy_rate = float(agg.purch.sum() / max(agg.clicks.sum(), 1))

        # ---- segment popularity and subcategory trend -----------------------------------
        sp = pd.read_parquet(d / "seg_pop.parquet")
        self.region_names = sorted(sp.region.unique())
        self.region_id = {n: i for i, n in enumerate(self.region_names)}
        self.seg = np.zeros((3, len(self.region_names) + 5, cap), np.float32)
        i = sp.p.map(self.idx)
        ok = i.notna() & sp.device.isin(DEVICES)
        self.seg[sp.device[ok].map(DEVICES).to_numpy(), sp.region[ok].map(self.region_id).to_numpy(),
                 i[ok].astype(int).to_numpy()] = sp.purchases[ok].to_numpy()
        self.seg_tot = self.seg.sum(2)

        tr = pd.read_parquet(d / "sub_trend.parquet")
        self.b0 = HISTORY_START_MS // BUCKET_MS
        self.trend = np.zeros((int((self.cutoff_ms + 30 * DAY_MS) // BUCKET_MS - self.b0), self.SCAP), np.float32)
        s, b = tr["sub"].map(self.sub_id), tr.bucket * 1000 // BUCKET_MS - self.b0
        ok = s.notna() & (b >= 0) & (b < len(self.trend))
        self.trend[b[ok].to_numpy(), s[ok].astype(int).to_numpy()] = tr.views[ok].to_numpy()
        self._trend_cache: dict[int, np.ndarray] = {}

        # ---- users -------------------------------------------------------------------------
        us = pd.read_parquet(d / "user_stats.parquet")
        self.uidx: dict[str, int] = {u: i for i, u in enumerate(us.u)}
        self.ucap = len(us) + 50_000
        self.us_v = np.zeros((self.ucap, self.SCAP), np.float32)
        self.us_c = np.zeros((self.ucap, self.SCAP), np.float32)
        self.us_b = np.zeros((self.ucap, self.SCAP), np.float32)
        self.u_events = np.zeros(self.ucap, np.float32)
        self.u_vprice = np.zeros((self.ucap, 2), np.float64)   # sum, count of viewed prices
        self.u_bprice = np.zeros((self.ucap, 2), np.float64)   # sum, count of bought prices
        n = len(us)
        self.u_events[:n] = us.n_events
        self.u_vprice[:n, 0], self.u_vprice[:n, 1] = (us.view_price.fillna(0) * us.n_views), us.n_views
        self.u_bprice[:n, 0], self.u_bprice[:n, 1] = (us.purch_price.fillna(0) * us.n_purch), us.n_purch
        usub = pd.read_parquet(d / "user_sub.parquet")
        ui, si = usub.u.map(self.uidx), usub["sub"].map(self.sub_id)
        ok = ui.notna() & si.notna()
        ui, si = ui[ok].astype(int).to_numpy(), si[ok].astype(int).to_numpy()
        self.us_v[ui, si], self.us_c[ui, si], self.us_b[ui, si] = usub.v[ok], usub.c[ok], usub.b[ok]

        # every purchase of every user, organic and carousel: {user: {product: [timestamps, ascending]}}
        self.u_buys: dict[int, dict[int, list[int]]] = {}
        buys = pd.concat([pd.read_parquet(d / "user_purchases.parquet"), pd.read_parquet(d / "car_purchases.parquet")])
        buys = buys.sort_values("t")
        pi = buys.p.map(self.idx)
        buys, pi = buys[pi.notna()], pi.dropna().astype(int).to_numpy()
        for u, p, t in zip(buys.u.to_numpy(), pi, buys.t.to_numpy()):
            self.u_buys.setdefault(self.user(u), {}).setdefault(int(p), []).append(int(t))

        # views and carts of each product by each user, as one sorted table (CSR): rows of user i
        # are up_idx[up_ptr[i]:up_ptr[i+1]]. Live events go to the small `up_live` dictionary on top.
        up = pd.read_parquet(d / "user_product.parquet")
        umap = np.array([self.user(u) for u in up.u.cat.categories])
        pmap = np.array([self.idx.get(p, -1) for p in up.p.cat.categories])
        ui, pi = umap[up.u.cat.codes.to_numpy()], pmap[up.p.cat.codes.to_numpy()]
        keep = pi >= 0
        ui, pi = ui[keep], pi[keep]
        order = np.lexsort((pi, ui))
        self.up_idx = pi[order].astype(np.int32)
        self.up_v = up.v.to_numpy()[keep][order].astype(np.float32)
        self.up_c = up.c.to_numpy()[keep][order].astype(np.float32)
        self.up_ptr = np.searchsorted(ui[order], np.arange(len(self.uidx) + 1))
        self.up_live: dict[int, dict[int, list[float]]] = {}
        self._dv, self._dc = np.zeros(cap, np.float32), np.zeros(cap, np.float32)   # scratch space
        del up

        self.sessions: OrderedDict[str, list[int]] = OrderedDict()   # live sessions, for co-visits
        if updates is not None:  # replay prices, stock and new products in order
            for r in updates.to_dict("records"):
                self.apply_catalog_update(r)
            self.catalog_cursor = int(updates.ts_ms.max()) if len(updates) else 0
        self.refresh()

    # ------------------------------------------------------------------ catalogue
    def _register(self, r: dict, new: bool = False) -> int:
        """Add a product (or update one that exists) from a catalogue record."""
        p = str(r["product_id"])
        i = self.idx.get(p)
        if i is None:
            if self.P >= self.cap:
                raise RuntimeError("product capacity exhausted")
            i = self.P
            self.P += 1
            self.idx[p] = i
            self.pid.append(p)
            self.tokens.append(frozenset())
            self.is_new[i] = new
            self.stock[i] = True
        for names, ids, arr, key in [(self.cat_names, self.cat_id, self.cat, "category"),
                                     (self.sub_names, self.sub_id, self.sub, "subcategory"),
                                     (self.brand_names, self.brand_id, self.brand, "brand")]:
            v = str(r.get(key, ""))
            if v not in ids:
                ids[v] = len(names)
                names.append(v)
            arr[i] = ids[v]
        if self.sub[i] < self.SCAP:
            self.sub_cat[self.sub[i]] = self.cat[i]
        else:  # more new subcategories than we left room for: fold into the last slot
            self.sub[i] = self.SCAP - 1
        self.price[i] = max(float(r.get("price_eur") or 1.0), 0.01)
        self.active[i] = bool(r.get("is_active", True))
        if not self.inherited[i]:
            created = r.get("created_at")
            self.created[i] = to_ms(created) if created else self.now_ms
        self.tokens[i] = frozenset(WORD.findall(str(r.get("description", "")).lower()))
        return i

    def apply_catalog_update(self, r: dict) -> None:
        change = str(r.get("change", ""))
        i = self._register(r, new=True)   # `new` only applies if we had never seen the product
        if change == "new_product":
            self.is_new[i] = True
        if change == "out_of_stock":
            self.stock[i] = False
        elif change in ("back_in_stock", "new_product"):
            self.stock[i] = True
        if self.is_new[i] and not self.inherited[i]:
            self._inherit(i)
        if hasattr(self, "ok"):   # stock changes must take effect at once, not at the next refresh
            self.ok[i] = self.active[i] and self.stock[i]

    def _inherit(self, i: int, k: int = 5) -> None:
        """Cold start: a new product borrows the statistics of its closest existing products
        (same subcategory, then brand, price and description words)."""
        self.inherited[i] = True
        old = ~self.is_new[:self.P]
        pool = np.flatnonzero(old & (self.sub[:self.P] == self.sub[i]))
        if len(pool) == 0:
            pool = np.flatnonzero(old & (self.cat[:self.P] == self.cat[i]))
        if len(pool) == 0:
            return
        words = self.tokens[i]
        jac = np.array([len(words & self.tokens[j]) / max(len(words | self.tokens[j]), 1) for j in pool])
        sim = (self.brand[pool] == self.brand[i]) + np.exp(-np.abs(np.log(self.price[pool] / self.price[i]))) + jac
        nb = pool[np.argsort(-sim)[:k]]
        self.created[i] = self.created[nb].mean()   # borrowed counts cover the neighbours' lifetime
        for arr in (self.views, self.carts, self.purch, self.viewers, self.views_7d, self.purch_7d,
                    self.car_exp, self.car_clicks, self.car_purch):
            arr[i] += arr[nb].mean()
        self.seg[:, :, i] += self.seg[:, :, nb].mean(2)
        for M in (self.VV, self.VP, self.BB):
            link = float(M[nb].max(1).mean())
            M[i, :] += M[nb].mean(0)
            M[:, i] += M[:, nb].mean(1)
            M[i, nb] += link
            M[nb, i] += link
            M[i, i] = 0

    # ------------------------------------------------------------------ users
    def user(self, u: str) -> int:
        i = self.uidx.get(u)
        if i is None:
            i = len(self.uidx)
            if i >= self.ucap:
                grow = 50_000
                for name in ("us_v", "us_c", "us_b"):
                    setattr(self, name, np.vstack([getattr(self, name), np.zeros((grow, self.SCAP), np.float32)]))
                self.u_events = np.concatenate([self.u_events, np.zeros(grow, np.float32)])
                self.u_vprice = np.vstack([self.u_vprice, np.zeros((grow, 2))])
                self.u_bprice = np.vstack([self.u_bprice, np.zeros((grow, 2))])
                self.ucap += grow
            self.uidx[u] = i
        return i

    # ------------------------------------------------------------------ live updates
    def apply_event(self, ts: int, et: str, u: str, s: str, p: str, price: float, device: str, region: str) -> None:
        i = self.idx.get(p)
        if i is None:
            return
        self.now_ms = max(self.now_ms, ts)
        ui, sub = self.user(u), self.sub[i]
        self.u_events[ui] += 1
        if et == "view":
            self.views[i] += 1
            self.views_7d[i] += 1
            self.us_v[ui, sub] += 1
            self.u_vprice[ui] += (price, 1)
            self.up_live.setdefault(ui, {}).setdefault(i, [0.0, 0.0])[0] += 1
            b = ts // BUCKET_MS - self.b0
            if 0 <= b < len(self.trend):
                self.trend[b, sub] += 1
            seen = self.sessions.get(s)
            if seen is None:
                seen = self.sessions[s] = []
                if len(self.sessions) > 100_000:
                    self.sessions.popitem(last=False)
            if i not in seen:
                for a in seen:
                    self.VV[a, i] += 1
                    self.VV[i, a] += 1
                seen.append(i)
        elif et == "add_to_cart":
            self.carts[i] += 1
            self.us_c[ui, sub] += 1
            self.up_live.setdefault(ui, {}).setdefault(i, [0.0, 0.0])[1] += 1
        elif et == "purchase":
            self.purch[i] += 1
            self.purch_7d[i] += 1
            self.us_b[ui, sub] += 1
            self.u_bprice[ui] += (price, 1)
            d, r = DEVICES.get(device, -1), self.region_id.get(region, -1)
            if d >= 0 and r >= 0:
                self.seg[d, r, i] += 1
                self.seg_tot[d, r] += 1
            for a in self.sessions.get(s, ()):
                if a != i:
                    self.VP[a, i] += 1
            bought = self.u_buys.setdefault(ui, {})
            if i not in bought:
                for a in list(bought)[-40:]:
                    self.BB[a, i] += 1
                    self.BB[i, a] += 1
            bought.setdefault(i, []).append(ts)

    def mark_bought(self, ui: int, i: int, ts: int) -> None:
        """A purchase made through our carousel (those are not in the organic events)."""
        times = self.u_buys.setdefault(ui, {}).setdefault(i, [])
        if ts not in times:
            times.append(ts)

    def user_products(self, ui: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Products the user viewed or carted: (indices, views, carts). A product can appear twice
        (once from the snapshot, once from live events): add the two up."""
        if 0 <= ui < len(self.up_ptr) - 1:
            a, b = self.up_ptr[ui], self.up_ptr[ui + 1]
            ix, v, c = self.up_idx[a:b].astype(np.int64), self.up_v[a:b], self.up_c[a:b]
        else:
            ix, v, c = np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32)
        live = self.up_live.get(ui)
        if live:
            lv = np.array(list(live.values()), np.float32)
            ix = np.concatenate([ix, np.fromiter(live.keys(), np.int64, len(live))])
            v, c = np.concatenate([v, lv[:, 0]]), np.concatenate([c, lv[:, 1]])
        return ix, v, c

    def bandit_update(self, items: np.ndarray, expected: np.ndarray, clicked: int) -> None:
        """One served carousel: expected clicks per shown item against the click that happened."""
        self.bandit_exp[items] += expected
        np.add.at(self.bandit_sub_exp, self.sub[items], expected)
        if clicked >= 0:
            self.bandit_clicks[clicked] += 1
            self.bandit_sub_clicks[self.sub[clicked]] += 1

    def bandit(self, cand: np.ndarray, rng: np.random.Generator, a_item: float = 4.0, a_sub: float = 8.0):
        """Multipliers on the ranker's score: (posterior mean, Thompson sample).

        Each item's observed/expected click ratio is shrunk towards its subcategory's ratio, which
        is shrunk towards the overall ratio. Little evidence -> wide posterior -> gets explored."""
        g = (self.bandit_sub_clicks.sum() + 20) / (self.bandit_sub_exp.sum() + 20)   # overall calibration
        sub = self.sub[cand]
        m_sub = (self.bandit_sub_clicks[sub] + a_sub * g) / (self.bandit_sub_exp[sub] + a_sub)
        shape = self.bandit_clicks[cand] + a_item * m_sub
        rate = self.bandit_exp[cand] + a_item
        mean = np.clip(shape / rate / g, 0.3, 3.0)
        sample = np.clip(rng.gamma(shape, 1.0 / rate) / g, 0.2, 4.0)
        return mean, sample

    # ------------------------------------------------------------------ derived tables
    def refresh(self) -> None:
        """Recompute the per-item feature table and helper arrays. Cheap: call every minute live."""
        n = self.P
        self.days = max((self.now_ms - HISTORY_START_MS) / DAY_MS, 1.0)
        age = np.clip((self.now_ms - self.created[:n]) / DAY_MS, 7.0, None)
        alive = np.minimum(age, self.days)
        views, purch = self.views[:n], self.purch[:n]
        g_conv = purch.sum() / max(views.sum(), 1)
        g_cart = self.carts[:n].sum() / max(views.sum(), 1)
        sub = self.sub[:n]
        med = pd.Series(self.price[:n]).groupby(sub).median()
        sub_price = np.ones(self.SCAP, np.float32)
        sub_price[med.index.to_numpy()] = med.to_numpy()

        IF = np.zeros((self.cap, NI), np.float32)
        IF[:n, 0] = self.cat[:n]
        IF[:n, 1] = sub
        IF[:n, 2] = np.log(self.price[:n])
        IF[:n, 3] = np.log1p(views / alive)
        IF[:n, 4] = np.log1p(purch / alive)
        IF[:n, 5] = (purch + 50 * g_conv) / (views + 50)
        IF[:n, 6] = (self.carts[:n] + 50 * g_cart) / (views + 50)
        IF[:n, 7] = (self.views_7d[:n] / 7 + 1) / (views / alive + 1)
        IF[:n, 8] = np.log1p(age)
        IF[:n, 9] = (self.car_clicks[:n] + 5) / (self.car_exp[:n] + 5)
        IF[:n, 10] = (self.car_purch[:n] + 10 * self.car_buy_rate) / (self.car_clicks[:n] + 10)
        IF[:n, 11] = np.log1p(self.car_exp[:n])
        IF[:n, 12] = self.price[:n] / sub_price[sub]
        self.IF = IF

        self.nv = np.sqrt(self.views + 1)   # norms for the co-visit cosines
        self.nb = np.sqrt(self.purch + 1)
        self.tot_purch = float(purch.sum())
        sv = np.bincount(sub, weights=views, minlength=self.SCAP)
        self.sub_share = ((sv + 1) / (sv.sum() + self.SCAP)).astype(np.float32)
        self.ok = self.active & self.stock
        self.ok[n:] = False
        # expected purchases per day: the popularity used to pick candidates
        self.pop = np.zeros(self.cap, np.float32)
        self.pop[:n] = (purch + 50 * g_conv) / alive
        order = np.argsort(-self.pop[:n])
        self.pop_order = order
        self.sub_top = {s: order[sub[order] == s] for s in np.unique(sub)}

    def sub_trend(self, ts: int) -> np.ndarray:
        """Per subcategory: share of views in the last hour over its share in the 24 hours before."""
        b = int(ts // BUCKET_MS - self.b0)
        out = self._trend_cache.get(b)
        if out is None:
            b = min(max(b, 100), len(self.trend))
            last, base = self.trend[b - 4:b].sum(0), self.trend[b - 100:b - 4].sum(0)
            out = ((last + 1) / (last.sum() + self.SCAP)) / ((base + 1) / (base.sum() + self.SCAP))
            out = out.astype(np.float32)
            if len(self._trend_cache) > 5000:
                self._trend_cache.clear()
            self._trend_cache[int(ts // BUCKET_MS - self.b0)] = out
        return out

    def bought_before(self, ui: int, ts: int) -> tuple[list[int], list[int], list[int]]:
        """Purchases of the user before `ts`: (products, how many times, time of the last one)."""
        products, counts, last = [], [], []
        for p, times in (self.u_buys.get(ui) or {}).items():
            n = sum(1 for t in times if t < ts)
            if n:
                products.append(p)
                counts.append(n)
                last.append(max(t for t in times if t < ts))
        return products, counts, last

    # ------------------------------------------------------------------ features
    def featurize(self, ts: int, page: int, dev: int, reg: int, ui: int, rv: list[int],
                  cand: np.ndarray, slot) -> np.ndarray:
        """Feature matrix for candidate products `cand` in one request.

        ts: request time (ms). page/dev/reg: integer codes (-1 unknown). ui: user index or -1.
        rv: recently viewed product indices, oldest first. slot: carousel position(s), 1-10."""
        n = len(cand)
        X = np.zeros((n, NF), np.float32)
        X[:, :NI] = self.IF[cand]
        sub, cat, price, brand = self.sub[cand], self.cat[cand], self.price[cand], self.brand[cand]

        if dev >= 0 and reg >= 0:
            seg = self.seg[dev, reg, cand]
            frac = self.seg_tot[dev, reg] / max(self.tot_purch, 1.0)
            X[:, D0] = np.log1p(seg / self.days)
            X[:, D0 + 1] = (seg + 3) / (self.purch[cand] * frac + 3)
        else:
            X[:, D0 + 1] = 1
        X[:, D0 + 2] = self.sub_trend(ts)[sub]

        if rv:
            r = np.asarray(rv)
            rsub, rcat, rbrand = self.sub[r], self.cat[r], self.brand[r]
            X[:, S0] = (cand[None, :] == r[:, None]).sum(0)
            X[:, S0 + 1] = cand == r[-1]
            X[:, S0 + 2] = sub == rsub[-1]
            X[:, S0 + 3] = cat == rcat[-1]
            X[:, S0 + 4] = (sub[None, :] == rsub[:, None]).mean(0)
            X[:, S0 + 5] = (cat[None, :] == rcat[:, None]).mean(0)
            vv = self.VV[np.ix_(r, cand)] / (self.nv[r][:, None] * self.nv[cand][None, :])
            X[:, S0 + 6], X[:, S0 + 7], X[:, S0 + 8] = vv.sum(0), vv[-1], vv.max(0)
            vp = self.VP[np.ix_(r, cand)] / (self.nv[r][:, None] * self.nb[cand][None, :])
            X[:, S0 + 9], X[:, S0 + 10] = vp.sum(0), vp[-1]
            X[:, S0 + 11] = np.log(price / self.price[r].mean())
            X[:, S0 + 12] = brand == rbrand[-1]
            X[:, S0 + 13] = (brand[None, :] == rbrand[:, None]).mean(0)
        else:
            X[:, S0 + 11] = np.nan

        X[:, U0 + 13:U0 + 15] = np.nan
        X[:, U0 + 19] = np.nan
        if ui >= 0 and self.u_events[ui] > 0:
            v, c, b = self.us_v[ui], self.us_c[ui], self.us_b[ui]
            tv, tc, tb = max(v.sum(), 1.0), max(c.sum(), 1.0), max(b.sum(), 1.0)
            X[:, U0] = 1
            X[:, U0 + 1] = math.log1p(self.u_events[ui])
            X[:, U0 + 2] = math.log1p(b.sum())
            X[:, U0 + 3] = v[sub] / tv
            X[:, U0 + 4] = b[sub] / tb
            X[:, U0 + 5] = c[sub] / tc
            X[:, U0 + 6] = np.bincount(self.sub_cat, weights=v, minlength=self.NC)[cat] / tv
            X[:, U0 + 7] = np.bincount(self.sub_cat, weights=b, minlength=self.NC)[cat] / tb
            X[:, U0 + 8] = np.log1p(v[sub])
            X[:, U0 + 9] = np.log1p(b[sub])
            X[:, U0 + 10] = (v[sub] / tv) / self.sub_share[sub]
            if self.u_vprice[ui, 1] > 0:
                X[:, U0 + 13] = np.log(price / (self.u_vprice[ui, 0] / self.u_vprice[ui, 1]))
            if self.u_bprice[ui, 1] > 0:
                X[:, U0 + 14] = np.log(price / (self.u_bprice[ui, 0] / self.u_bprice[ui, 1]))
        if ui >= 0:
            dv, dc = self._dv, self._dc
            ix, v, c = self.user_products(ui)
            if len(ix):
                np.add.at(dv, ix, v)
                np.add.at(dc, ix, c)
                X[:, U0 + 15] = np.log1p(dv[cand])
                X[:, U0 + 16] = dc[cand]
                X[:, U0 + 17] = dv[cand] / max(float(v.sum()), 1.0)
                dv[ix] = 0
                dc[ix] = 0
            past, counts, last = self.bought_before(ui, ts)
            if past:
                pa = np.asarray(past)
                dv[pa] = counts
                dc[pa] = (ts - np.asarray(last)) / DAY_MS
                n_buys = dv[cand]
                X[:, U0 + 18] = n_buys
                X[:, U0 + 19] = np.where(n_buys > 0, dc[cand], np.nan)
                dv[pa] = 0
                dc[pa] = 0
                pa = pa[-40:]
                bb = self.BB[np.ix_(pa, cand)] / (self.nb[pa][:, None] * self.nb[cand][None, :])
                X[:, U0 + 11], X[:, U0 + 12] = bb.sum(0), bb.max(0)

        X[:, C0] = page
        X[:, C0 + 1] = dev
        X[:, C0 + 2] = reg
        X[:, C0 + 3] = slot
        X[:, C0 + 4] = len(rv)
        X[:, C0 + 5] = (ts // 3_600_000) % 24
        return X
