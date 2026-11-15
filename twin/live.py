"""The live loop: read requests, answer inside the 3-second deadline, keep learning.

    python -m twin.live           # answer live traffic (only ONE copy per team may run)
    python -m twin.live --dry     # everything except sending answers: a safe test

Settings (environment variables):
    TWIN_VERSION   model_version written with each answer        (default twin-v1)
    TWIN_HOLDOUT   share of requests answered with the baseline, to measure the model's lift live
                   (default 0 = none; use e.g. 0.1 during practice, never during the final)
    TWIN_ARMS      model variants to compare on live traffic, e.g. "twin-v1:60,noexp:20,price0:20"
                   (see twin/variants.py; default: one arm, the base model, named TWIN_VERSION)

To change the arms or the holdout WITHOUT restarting, write them in config/twin_arms.txt: the loop
re-reads it every 15 seconds and it takes precedence over the two variables above.
Every answer's features are saved to artifacts/declog/ (twin/declog.py) for `python -m twin.train_feedback`.

The main thread only answers requests. Background threads download the organic events, the
catalogue updates and our feedback; the main thread applies them between two batches, so the
model always works with fresh popularity, user profiles, stock, prices and bandit statistics.
Any error while scoring a request falls back to the best sellers in stock: always a valid answer.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from twin.declog import DecLog
from twin.recommend import Ranker, fallback, recommend
from twin.rt import API, ARTIFACTS, TEAM
from twin.state import NF, State
from twin.variants import (ARMS_FILE, FINAL_START_MS, LiveLayer, arms_from_env, parse_arms, pick_arm,
                           read_arms_file)

VERSION = os.environ.get("TWIN_VERSION", "twin-v1")
HOLDOUT = float(os.environ.get("TWIN_HOLDOUT", "0"))
POLL = 0.25            # seconds between two reads of shop.requests
OVERLAP_MS = 1500      # re-read this much behind the newest request seen: rows can land out of order
TOO_OLD_MS = 2200      # a request this old has already been served by the fallback

REQUESTS = """
SELECT request_id::String AS request_id, toUnixTimestamp64Milli(ts::DateTime64(3)) AS t, user_id::String AS user_id,
       `context.device`::String AS device, `context.page`::String AS page, `context.region`::String AS region,
       arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv
FROM shop.requests WHERE ts::DateTime64(3) > fromUnixTimestamp64Milli({cursor}) ORDER BY ts LIMIT 1000"""
EVENTS = """
SELECT toUnixTimestamp64Milli(ts::DateTime64(3)) AS t, event_type::String AS et, user_id::String AS u, session_id::String AS s,
       product_id::String AS p, price_eur::Float64 AS price, device::String AS device, region::String AS region
FROM shop.events WHERE ts::DateTime64(3) > fromUnixTimestamp64Milli({cursor}) ORDER BY ts LIMIT {limit}"""
CATALOG = """
SELECT *, toUnixTimestamp64Milli(ts::DateTime64(3)) AS t
FROM shop.catalog_updates WHERE ts::DateTime64(3) > fromUnixTimestamp64Milli({cursor}) ORDER BY ts LIMIT {limit}"""
FEEDBACK = """
SELECT request_id::String AS request_id, toUnixTimestamp64Milli(ts::DateTime64(3)) AS t, served_by::String AS served_by,
       clicked::String AS clicked, purchased::String AS purchased
FROM {team}.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) > fromUnixTimestamp64Milli({cursor})
ORDER BY ts LIMIT {limit}"""
BANDIT_RESTORE = """
SELECT s.p AS p, sum(s.e) AS expected, countIf(f.clicked = s.p) AS clicks FROM (
    SELECT rid, pe.1 AS p, pe.2 AS e FROM (
        SELECT request_id::String AS rid, arrayJoin(arrayZip(items::Array(String), expected::Array(Float64))) AS pe
        FROM {team}.scratch WHERE kind::String = 'twin_pred')) AS s
INNER JOIN (SELECT request_id::String AS rid, clicked::String AS clicked FROM {team}.feedback
            WHERE served_by::String = 'team') AS f ON f.rid = s.rid
GROUP BY p"""
LAYER_RESTORE = """
SELECT s.p AS p, sum(s.e * s.w) AS expected, sumIf(s.w, f.clicked = s.p) AS clicks FROM (
    SELECT rid, w, pe.1 AS p, pe.2 AS e FROM (
        SELECT request_id::String AS rid,
               exp2(-({now} - t::Int64) / {hl}) * if({now} >= {final} AND t::Int64 < {final}, 0.25, 1) AS w,
               arrayJoin(arrayZip(items::Array(String), expected::Array(Float64))) AS pe
        FROM {team}.scratch WHERE kind::String = 'twin_pred' AND t::Int64 > {now} - {window})) AS s
INNER JOIN (SELECT request_id::String AS rid, any(clicked::String) AS clicked FROM {team}.feedback
            WHERE served_by::String = 'team' GROUP BY rid) AS f ON f.rid = s.rid
GROUP BY p"""
BOUGHT_RESTORE = """
SELECT r.user_id::String AS u, f.purchased::String AS p, toUnixTimestamp64Milli(f.ts::DateTime64(3)) AS t
FROM {team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.purchased::String != ''"""


def new_http() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
    return s


def sql(http: requests.Session, text: str, timeout=(2, 4)) -> list[dict]:
    r = http.post(f"{API}/v1/query", json={"sql": text}, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]


_pool = ThreadPoolExecutor(max_workers=8)


def bounded(seconds: float, fn, *args, **kwargs):
    """Run a network call with a hard limit on total time. The timeouts of `requests` only bound
    each read, so a connection that trickles or stalls could otherwise freeze the loop for minutes."""
    return _pool.submit(fn, *args, **kwargs).result(timeout=seconds)


class Feed(threading.Thread):
    """Follows one append-only table in the background and queues its new rows for the main loop."""

    def __init__(self, name: str, template: str, cursor: int, interval: float, limit: int, out: queue.Queue):
        super().__init__(daemon=True, name=name)
        self.template, self.cursor, self.interval, self.limit, self.out = template, int(cursor), interval, limit, out
        self.errors = 0

    def run(self) -> None:
        http = new_http()
        while True:
            try:
                rows = sql(http, self.template.format(cursor=self.cursor, limit=self.limit, team=TEAM), timeout=(3, 25))
                if rows:
                    self.cursor = max(int(r["t"]) for r in rows)
                    self.out.put((self.name, rows))
                if len(rows) < self.limit:
                    time.sleep(self.interval)
            except Exception:
                self.errors += 1
                http = new_http()
                time.sleep(2)


class Writer(threading.Thread):
    """Saves our predictions to team_NN.scratch (for the bandit after a restart) without slowing answers."""

    def __init__(self):
        super().__init__(daemon=True, name="writer")
        self.q: queue.Queue = queue.Queue()

    def run(self) -> None:
        http = new_http()
        while True:
            batch = [self.q.get()]
            time.sleep(2)
            while not self.q.empty() and len(batch) < 2000:
                batch.append(self.q.get_nowait())
            try:
                http.post(f"{API}/v1/tables/scratch", params={"database": TEAM}, json=batch, timeout=(3, 15)).raise_for_status()
            except Exception:
                http = new_http()


class Clock(threading.Thread):
    """The organisers' clock: our clock plus an offset measured against the server, and measured
    again every minute. Measuring it once was fragile: this laptop runs ~1.7 s behind the server, so
    if Windows corrected its clock mid-run every request would look 1.7 s older and most would be
    skipped as "too old". Each measurement keeps the fastest of 3 round trips and takes the server
    time as the middle of it."""

    def __init__(self, interval: float = 60.0):
        super().__init__(daemon=True, name="clock")
        self.interval, self.errors = interval, 0
        self.offset = self.measure(new_http())          # raises if the server cannot be reached at all

    @staticmethod
    def measure(http: requests.Session) -> int:
        best = None
        for _ in range(3):
            try:
                t0 = time.time()
                server = int(sql(http, "SELECT toUnixTimestamp64Milli(now64(3)) AS t", timeout=(2, 3))[0]["t"])
                t1 = time.time()
            except Exception:
                continue
            if best is None or t1 - t0 < best[0]:
                best = (t1 - t0, server - int((t0 + t1) * 500))
        if best is None:
            raise RuntimeError("server clock unreachable")
        return best[1]

    def now(self) -> int:
        return int(time.time() * 1000) + self.offset

    def run(self) -> None:
        http = new_http()
        while True:
            time.sleep(self.interval)
            try:
                new = self.measure(http)
            except Exception:
                self.errors += 1
                http = new_http()
                continue
            if abs(new - self.offset) > 250:
                print(f"{time.strftime('%H:%M:%S')} server clock offset changed: {self.offset} -> {new} ms", flush=True)
            self.offset = new


def restore(st: State, http: requests.Session, layer: LiveLayer, now_ms: int) -> None:
    """After a restart: bandit statistics, the m3 live layer and carousel purchases from what RawTree already holds."""
    try:
        rows = sql(http, LAYER_RESTORE.format(team=TEAM, now=int(now_ms), hl=int(layer.hl_ms), final=FINAL_START_MS,
                                              window=int(6 * layer.hl_ms)), timeout=(3, 25))
        n = layer.restore(st, rows, now_ms)
        print(f"live layer restored for {n} products ({layer.click_obs.sum():.0f} clicks vs "
              f"{layer.click_exp.sum():.0f} expected, decayed)")
    except Exception as e:
        print(f"live layer starts empty ({str(e)[:120]})")
    try:
        n = 0
        for r in sql(http, BANDIT_RESTORE.format(team=TEAM), timeout=(3, 25)):
            i = st.idx.get(r["p"])
            if i is not None:
                st.bandit_exp[i] += float(r["expected"])
                st.bandit_clicks[i] += float(r["clicks"])
                st.bandit_sub_exp[st.sub[i]] += float(r["expected"])
                st.bandit_sub_clicks[st.sub[i]] += float(r["clicks"])
                n += 1
        print(f"bandit restored for {n} products ({st.bandit_clicks.sum():.0f} clicks vs {st.bandit_exp.sum():.0f} expected)")
    except Exception as e:
        print(f"bandit starts empty ({str(e)[:120]})")
    try:
        rows = sql(http, BOUGHT_RESTORE.format(team=TEAM), timeout=(3, 25))
        for r in rows:
            if r["p"] in st.idx:
                st.mark_bought(st.user(r["u"]), st.idx[r["p"]], int(r["t"]))
        print(f"carousel purchases restored: {len(rows)}")
    except Exception as e:
        print(f"carousel purchases not restored ({str(e)[:120]})")


def main(dry: bool, seconds: float = 0) -> None:
    t0 = time.time()
    st, rng = State("live"), np.random.default_rng()
    if os.environ.get("TWIN_EMBEDDINGS") == "1":
        from twin.embeddings import semantic_candidates
        semantic_candidates(st, [], -1)  # fit before reading requests
    arms = arms_from_env(VERSION)
    holdout = HOLDOUT
    rankers = {m: Ranker(ARTIFACTS / m) for m in {v.model for v, _ in arms}}     # one ranker per model folder
    layer = LiveLayer(st)                                                        # fed by every arm, used by m3-live
    declog = DecLog(NF)
    arms_stamp, arms_checked = None, 0.0

    def reload_arms():
        """config/twin_arms.txt changed: switch arms (and holdout) without restarting. A file that
        cannot be used leaves everything as it was."""
        nonlocal arms, holdout, arms_stamp
        try:
            stamp = ARMS_FILE.stat().st_mtime_ns
        except OSError:
            return
        if stamp == arms_stamp:
            return
        arms_stamp = stamp
        try:
            spec, new_holdout = read_arms_file()
            new = parse_arms(spec, VERSION, strict=True) if spec else None
            if not new:
                print("config/twin_arms.txt has no usable arm: keeping the current ones")
                return
            for m in {v.model for v, _ in new} - set(rankers):
                rankers[m] = Ranker(ARTIFACTS / m)             # a model folder seen for the first time
            arms = new
            if new_holdout is not None:
                holdout = new_holdout
            print(f"{time.strftime('%H:%M:%S')} arms now: " + ", ".join(f"{v.name} {100 * w:.0f}%" for v, w in arms)
                  + f" | holdout {holdout:.0%}", flush=True)
        except Exception as e:
            print(f"config/twin_arms.txt ignored ({type(e).__name__}: {str(e)[:120]}): keeping the current arms")

    reload_arms()
    print(f"state loaded in {time.time() - t0:.0f}s: {st.P} products ({int(st.is_new.sum())} new), {len(st.uidx)} users")
    print("arms: " + ", ".join(f"{v.name} {100 * w:.0f}%" for v, w in arms))
    http = new_http()
    clock = Clock()
    now = clock.now                                          # the organisers' clock
    restore(st, http, layer, now())
    print(f"server clock offset {clock.offset} ms | version {VERSION} | holdout {holdout:.0%} | {'DRY RUN' if dry else 'LIVE'}")
    heartbeat = os.environ.get("TWIN_HEARTBEAT")             # set by twin.supervisor: proof the loop is not frozen
    last_beat = 0.0

    inbox: queue.Queue = queue.Queue()
    feeds = [Feed("events", EVENTS, st.cutoff_ms, 3.0, 20000, inbox),
             Feed("catalog", CATALOG, st.catalog_cursor, 0.7, 2000, inbox),
             Feed("feedback", FEEDBACK, now() - 60_000, 3.0, 5000, inbox)]
    writer = Writer()
    for th in (*feeds, writer, declog, clock):
        th.start()

    pending: OrderedDict[str, tuple] = OrderedDict()        # request -> (items, expected clicks, user, expected purchases)
    seen: OrderedDict[str, None] = OrderedDict()
    cursor = now() - OVERLAP_MS
    fb = fallback(st)
    count = {"answered": 0, "holdout": 0, "errors": 0, "too_old": 0, "pull_errors": 0, "push_errors": 0,
             "events": 0, "feedback": 0, "apply_errors": 0}
    per_arm: dict[str, int] = {}
    ages: list[int] = []
    timing: dict[str, list[float]] = {"pull": [], "score": [], "push": []}
    last_refresh = last_report = time.time()
    shown_example = 0

    while True:
        tick = time.time()
        if heartbeat and tick - last_beat > 5:
            try:
                with open(heartbeat, "w") as f:
                    f.write(str(tick))
                last_beat = tick
            except OSError:
                pass
        # ---- 1. apply what the background threads brought (never more than 80 ms per tick) ------
        while time.time() - tick < 0.08:
            try:
                name, rows = inbox.get_nowait()
            except queue.Empty:
                break
            # One bad row (an unexpected value, a product beyond the reserved capacity...) must never
            # stop the loop: the row is skipped and counted.
            if name == "events":
                for r in rows:
                    try:
                        st.apply_event(int(r["t"]), r["et"], r["u"], r["s"], r["p"], float(r["price"] or 0), r["device"], r["region"])
                    except Exception as e:
                        count["apply_errors"] += 1
                        if count["apply_errors"] <= 10:
                            print(f"event skipped: {type(e).__name__}: {str(e)[:100]}")
                try:
                    layer.events(st, rows)
                except Exception as e:
                    count["apply_errors"] += 1
                    print(f"trend update failed: {type(e).__name__}: {str(e)[:100]}")
                count["events"] += len(rows)
            elif name == "catalog":
                for r in rows:
                    try:
                        i = st.idx.get(str(r.get("product_id")))
                        old_price = float(st.price[i]) if i is not None else 0.0
                        st.apply_catalog_update(r)
                        if i is not None:
                            layer.price_change(i, old_price, float(st.price[i]), int(r["t"]))
                    except Exception as e:
                        count["apply_errors"] += 1
                        if count["apply_errors"] <= 10:
                            print(f"catalogue update skipped: {type(e).__name__}: {str(e)[:100]}")
                fb = fallback(st)
            else:
                for r in rows:
                    rec = pending.pop(r["request_id"], None)
                    if rec is None:
                        continue
                    try:
                        items, expected, ui, expected_buy = rec
                        bought = st.idx.get(r["purchased"]) if r["purchased"] else None
                        if bought is not None:
                            st.mark_bought(ui, bought, int(r["t"]))
                        if items is not None and r["served_by"] == "team":
                            clicked = st.idx.get(r["clicked"], -1) if r["clicked"] else -1
                            clicked = clicked if clicked in items else -1
                            st.bandit_update(items, expected, clicked)
                            layer.feedback(st, items, expected, expected_buy, clicked,
                                           bought if bought is not None and bought in items else -1, int(r["t"]))
                            count["feedback"] += 1
                    except Exception as e:
                        count["apply_errors"] += 1
                        if count["apply_errors"] <= 10:
                            print(f"feedback row skipped: {type(e).__name__}: {str(e)[:100]}")
        if time.time() - arms_checked > 15:
            arms_checked = time.time()
            reload_arms()
        if time.time() - last_refresh > 60:
            st.refresh()
            fb = fallback(st)
            last_refresh = time.time()

        # ---- 2. new requests ---------------------------------------------------------------
        try:
            t_pull = time.time()
            rows = bounded(1.2, sql, http, REQUESTS.format(cursor=cursor - OVERLAP_MS), timeout=(0.4, 0.7))
            timing["pull"].append(time.time() - t_pull)
        except Exception as e:
            count["pull_errors"] += 1
            print(f"pull failed: {type(e).__name__} {str(e)[:100]}")
            http = new_http()
            time.sleep(0.2)
            continue

        answers, logs = [], []
        for r in rows:
            rid = r["request_id"]
            if rid in seen:
                continue
            seen[rid] = None
            cursor = max(cursor, int(r["t"]))
            if now() - int(r["t"]) > TOO_OLD_MS:
                count["too_old"] += 1
                continue
            ui = st.user(r["user_id"])
            try:
                if holdout and zlib.crc32(rid.encode()) % 1000 < holdout * 1000:
                    ids, version = [st.pid[i] for i in fb], "baseline-holdout"
                    pending[rid] = (None, None, ui, None)
                    count["holdout"] += 1
                else:
                    t_score = time.time()
                    arm = pick_arm(arms, rid)
                    ids, items, expected, expected_buy, Xs = recommend(st, rankers[arm.model], r, rng, arm, layer)
                    timing["score"].append(time.time() - t_score)
                    if len(set(ids)) != 10 or not st.ok[items].all():
                        raise ValueError("invalid slate")
                    version = arm.name
                    per_arm[version] = per_arm.get(version, 0) + 1
                    pending[rid] = (items, expected, ui, expected_buy)
                    if not dry:
                        declog.add(rid, int(r["t"]), version, ids, Xs, expected, expected_buy)
                    logs.append({"kind": "twin_pred", "request_id": rid, "t": int(r["t"]), "model_version": version,
                                 "items": ids, "expected": [round(float(e), 5) for e in expected]})
                    if dry and shown_example < 5:
                        shown_example += 1
                        print(f"  {r['page']:8s} views={len(r['rv'] or [])} -> "
                              + ", ".join(f"{st.sub_names[st.sub[i]]}" for i in items[:6]))
            except Exception as e:
                count["errors"] += 1
                if count["errors"] <= 20:
                    print(f"scoring failed, sent best sellers: {type(e).__name__}: {str(e)[:120]}")
                ids, version = [st.pid[i] for i in fb], "fallback-error"
            answers.append({"request_id": rid, "items": ids, "model_version": version})
            ages.append(now() - int(r["t"]))
        while len(seen) > 50_000:
            seen.popitem(last=False)
        while len(pending) > 30_000:
            pending.popitem(last=False)

        # ---- 3. send -------------------------------------------------------------------------
        if answers and not dry:
            sent_ok = False
            for attempt in (1,):
                try:
                    t_push = time.time()
                    # 0.4 s was too tight: a push takes ~0.33 s on average, so ~5 % timed out and some
                    # of those answers never counted (96 % answered in one 10-minute slot). 1.2 s still
                    # lands well inside the 3-second deadline (answers leave ~0.7 s after the request).
                    bounded(1.5, http.post, f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers,
                            timeout=(0.5, 1.2)).raise_for_status()
                    sent_ok = True
                    timing["push"].append(time.time() - t_push)
                    break
                except Exception as e:
                    count["push_errors"] += 1
                    print(f"push failed (attempt {attempt}): {type(e).__name__} {str(e)[:100]}")
                    http = new_http()
            for rec in logs:
                writer.q.put(rec)
        count["answered"] += len(answers) if dry or not answers or sent_ok else 0

        if time.time() - last_report > 30:
            a = np.asarray(ages[-2000:]) if ages else np.zeros(1)
            print(f"{time.strftime('%H:%M:%S')} answered {count['answered']} (holdout {count['holdout']}, errors {count['errors']}, "
                  f"too old {count['too_old']}) | age at send p50 {np.median(a):.0f} ms, p95 {np.percentile(a, 95):.0f} ms | "
                  f"pull/push errors {count['pull_errors']}/{count['push_errors']} | events {count['events']}, "
                  f"feedback {count['feedback']}, backlog {inbox.qsize()} | products {st.P}"
                  + (f" | skipped rows {count['apply_errors']}" if count["apply_errors"] else ""), flush=True)
            if len(per_arm) > 1:
                print("         arms: " + ", ".join(f"{k} {n}" for k, n in per_arm.items()), flush=True)
            if declog.errors:
                print(f"         decision log: {declog.rows} rows, {declog.errors} write errors", flush=True)
            print("         " + " | ".join(f"{k} avg {1000 * np.mean(v):.0f} ms, max {1000 * np.max(v):.0f} ms"
                                           for k, v in timing.items() if v), flush=True)
            timing = {k: [] for k in timing}
            ages = ages[-2000:]
            last_report = time.time()
            if seconds and time.time() - t0 > seconds:   # test runs only
                return
        time.sleep(max(0.0, POLL - (time.time() - tick)))


if __name__ == "__main__":
    limit = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 0
    main("--dry" in sys.argv, limit)
