"""Decision log: the features of every carousel we serve, saved to disk for retraining.

Our feedback says what the hidden shopper model did with OUR lists: the most valuable training
data there is. To learn from it we need the features each shown product had at that moment,
and those cannot be recomputed later (the state keeps moving). The live loop hands every answer
to `DecLog.add`; a background thread writes one file per hour:

    artifacts/declog/<YYYYMMDD-HH>-<run>.npz    rid, t, arm, pids[n,10], X[n,10,F], exp_click, exp_buy

`python -m twin.train_feedback` joins these files with team_NN.feedback and retrains the ranker.
Nothing here can stop or slow the loop: adding is an append, writing happens in its own thread,
and any error is counted and skipped.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from twin.rt import ARTIFACTS

DIR = ARTIFACTS / "declog"
HOUR_MS = 3_600_000


class DecLog(threading.Thread):
    def __init__(self, n_features: int, flush_s: float = 120, directory=None):
        super().__init__(daemon=True, name="declog")
        self.dir = directory or DIR
        self.F, self.flush_s = n_features, flush_s
        self.tag = time.strftime("%d%H%M%S", time.gmtime())     # one set of files per start of the loop
        self.hours: dict[int, list] = {}                         # hour -> rows not yet superseded
        self.dirty: set[int] = set()
        self.lock = threading.Lock()
        self.rows = self.errors = 0

    def add(self, rid: str, t: int, arm: str, pids: list[str], X: np.ndarray, exp_click, exp_buy) -> None:
        if len(pids) != 10:
            return
        x = np.zeros((10, self.F), np.float32)
        ec, eb = np.zeros(10, np.float32), np.zeros(10, np.float32)
        k = min(len(X), 10)
        x[:k], ec[:k], eb[:k] = X[:k, :self.F], exp_click[:k], exp_buy[:k]   # a v5 ranker has 4 more columns
        with self.lock:
            hour = int(t) // HOUR_MS
            self.hours.setdefault(hour, []).append((rid, int(t), arm, list(pids), x, ec, eb))
            self.dirty.add(hour)
            self.rows += 1

    def flush(self) -> None:
        with self.lock:
            todo = {h: list(self.hours[h]) for h in self.dirty}
            self.dirty.clear()
            newest = max(self.hours, default=0)
            for h in [h for h in self.hours if h < newest - 1 and h not in todo]:
                del self.hours[h]                                # written and closed: free the memory
        self.dir.mkdir(parents=True, exist_ok=True)
        for hour, rows in todo.items():
            name = time.strftime("%Y%m%d-%H", time.gmtime(hour * 3600)) + f"-{self.tag}"
            tmp = self.dir / f"{name}.{os.getpid()}.tmp.npz"
            np.savez(tmp, rid=np.array([r[0] for r in rows]), t=np.array([r[1] for r in rows], np.int64),
                     arm=np.array([r[2] for r in rows]), pids=np.array([r[3] for r in rows]),
                     X=np.stack([r[4] for r in rows]), exp_click=np.stack([r[5] for r in rows]),
                     exp_buy=np.stack([r[6] for r in rows]))
            os.replace(tmp, self.dir / f"{name}.npz")            # the hour's file is rewritten whole, atomically

    def run_once(self) -> None:
        try:
            self.flush()
        except Exception as e:
            self.errors += 1
            if self.errors <= 5:
                print(f"decision log not written: {type(e).__name__}: {str(e)[:120]}")

    def run(self) -> None:  # noqa: A003 (Thread API)
        while True:
            time.sleep(self.flush_s)
            self.run_once()


def load(directory=None) -> dict[str, np.ndarray]:
    """Every decision-log file as one set of arrays (a request logged twice keeps its last copy)."""
    files = sorted((directory or DIR).glob("*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    if not files:
        raise SystemExit(f"no decision log in {directory or DIR}: the live loop writes it while it runs")
    parts = [dict(np.load(f, allow_pickle=False)) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    _, last = np.unique(out["rid"][::-1], return_index=True)
    keep = np.sort(len(out["rid"]) - 1 - last)
    return {k: v[keep] for k, v in out.items()}
