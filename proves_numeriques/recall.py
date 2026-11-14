"""Recall dels candidats: en quin % de compres de validacio el producte comprat era entre els candidats?

    python -m twin.recall          # 5000 sessions
    python -m twin.recall 20000
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from twin.build import VALID_START
from twin.recommend import candidates
from twin.rt import ARTIFACTS
from twin.state import DEVICES, PAGES, State, to_ms


def main(n: int = 5000) -> None:
    st = State("train")
    s = pd.read_parquet(ARTIFACTS / "train" / "sessions.parquet")
    s = s[(s.ts >= to_ms(VALID_START)) & (s.purchased != "")]
    s = s.sample(min(n, len(s)), random_state=0)
    rng = np.random.default_rng(0)

    targets: list[int] = []
    hit_known: list[bool] = []
    hit_cold: list[bool] = []
    size_known: list[int] = []
    size_cold: list[int] = []

    for r in s.itertuples(index=False):
        target = st.idx.get(r.purchased)
        if target is None:
            continue
        ui = st.uidx.get(r.user_id, -1)
        if ui >= 0 and target in st.bought_before(ui, r.ts)[0]:
            continue                                     # ja comprat: la regla el bloqueja igualment
        rv = [st.idx[p] for p in r.rv.split(" ") if p in st.idx] if r.rv else []
        page, dev = PAGES.get(r.page, -1), DEVICES.get(r.device, -1)
        reg = st.region_id.get(r.region, -1)

        known = candidates(st, r.ts, dev, reg, ui, rv, rng)
        cold = candidates(st, r.ts, dev, reg, -1, rv, rng)   # sense perfil d'usuari

        targets.append(target)
        hit_known.append(target in known)
        hit_cold.append(target in cold)
        size_known.append(len(known))
        size_cold.append(len(cold))

    k_known, k_cold = int(np.mean(size_known)), int(np.mean(size_cold))

    def pop_recall(k: int) -> float:
        top = set(st.pop_order[:k].tolist())
        return float(np.mean([t in top for t in targets]))

    print(f"{len(targets)} compres de validacio\n")
    print(f"{'':28s}{'recall':>9s}{'mida pool':>12s}")
    print(f"{'candidats, coneguts':28s}{100 * np.mean(hit_known):8.1f}%{k_known:12d}")
    print(f"{'candidats, freds':28s}{100 * np.mean(hit_cold):8.1f}%{k_cold:12d}")
    print(f"{'popularitat (mida coneguts)':28s}{100 * pop_recall(k_known):8.1f}%{k_known:12d}")
    print(f"{'popularitat (mida freds)':28s}{100 * pop_recall(k_cold):8.1f}%{k_cold:12d}")
    print(f"{'popularitat (300)':28s}{100 * pop_recall(300):8.1f}%{300:12d}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)