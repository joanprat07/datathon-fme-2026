You are taking over a live project. Read all of this before acting. Reply to me in Catalan.

## Project
Datathon FME 2026, RawTree challenge: real-time recommendation. Every ~0.33 s a shopper arrives in
`shop.requests`; we must write 10 distinct, active, in-stock product IDs to `team_28.submissions`
within 3 s. Results come back in `team_28.feedback`. Leaderboard: https://datathon.rawtree.dev/leaderboard
(per `model_version`, uses each shopper's purchase *probability*). Practice round now; the FINAL is
Sunday 15 Nov 10:00–12:00 CET (UTC+1; data timestamps are UTC). Score: conversion 50, cold start 10,
reliability 10, jury presentation 30. The full rules are in README.md.

## Environment (Windows 11, PowerShell 5.1)
- Repo: `C:\Users\casas\OneDrive\Documents\GitHub\datathon-fme-2026`, private GitHub repo `joanprat07/datathon-fme-2026`.
- Python 3.13 venv: `.venv` (run `.\.venv\Scripts\python.exe ...`). Credentials in `.env` (TEAM_ID=team_28, RAWTREE_API_KEY): never print, copy or commit them.
- `git` is not on PATH: use `C:\Users\casas\AppData\Local\GitHubDesktop\app-3.6.11\resources\app\git\cmd\git.exe`.
  You cannot fetch or push (needs my GitHub login): ask me to press "Fetch origin" in GitHub Desktop.
- `rtree` CLI 0.8.4 is installed; the Python client is `twin/rt.py` (`q(sql)` -> DataFrame).

## Code (package `twin/`)
Two-stage recommender "Shopper Twin": ~300 candidates (co-views, user history, segment popularity,
trends, new products) -> two LightGBM models p(click) x p(purchase|click) -> bandit correction ->
slate (never a product the shopper already bought) -> fallback to best sellers on any error.
- `build.py` downloads SQL aggregates to `artifacts/{train,live}`; `train.py` trains `artifacts/model`;
  `live.py` is the live loop; `supervisor.py` restarts it if it crashes; `monitor.py` reads feedback;
  `variants.py` defines arms (`TWIN_ARMS`, e.g. `"a:50,b:50"`, names combine with `+`).
- Current branch: `twin-v5-hybrid-live` (LOCAL ONLY, no upstream, NOTHING COMMITTED), made from
  `origin/twin-v4-hybrid`. My uncommitted changes:
  - `twin/hybrid_live.py` (new): the 4 hybrid features (content/collaborative similarity with the
    session and with the user profile) served live; `python -m twin.hybrid_live` exports embeddings
    and trains `artifacts/model_v5` (60 features). New products get vectors from the text transform.
  - `twin/recommend.py`: `Ranker` accepts 56- or 60-feature models; v5 appends the 4 columns.
  - `twin/variants.py`: new variant `v5-hybrid` (model `model_v5`).
  - `twin/declog.py`: logs only the first 56 columns.
  - `twin/live.py`: submission push timeout raised from 0.4 s to 1.2 s (0.4 s lost answers).
- Other branches: `variants-m1-m3` (m1-rebuy, m2-intent, m3-live), `twin-v3-experimental`
  (decision log, `TWIN_EMBEDDINGS=1` semantic candidates, `config/twin_arms.txt` hot reload),
  `twin-v4-hybrid` (offline hybrid model `twin/hybrid.py`, not servable as is).

## What is running now
ONE live loop, started 22:00 local in a terminal tab of the previous chat (you cannot control that
tab; stop it by killing the python processes `-m twin.supervisor` then `-m twin.live`):
`TWIN_ARMS='m2-intent+m3-live:50,v5-hybrid+m2-intent+m3-live:50'`, no holdout, no TWIN_EMBEDDINGS.
It answers 100 % of requests. Goal: see if v5 (the 4 hybrid features) beats m2+m3 after 1-2 h.

## Results so far (compare only arms that ran at the same time, windows of 1 h or more)
- `m2-intent+m3-live` (teammate's variants) beats `twin-v1` consistently: 1.85 % vs 1.68 % (3 h),
  2.16 % vs 1.92 % (1 h) expected purchases; revenue/request ~0.91-0.99 EUR vs ~0.77-0.86 EUR.
- Every "rebuy" variant (recommending already-bought products) is the worst (1.1-1.5 %). Decision:
  strict rule, never recommend already-bought products. Do not test rebuy again.
- `TWIN_EMBEDDINGS=1` (semantic candidates): inconclusive.
- v5 offline (same validation rows): first pick = bought 47.5 % vs 45.1 % for twin-v1; cold 44.2 %
  vs 43.0 %; purchase AUC 0.962 vs 0.953. Live test started 22:00.

## Rules learned the hard way
1. Only ONE live process per team. Before starting one, list python processes running
   `twin.live`/`twin.supervisor`; several times two ran at once and mixed the comparison.
2. Every restart costs ~1 minute of missed requests: avoid restarts; dry-run first
   (`python -m twin.live --dry --seconds 60`).
3. The laptop must not sleep (it did twice, losing ~26 min). Keep it plugged in, sleep = Never.
4. Ask before stopping a process I started myself, before commits/pushes, and before anything outward.
5. `.env.example` on several branches contains the real API key: it must be replaced by placeholders
   before any merge to `main`.

## Useful commands
`python -m twin.build live` (refresh data, ~3 min) · `python -m twin.monitor 60` (results per version)
· `python -m twin.supervisor` (serve) · `python -m twin.live --dry --seconds 60` (test, sends nothing).

## Next steps
1. In 1-2 h, compare `v5-hybrid+m2-intent+m3-live` vs `m2-intent+m3-live` (twin.monitor + leaderboard).
2. Decide the final model by ~08:00 Sunday; I commit the branch from GitHub Desktop (new branch name).
3. Sunday ~09:30: refresh the snapshot, start ONLY the winner (100 %, `TWIN_HOLDOUT=0`) with the
   supervisor, check `twin.monitor 10`, and do not touch anything from 10:00 to 12:00.
4. Presentation: a summary doc already exists (Claude Docs "Shopper Twin: resum del treball").
