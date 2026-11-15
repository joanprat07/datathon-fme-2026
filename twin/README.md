# Shopper Twin · v5tcs

This repository holds only the final model, **v5tcs**:

```
v5tcs = v5-hybrid + m2-intent + m3-live + trend2 + cold8 + seedtrend
```

| Part | What it does | Where |
|---|---|---|
| `v5-hybrid` | LightGBM ranker with 60 variables (the 56 base ones + 4 content/collaborative similarities) | `hybrid.py`, `hybrid_live.py` -> `artifacts/model_v5` |
| `m2-intent` | slate filled by the shopper's intent instead of "at most 5 per subcategory" | `variants.py` |
| `m3-live` | bandit with forgetting (2 h half-life), learns from the stream | `variants.py` (`LiveLayer`) |
| `trend2` | per-product trend multiplier squared (0.64-1.96) | `variants.py` |
| `cold8` | products without history are ranked by `model_v7s` (de-biased, trained for new products) | `train_v7.py` -> `artifacts/model_v7s` |
| `seedtrend` | new products start from their inherited 7-day rate, so they are not "rising" just for being new | `variants.py` |

Rebuild and run (from the repository root, venv active, `pip install -r requirements.txt`):

1. `python -m twin.build train` · `python -m twin.train --refit` (base ranker, `artifacts/model`)
2. `python -m twin.hybrid` (feature matrix with the 4 hybrid variables) · `python -m twin.hybrid_live` (`artifacts/model_v5`)
3. `python -m twin.train_v7 --out model_v7s` (`artifacts/model_v7s`, used by `cold8`)
4. `python -m twin.build live` · `python -m twin.live --dry --seconds 60`
5. `python -m twin.supervisor` (or `.\start_live.ps1` on Windows). The arm is read from `config/twin_arms.txt`.

Tests: `python -m pytest -q tests` (`test_v9.py` needs `artifacts/model_v5` and `artifacts/model_v7s`).

---


A two-stage recommender with an online learning layer, for the RawTree challenge.

```
request ─▶ candidates (~300) ─▶ LightGBM ranker ─▶ bandit correction ─▶ filters ─▶ 10 products
              ▲                      ▲                   ▲
   co-visits, user history,   trained on the old    our own feedback
   popularity, trends, new    carousel's log        (observed vs expected clicks)
```

## How to run it

All commands from the repository root, with the virtual environment active.

| Step | Command | Takes |
|---|---|---|
| 1. Download the training snapshot | `python -m twin.build train` | ~4 min |
| 2. Train and validate the ranker | `python -m twin.train --refit` | ~6 min |
| 3. Download the live snapshot (history + stream so far) | `python -m twin.build live` | ~3 min |
| 4. Test without sending anything | `python -m twin.live --dry --seconds 60` | 1 min |
| 5. Go live (**one copy per team**) | `python -m twin.live` | runs forever |
| 6. Watch the results | `python -m twin.monitor 60` | seconds |

Steps 1-2 are only needed again if the features or the model change. Run step 3 again before each
start of step 5: the loop replays the organic events from the snapshot's time, so an old snapshot
means a slow catch-up.

Settings for step 5 (environment variables):

- `TWIN_VERSION` (default `twin-v1`): the `model_version` written with each answer.
- `TWIN_HOLDOUT` (default `0`): share of requests answered with the baseline on purpose, labelled
  `baseline-holdout`, to measure the model's lift on live traffic. Use `0.1` during practice and
  **`0` during the final round**.

- `TWIN_ARMS` (default: none): compare model variants on live traffic. Each request goes to one arm by a
  stable hash and the arm's name is its `model_version`, so step 6 shows them side by side. Example:
  `TWIN_ARMS="twin-v1:60,m1-rebuy:20,m2-intent:10,m3-live:10"`. Variants are in `variants.py`; join two
  with `+` to run both changes at once (`m1-rebuy+m3-live`). `m1-rebuy` needs its own model first:
  `python -m twin.train --keep-bought --out model_rebuy --refit`. Use one arm during the final round.

To restart the loop automatically if it ever stops: `python -m twin.supervisor` instead of step 5.

## What each file does

| File | Role |
|---|---|
| `rt.py` | RawTree client: SQL in, rows or DataFrames out |
| `build.py` | SQL aggregates computed in RawTree and downloaded as parquet (co-visits, user profiles, popularity, trends, carousel statistics) |
| `state.py` | Everything the model knows, in memory, plus `featurize()`: the one place where features are computed, for training and for serving |
| `train.py` | Trains `p(click)` and `p(purchase | click)` and validates them on the last 3 days of history |
| `recommend.py` | Candidates, ranking, bandit, filters: one request in, ten products out |
| `live.py` | The loop: answers requests, follows events, catalogue updates and feedback in the background |
| `monitor.py` | Reliability and conversion per model version, from our feedback |
| `variants.py` | Model variants for live comparison: `m1-rebuy`, `m2-intent`, `m3-live` (see `docs/ANALISI_TWIN.md`) |
| `supervisor.py` | Starts the live loop and starts it again if it stops |
| `hybrid.py`, `hybrid_live.py` | The 4 hybrid variables and the 60-variable ranker `model_v5` |
| `train_v7.py` | The ranker for products without history (`model_v7s`, variant `cold8`) |
| `embeddings.py`, `declog.py`, `train_feedback.py` | Text vectors for new products; decision log of every answer; retraining from our feedback |

## Design choices

- **Time split.** Features come from events before 25 Oct; the ranker is trained on carousel
  sessions from 25 Oct to 11 Nov and validated on 11-14 Nov. Nothing is computed from the future
  of the session being scored.
- **Position bias.** The slot is a feature in training; at serving every candidate is scored as
  if shown in slot 1, so the ranking is not distorted by where the old system placed things.
- **No product IDs in the model.** It learns rules (affinity, co-visits, price, popularity), so it
  works for products and shoppers it has never seen.
- **Cold start.** A quarter of the training sessions are repeated with the shopper's profile
  hidden, so the ranker also learns to rank from the session alone. New products borrow the
  statistics and co-visits of their 5 closest products (same subcategory, then brand, price and
  description words), and the bandit explores them in slots 8-10.
- **Already-bought products are never recommended**, and those rows are left out of training.
  The README says shoppers tend not to want them again. Note: in the old carousel's log those
  products convert far better than the rest (about 0.7-4% against 0.05%); we treat that as a bias
  of the log. `BLOCK_BOUGHT` in `recommend.py` switches the rule if a live test says otherwise.
- **Learning from the stream.** Organic events update popularity, user profiles, co-visits and
  subcategory trends every 3 seconds; catalogue updates change stock and prices within a second;
  feedback updates the bandit (per product, shrunk towards its subcategory).
- **Reliability.** Any error while scoring falls back to the best sellers in stock. Network calls
  have a hard time limit. Predictions are saved to `team_NN.scratch`, so the bandit survives a restart.

## Offline validation (last 3 days of history, never seen in training)

Among the 10 products the old carousel showed, how often is the model's first choice the one that
was bought? (Sessions with a purchase; already-bought products excluded.)

| Ranking | First choice is the bought product |
|---|---|
| Random | 10% |
| Popularity | 35% |
| Shopper Twin, shopper unknown (cold) | 43% |
| Shopper Twin | 48% |

Purchase AUC 0.954 (click AUC 0.866, against 0.639 for a position-only model).
