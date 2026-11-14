# Shopper Twin

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
