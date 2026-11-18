# Real-time recommender · Datathon FME 2026

[![tests](https://github.com/joanprat07/datathon-fme-2026/actions/workflows/tests.yml/badge.svg)](https://github.com/joanprat07/datathon-fme-2026/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)

A live product recommender that answers every shopper of an online store with 10 products in under
3 seconds, and keeps learning from clicks and purchases while it runs. Built in a weekend by team 28
for the RawTree challenge at the Datathon FME 2026 (UPC).

<!-- TODO: add a screenshot or GIF of the presentation (frontend/pandabuy.html, "Live" tab) here:
![demo](docs/demo.gif) -->

**Result:** [placeholder: final ranking / conversion on the final round]

## Highlights

- **Two-stage recommender:** ~300 candidates (co-visits, shopper history, popularity, trends, new
  products) re-ranked by LightGBM as `p(click) × p(purchase | click)`.
- **Online learning:** a bandit layer corrects the ranker with our own live feedback, forgetting
  with a 2-hour half-life.
- **Cold start on both sides:** unknown shoppers (the ranker is also trained with the profile
  hidden) and brand-new products (a second ranker, `model_v7s`, trained for them).
- **Built for a hard deadline:** streaming loop with background threads, a hard network timeout,
  and a best-seller fallback so every request gets a valid answer.
- **Live A/B testing:** several model variants can serve side by side, switched without restarting.

## How it works

```
request ─▶ candidates (~300) ─▶ LightGBM ranker ─▶ bandit correction ─▶ filters ─▶ 10 products
              ▲                      ▲                   ▲
   co-visits, user history,   trained on the old    our own feedback
   popularity, trends, new    carousel's log        (observed vs expected clicks)
```

The final model, `v5tcs`, is a combination of small, separately tested pieces:

```
v5tcs = v5-hybrid + m2-intent + m3-live + trend2 + cold8 + seedtrend
```

| Piece | What it does |
|---|---|
| `v5-hybrid` | LightGBM ranker with 60 features, including 4 text and co-visit similarities to the session and the shopper's profile |
| `m2-intent` | Fills the 10 slots following the shopper's intent, instead of a fixed "at most 5 per subcategory" rule |
| `m3-live` | Bandit updated from our feedback, with a 2 h half-life |
| `trend2` | Boosts products rising in the last ~20 minutes (multiplier between 0.64 and 1.96) |
| `cold8` | Products without history are scored by `model_v7s`, trained for new products |
| `seedtrend` | New products start from the rate of similar products, so they don't look "rising" just for being new |

Products the shopper already bought are never recommended.

## Technical decisions and challenges

- **No leakage from the future.** Features come from events before 25 Oct; the ranker is trained on
  sessions from 25 Oct to 11 Nov and validated on 11–14 Nov.
- **Position bias.** The slot where the old carousel showed a product is a training feature; at
  serving time every candidate is scored as if it were in slot 1.
- **No product IDs in the model.** It learns rules (affinity, co-visits, price, popularity), so it
  generalises to products and shoppers it has never seen.
- **One feature function for training and serving** (`State.featurize`), so offline and live
  numbers can't drift apart.
- **Already-bought products.** In the old carousel's log they convert far better (≈0.7–4% vs
  0.05%); we treated that as a bias of the log and block them. `BLOCK_BOUGHT` switches the rule.
- **Reliability under a 3-second budget.** Only the main thread answers; event, catalogue and
  feedback downloads run in background threads. A supervisor restarts the loop if it stops, and
  the bandit state is saved so it survives a restart.

More detail in [twin/README.md](twin/README.md).

### Offline validation

Last 3 days of history, never seen in training. Among the 10 products the old carousel showed, how
often is the model's first choice the product that was bought?

| Ranking | First choice is the bought product |
|---|---|
| Random | 10% |
| Popularity | 35% |
| Our model, shopper unknown (cold) | 43% |
| Our model | 48% |

Purchase AUC 0.954 · click AUC 0.866 (vs 0.639 for a position-only model).

## Tech stack

Python 3.13 · LightGBM · scikit-learn (TF-IDF + SVD embeddings) · pandas · NumPy · SciPy ·
RawTree (SQL over the event streams) · Streamlit (presentation) · pytest

## Installation

```bash
git clone https://github.com/joanprat07/datathon-fme-2026.git
cd datathon-fme-2026
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in TEAM_ID and RAWTREE_API_KEY
```

Running the pipeline needs credentials for the challenge's RawTree database. The tests and the
presentation run without them.

## Usage

All commands from the repository root.

| Step | Command | Time |
|---|---|---|
| 1. Training data | `python -m twin.build train` | ~4 min |
| 2. Base ranker | `python -m twin.train --refit` | ~6 min |
| 3. Hybrid feature matrix | `python -m twin.hybrid` | |
| 4. 60-feature ranker | `python -m twin.hybrid_live` | |
| 5. Ranker for new products | `python -m twin.train_v7 --out model_v7s` | a few min |
| 6. Live snapshot | `python -m twin.build live` | ~3 min |
| 7. Dry run (sends nothing) | `python -m twin.live --dry --seconds 60` | 1 min |
| 8. Live | `python -m twin.supervisor` | runs forever |
| 9. Results | `python -m twin.monitor 60` | seconds |

Which model serves is set in `config/twin_arms.txt` (one `name=recipe: share` per line, re-read
every 15 s). With several lines the traffic is split and `twin.monitor` compares them.

**Presentation:** open `frontend/pandabuy.html` in a browser, or run
`pip install streamlit && cd frontend && streamlit run frontend.py`.

**Tests:**

```bash
python -m pytest -q tests --ignore=tests/test_v9.py   # test_v9 needs the trained model_v5 and model_v7s
```

## Project structure

```
twin/          the recommender: data, features, training, live loop, monitoring
config/        which model variant serves live
artifacts/     trained base ranker (other models and data are rebuilt by the steps above)
frontend/      presentation: Streamlit app and a self-contained HTML export
tests/         pytest suite
```

## Team

- Joan Prat — [@joanprat07](https://github.com/joanprat07)
- Roger Casas — [@Wanxer](https://github.com/Wanxer)
- Àlex Sánchez — [@alexsanchezcastillo](https://github.com/alexsanchezcastillo)
- Martí Pujadas — [@martipujadas](https://github.com/martipujadas)

[placeholder: who did what, if the team agrees, e.g. "Joan: live loop and bandit"]

## Roadmap

- [placeholder: e.g. record the exact `model_v7s` training options used in the final]
- [placeholder]

## License

[placeholder: licence agreed by the team, e.g. MIT]
