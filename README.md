# PathSnap

Map matching and route reconstruction for noisy GPS tracks (UE24CS352A hackathon). Each GPS point is
matched to a road arc with a gradient-boosting classifier over candidate arcs, then an HMM/Viterbi
decoder turns the per-point probabilities into a connected route. The full design is in `docs/PLAN.md`.

## Setup

```
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.14). Run everything as modules from `codebase/`.

## Data

Raw data is not in the repo. Expected layout, one folder above `codebase/`:

```
Hackathon/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
  student_test/<trip_id>/...
  evaluator_test/<trip_id>/...      (later)
  codebase/
```

To use another location set `PATHSNAP_DATA` to the folder that contains `train/` and `student_test/`.

## Build the candidate tables

```
python -m src.features train
python -m src.features student_test
```

Writes `cache/<split>_candidates.pkl` (gitignored, about 1 minute for train) and a 3-trip CSV to
`artifacts/sample/`. Prints trip/point/row counts, candidate miss rate and the two naive baselines.
In code: `from src.features import load_dataset, FEATURES`.

## Reproduce the results

Run in this order from `codebase/` (after building the candidate tables above). Every step writes
to `artifacts/` so the committed CSVs and plots can be regenerated.

```
python -m src.eda                                   # EDA tables, plots, Folium maps
python -m src.train                                 # 6 models, 5-fold GroupKFold CV, student_test scores, plots, final bundle
python -m src.hpo --model xgb --trials 15           # Optuna TPE search, tuned vs untuned, re-saves final.joblib
python -m src.evaluate --split student_test         # classifier-only vs classifier+HMM route metrics
python -m src.maps student_test 00000007 artifacts/models/final.joblib   # Folium map for one trip
python -m pytest                                    # tests
```

`evaluator_test` is never used for training or tuning; run
`python -m src.evaluate --split evaluator_test` only once it is released.

## Results (student_test, 10 trips)

Columns follow the model comparison table in the Deliverables PDF. CV = mean of 5 grouped folds on
train; Test = student_test. Route continuity is classifier-only (argmax, then shortest-path gap
filling), so every model is compared the same way. Sources: `artifacts/results/model_summary.csv`,
`classifier_route_metrics.csv`, `hpo_comparison.csv`, `hmm_summary.csv`.

| Model | Family | Key configuration | CV macro F1 | CV Top-1 | Test Top-1 | Route continuity | Test macro F1 | Test log loss | Inference ms/trip |
|---|---|---|---|---|---|---|---|---|---|
| Naive Bayes | probabilistic | Gaussian | 0.894 | 0.815 | 0.803 | 0.985 | 0.889 | 0.739 | 2.0 |
| Logistic regression | linear | balanced weights, log1p + scaler | 0.916 | 0.853 | 0.856 | 0.988 | 0.919 | 0.183 | 8.0 |
| Random forest | bagging (trees) | 300 trees, min leaf 20 | 0.917 | 0.855 | 0.863 | 0.986 | 0.923 | 0.078 | 57.6 |
| XGBoost | boosting (trees) | 400 trees, depth 6, lr 0.1 | 0.917 | 0.855 | 0.861 | 0.986 | 0.922 | 0.080 | 10.0 |
| LightGBM | boosting (trees) | 400 trees, 63 leaves, lr 0.05 | 0.914 | 0.850 | 0.860 | 0.984 | 0.921 | 0.090 | 49.4 |
| MLP | neural network | (64, 32), early stopping, log1p + scaler | 0.910 | 0.843 | 0.853 | 0.986 | 0.917 | 0.087 | 5.3 |
| **XGBoost tuned (final)** | boosting (trees) | Optuna TPE, 15 trials | 0.918 | 0.857 | 0.864 | 0.986 | 0.924 | 0.078 | 8.8 |

Route-level with the tuned XGBoost on student_test (`artifacts/results/hmm_summary.csv`):

| Decoder | Top-1 | Route F1 | Continuity | ms/trip |
|---|---|---|---|---|
| Classifier only (argmax) | 0.883 | 0.847 | 0.986 | 40 |
| Classifier + HMM/Viterbi | 0.949 | 0.985 | 1.000 | 733 |

Route-level Top-1 is a per-trip mean, so it differs from the pooled point Top-1 above.

## Where each Deliverable 1 component is

| Component | Code | Outputs |
|---|---|---|
| Data processing | `src/data.py`, `src/eda.py` | `artifacts/results/eda_*.csv`, `artifacts/plots/eda_*.png` |
| Feature engineering | `src/features.py`, `src/preprocess.py` | `artifacts/sample/*.csv` |
| Model training | `src/train.py` | `artifacts/results/cv_results.csv`, `folds.csv`, `model_summary.csv` |
| Model evaluation | `src/metrics.py`, `src/train.py`, `src/evaluate.py` | `artifacts/plots/confusion_*.png`, `cv_*.png`, `log_loss.png`, `student_test_top1_f1.png`, `mlp_loss_curve.png` |
| Hyperparameter optimisation | `src/hpo.py` | `artifacts/hpo/trials.csv`, `hpo_summary.csv`, `artifacts/results/hpo_comparison.csv`, `artifacts/plots/hpo_*.png` |
| HMM and Viterbi | `src/hmm.py`, `src/predict.py` | `artifacts/results/hmm_*.csv`, `artifacts/plots/hmm_*.png` |
| API | `api/server.py`, `api/smoke.py` | loads `artifacts/models/final.joblib` |
| UI | `ui/index.html` | |
| Deployment | `requirements.txt`, `docs/DEPLOYMENT.md` | |
| Visualisations | `src/maps.py`, `src/eda.py` | `artifacts/maps/*.html` |

## Layout

```
src/data.py       loading, projection, label alignment
src/features.py   candidates, features, dataset cache
src/metrics.py    point metrics
src/train.py      models, CV, final bundle
src/hpo.py        Optuna search
src/hmm.py        Viterbi, route reconstruction, route metrics
src/predict.py    end-to-end single-trip inference
src/evaluate.py   classifier vs HMM evaluation
src/maps.py       folium maps
api/server.py     FastAPI server
ui/index.html     web UI
tests/            python -m pytest
artifacts/        committed results, models, plots, samples
cache/            local candidate tables (not committed)
```

## Team

| Module | Owner |
|---|---|
| `data.py`, `features.py`, `artifacts/sample/` | Merull (lead) |
| `metrics.py`, `train.py`, `hpo.py` | A |
| `hmm.py`, `predict.py`, `evaluate.py`, `maps.py` | B |
| `api/server.py`, `ui/index.html`, `README.md` | C |

## Deployment (two laptops)

Laptop 1 serves the model API; Laptop 2 runs only the static UI. Full setup, Windows hotspot and
firewall steps, the `student_test` dry run, the demo-day runbook and troubleshooting are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Quick start:

```
PATHSNAP_SPLIT=student_test uvicorn api.server:app --host 0.0.0.0 --port 8000
python -m http.server 5500 --directory ui    # Laptop 2: http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
python api/smoke.py --api http://localhost:8000
```
