# Part A report notes (model comparison and HPO)

All numbers come from `artifacts/results/*.csv`. Same 5 grouped folds (78 clean trips, trips 33 and 54 excluded) for every model.
Primary metric: point Top-1 on labelled points. Baselines: nearest arc 55.1 %, nearest + heading rule 83.1 % (train).

## 3.1 Comparison (CV = mean of 5 folds; ST = student_test)

| Model | Family / config | CV Top-1 | CV macro F1 | CV log loss | ST Top-1 | ST macro F1 | ST log loss | Route F1 / continuity (classifier-only, ST) | Train s | Infer ms/trip |
|---|---|---|---|---|---|---|---|---|---|---|
| NB | probabilistic, Gaussian | 0.815 | 0.894 | 0.756 | 0.803 | 0.889 | 0.739 | 0.784 / 0.985 | 0.2 | 2.0 |
| LR | linear, balanced weights, log1p+scaler | 0.853 | 0.916 | 0.195 | 0.856 | 0.919 | 0.183 | 0.843 / 0.988 | 2.9 | 8.0 |
| RF | 300 trees, min leaf 20 | 0.855 | 0.917 | 0.091 | 0.863 | 0.923 | 0.078 | 0.843 / 0.986 | 142 | 57.6 |
| XGB | 400 trees, depth 6, lr 0.1 | 0.855 | 0.917 | 0.093 | 0.861 | 0.922 | 0.080 | 0.844 / 0.986 | 6.9 | 10.0 |
| LGBM | 400 trees, 63 leaves, lr 0.05 | 0.850 | 0.914 | 0.104 | 0.860 | 0.921 | 0.090 | 0.791 / 0.984 | 13.1 | 49.4 |
| MLP | (64,32), early stopping, log1p+scaler | 0.843 | 0.910 | 0.115 | 0.853 | 0.917 | 0.087 | 0.846 / 0.986 | 15.7 | 5.3 |
| **XGB tuned (final)** | Optuna TPE, 15 trials | **0.857** | **0.918** | **0.091** | **0.864** | **0.924** | **0.078** | 0.847 / 0.986 (+HMM: 0.985 / 1.000) | 6.9 | 8.8 |

Route F1 and continuity use per-trip means of `evaluate.py --no-hmm`, so student_test Top-1 there differs slightly from the pooled Top-1 above.

## Selection
Rule: best mean CV Top-1, log loss as tie-break. RF is first by 0.0005, far inside the fold std (about 0.026), so it is a tie.
RF is rejected on practical grounds: 365 MB bundle (limit 50 MB), 142 s fit, 58 ms/trip. XGB (1.6 MB, 7 s) is the winner and was tuned.
Folds differ in difficulty (fold 1 about 0.82, fold 4 about 0.88 for every model), so compare models fold by fold, not across folds.

## Over/underfitting evidence (train-fold vs validation-fold Top-1)
NB 0.834 vs 0.815 (underfits, correlated distance features break independence). LR 0.865 vs 0.853 (small gap, cannot model the bearing x heading_ok interaction).
XGB 0.916 vs 0.855 and LGBM 0.906 vs 0.850 (largest gaps: mild overfitting). RF 0.895 vs 0.855. MLP 0.888 vs 0.843 (see `mlp_loss_curve.png`).
student_test is within about 1 point of CV for all models, so there is no sign of CV overfitting. Log loss: NB 0.76 vs about 0.09 for boosting (probabilities badly calibrated).

## 4. Bayesian optimisation (XGBoost, `src/hpo.py`)
1. Searched: n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha.
2. Ranges: n_estimators 200-1500; learning_rate 0.01-0.3 (log); max_depth 3-10; min_child_weight 1-20 (log); subsample and colsample_bytree 0.5-1; reg_lambda, reg_alpha 1e-3-10 (log).
3. Objective: maximise mean 5-fold grouped-CV Top-1 (TPE sampler seed 42, MedianPruner with 2 warm-up folds). student_test was not used.
4. Trials: 15 (11 complete, 4 pruned), 711 s wall, 47 s per trial. Kept short because of time: the gain is small.
5. Best: n_estimators 333, learning_rate 0.049, max_depth 6, min_child_weight 7.5, subsample 0.54, colsample_bytree 0.76, reg_lambda 0.0087, reg_alpha 1.98.
6. Tuned vs untuned (same folds): CV Top-1 0.8547 to 0.8569, CV macro F1 0.9171 to 0.9183, CV log loss 0.0935 to 0.0908; student_test Top-1 0.8614 to 0.8642, macro F1 0.9219 to 0.9235, log loss 0.0799 to 0.0780; inference 10.0 to 8.8 ms/trip. A small but consistent gain, well inside the fold std (0.026), so it is modest evidence.

## Open items for the report
HMM rows (winner + HMM, Gaussian HMM) come from Part B `evaluate.py`. Per-model A1/A2/B5/B6/C7-C9 prose still to write from the table above.
