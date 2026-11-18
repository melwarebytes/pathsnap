"""Optuna TPE hyperparameter search for the CV winner. Owner: A.

python -m src.hpo --model xgb --trials 40
Objective = mean 5-fold grouped-CV Top-1 (folds from artifacts/results/folds.csv). student_test is only reported.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from .features import FEATURES, load_dataset
from .metrics import point_metrics
from .train import CACHE, PLOTS, RES, ROOT, SEED, clean, load_folds, predict_proba

HPO = ROOT / "artifacts" / "hpo"


def make_estimator(model: str, params: dict):
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, verbose=-1, subsample_freq=1, **params)
    from xgboost import XGBClassifier
    return XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1, **params)


def suggest(trial: optuna.Trial, model: str) -> dict:
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    if model == "lgbm":
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 255, log=True)
        p["max_depth"] = trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12])
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 200, log=True)
    else:
        p["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 1, 20, log=True)
    return p


def fit_eval(model, params, tr, va):
    d = tr[tr["label_arc"] >= 0]
    est = make_estimator(model, params)
    t0 = time.perf_counter()
    est.fit(d[FEATURES].to_numpy(np.float32), d["label"].to_numpy())
    fit_s = time.perf_counter() - t0
    return est, fit_s, point_metrics(va, predict_proba({"model": est, "features": FEATURES}, va))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgb", choices=["xgb", "lgbm"])
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    HPO.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    folds = load_folds()
    df = clean(load_dataset("train")).reset_index(drop=True)
    fold_of = df["trip_id"].map(folds.set_index("trip_id")["fold"]).to_numpy()
    parts = [(df[fold_of != k], df[fold_of == k]) for k in range(5)]

    def objective(trial):
        params = suggest(trial, a.model)
        ms = []
        for k, (tr, va) in enumerate(parts):
            ms.append(fit_eval(a.model, params, tr, va)[2])
            running = float(np.mean([m["top1"] for m in ms]))
            trial.report(running, k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        for key in ("macro_f1", "log_loss"):
            trial.set_user_attr(key, float(np.mean([m[key] for m in ms])))
        return running

    def save(study, trial):
        study.trials_dataframe().to_csv(HPO / "trials.csv", index=False)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=a.trials, callbacks=[save])
    wall = time.perf_counter() - t0
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"trials={len(study.trials)} complete={len(done)} pruned={len(pruned)} wall={wall:.0f}s "
          f"best cv top1={study.best_value:.4f}")

    # refit best on all 78 clean trips, evaluate on student_test
    best = study.best_params
    st = load_dataset("student_test").reset_index(drop=True)
    est, train_s, _ = fit_eval(a.model, best, df, st)
    b = {"name": a.model, "model": est, "features": FEATURES, "params": best, "cv_top1": float(study.best_value)}
    t1 = time.perf_counter()
    p = predict_proba(b, st)
    infer = (time.perf_counter() - t1) / st["trip_id"].nunique() * 1e3
    stm = point_metrics(st, p)
    (ROOT / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(b, ROOT / "artifacts" / "models" / "final.joblib", compress=3)
    joblib.dump(b, CACHE / "models" / f"{a.model}_tuned.joblib")

    # tuned out-of-fold probabilities for the HMM CV evaluation (same folds as the search)
    oof = np.full(len(df), np.nan)
    cv_rows = []
    for k, (tr, va) in enumerate(parts):
        e, _, m = fit_eval(a.model, best, tr, va)
        oof[fold_of == k] = predict_proba({"model": e, "features": FEATURES}, va)
        cv_rows.append(m)
    o = df[["trip_id", "pt_idx", "arc_id"]].copy()
    o["p"] = oof.astype(np.float32)
    o.to_pickle(CACHE / f"oof_{a.model}_tuned.pkl")

    s = pd.read_csv(RES / "model_summary.csv").set_index("model").loc[a.model]
    cvu = pd.read_csv(RES / "cv_results.csv").query("model == @a.model")
    comp = pd.DataFrame([
        {"variant": "untuned", "cv_top1": cvu["top1"].mean(), "cv_macro_f1": cvu["macro_f1"].mean(),
         "cv_log_loss": cvu["log_loss"].mean(), "st_top1": s["st_top1"], "st_macro_f1": s["st_macro_f1"],
         "st_log_loss": s["st_log_loss"], "infer_ms_per_trip": s["infer_ms_per_trip"], "train_s": s["train_s"]},
        {"variant": "tuned", "cv_top1": np.mean([m["top1"] for m in cv_rows]),
         "cv_macro_f1": np.mean([m["macro_f1"] for m in cv_rows]),
         "cv_log_loss": np.mean([m["log_loss"] for m in cv_rows]), "st_top1": stm["top1"],
         "st_macro_f1": stm["macro_f1"], "st_log_loss": stm["log_loss"], "infer_ms_per_trip": infer,
         "train_s": train_s}]).rename(columns={"variant": "model"})
    comp.insert(0, "model_family", a.model)
    comp.to_csv(RES / "hpo_comparison.csv", index=False)
    pd.DataFrame([{"model": a.model, "trials": len(study.trials), "complete": len(done), "pruned": len(pruned),
                   "wall_s": wall, "s_per_trial": wall / len(study.trials), **{f"best_{k}": v for k, v in best.items()}}]
                 ).to_csv(HPO / "hpo_summary.csv", index=False)
    plots(study, comp)
    print(comp.T)


def plots(study, comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [t.number for t in done]
    y = [t.value for t in done]
    ax.scatter(x, y, label="complete trial")
    pr = [t.number for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    ax.scatter(pr, [min(y)] * len(pr), marker="x", c="grey", label="pruned")
    ax.plot(x, np.maximum.accumulate(y), c="r", label="best so far")
    ax.set(xlabel="trial", ylabel="mean CV Top-1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "hpo_history.png", dpi=150)
    plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1])
        ax.set_xlabel("importance (fANOVA)")
        fig.tight_layout()
        fig.savefig(PLOTS / "hpo_param_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # too few completed trials
        print("importance skipped:", e)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
    for a_, cols, t in zip(ax, [["cv_top1", "st_top1"], ["cv_macro_f1", "st_macro_f1"], ["cv_log_loss", "st_log_loss"]],
                           ["Top-1", "Macro F1", "Log loss"]):
        w = 0.35
        for i, c in enumerate(cols):
            a_.bar(np.arange(2) + (i - .5) * w, comp[c], w, label=c.split("_")[0].upper())
        a_.set_xticks(range(2), comp["model"])
        a_.set_title(t)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "hpo_tuned_vs_untuned.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
