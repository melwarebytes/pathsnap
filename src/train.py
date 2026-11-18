"""Six-model comparison, GroupKFold CV, plots and the final model bundle. Owner: A.

python -m src.train [--models lr,nb,rf,xgb,lgbm,mlp] [--subsample 300000] [--no-final]
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import EXCLUDED_TRIPS
from .features import FEATURES, load_dataset
from .metrics import point_metrics
from .preprocess import log1p_heavy

ROOT = Path(__file__).resolve().parents[1]
RES, PLOTS, CACHE = ROOT / "artifacts" / "results", ROOT / "artifacts" / "plots", ROOT / "cache"
SEED = 42


def _scaled(est):  # heavy-tailed columns (gaps up to 153,064 s, arcs up to 9 km) are log-scaled for LR/MLP
    return Pipeline([("log", FunctionTransformer(log1p_heavy)), ("scale", StandardScaler()), ("clf", est)])


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(tree_method="hist", n_estimators=400, learning_rate=0.1, max_depth=6,
                         random_state=SEED, n_jobs=-1)


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=SEED, verbose=-1)


MODELS = {
    "lr": lambda: _scaled(LogisticRegression(class_weight="balanced", max_iter=1000)),
    "nb": lambda: GaussianNB(),
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
    "xgb": _xgb,
    "lgbm": _lgbm,
    "mlp": lambda: _scaled(MLPClassifier((64, 32), early_stopping=True, random_state=SEED)),
}
SUBSAMPLED = {"mlp", "knn"}
FAMILY = {"lr": "linear", "nb": "probabilistic", "rf": "bagging (trees)", "xgb": "boosting (trees)",
          "lgbm": "boosting (trees)", "mlp": "neural network"}


def predict_proba(bundle: dict, cands: pd.DataFrame) -> np.ndarray:
    """P(label=1) per candidate row, using bundle["features"] and bundle["model"]."""
    return bundle["model"].predict_proba(cands[bundle["features"]].to_numpy(np.float32))[:, 1]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["trip_id"].isin(EXCLUDED_TRIPS)]


def make_folds() -> pd.DataFrame:
    """GroupKFold(5) over the clean train trips sorted by trip_id; written to artifacts/results/folds.csv."""
    ids = np.array(sorted(clean(load_dataset("train"))["trip_id"].unique()))
    fold = np.zeros(len(ids), int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=5).split(ids, groups=ids)):
        fold[va] = k
    out = pd.DataFrame({"trip_id": ids, "fold": fold})
    RES.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES / "folds.csv", index=False)
    return out


def load_folds() -> pd.DataFrame:
    f = RES / "folds.csv"
    return pd.read_csv(f, dtype={"trip_id": str}) if f.exists() else make_folds()


def _labelled(df):
    return df[df["label_arc"] >= 0]


def _subsample(df: pd.DataFrame, max_rows: int, seed=SEED) -> pd.DataFrame:
    """Whole trips until ~max_rows rows."""
    rng = np.random.default_rng(seed)
    sizes = df.groupby("trip_id").size()
    keep, total = [], 0
    for tid in rng.permutation(sizes.index.to_numpy()):
        if total >= max_rows:
            break
        keep.append(tid)
        total += sizes[tid]
    return df[df["trip_id"].isin(keep)]


def fit(name: str, tr: pd.DataFrame, subsample: int):
    d = _labelled(tr)
    if name in SUBSAMPLED:
        d = _subsample(d, subsample)
    est = MODELS[name]()
    t0 = time.perf_counter()
    est.fit(d[FEATURES].to_numpy(np.float32), d["label"].to_numpy())
    return est, time.perf_counter() - t0, len(d)


def cross_validate(name: str, df: pd.DataFrame, folds: pd.DataFrame, subsample=300_000):
    """Per-fold metrics (train-fold and validation-fold) and out-of-fold probabilities."""
    fmap = folds.set_index("trip_id")["fold"]
    fold_of = df["trip_id"].map(fmap).to_numpy()
    rows, oof = [], np.full(len(df), np.nan)
    for k in range(5):
        tr, va = df[fold_of != k], df[fold_of == k]
        est, train_s, n_rows = fit(name, tr, subsample)
        b = {"model": est, "features": FEATURES}
        t0 = time.perf_counter()
        p = predict_proba(b, va)
        inf = (time.perf_counter() - t0) / len(va) * 1e6  # ms per 1k rows
        oof[fold_of == k] = p
        m = point_metrics(va, p)
        trs = _subsample(_labelled(tr), 300_000, seed=k)  # train-fold score on a trip subsample (overfitting check)
        mt = point_metrics(trs, predict_proba(b, trs))
        rows.append({"model": name, "fold": k, **m, "train_top1": mt["top1"], "train_log_loss": mt["log_loss"],
                     "train_s": train_s, "infer_ms_per_1k": inf, "n_train_rows": n_rows})
        print(f"  {name} fold {k}: top1={m['top1']:.4f} f1={m['macro_f1']:.4f} ll={m['log_loss']:.4f} "
              f"fit={train_s:.0f}s", flush=True)
    return pd.DataFrame(rows), oof


def _merge(path: Path, new: pd.DataFrame, key="model", dtype=None) -> pd.DataFrame:
    old = pd.read_csv(path, dtype=dtype) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old[key].isin(new[key].unique())]
    out = pd.concat([old, new], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def run_model(name, df, st, folds, subsample):
    cv, oof = cross_validate(name, df, folds, subsample)
    CACHE.mkdir(exist_ok=True)
    o = df[["trip_id", "pt_idx", "arc_id"]].copy()
    o["p"] = oof.astype(np.float32)
    o.to_pickle(CACHE / f"oof_{name}.pkl")
    # refit on all 78 clean trips, in the bundle format of PLAN 6.3
    est, train_s, _ = fit(name, df, subsample)
    bundle = {"name": name, "model": est, "features": FEATURES, "params": est.get_params(deep=False),
              "cv_top1": float(cv["top1"].mean())}
    (CACHE / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, CACHE / "models" / f"{name}.joblib")
    t0 = time.perf_counter()
    p = predict_proba(bundle, st)
    infer_trip = (time.perf_counter() - t0) / st["trip_id"].nunique() * 1e3
    stm = point_metrics(st, p)
    per_trip = []
    for tid, g in st.groupby("trip_id", sort=True):
        m = point_metrics(g, p[st["trip_id"].to_numpy() == tid])
        per_trip.append({"model": name, "trip_id": tid, **{k: m[k] for k in ("top1", "macro_f1", "log_loss")}})
    summ = {"model": name, "family": FAMILY[name]}
    for k in ("top1", "macro_f1", "log_loss", "miss_rate", "train_top1", "train_log_loss"):
        summ[f"cv_{k}_mean"], summ[f"cv_{k}_std"] = cv[k].mean(), cv[k].std()
    summ.update({f"st_{k}": stm[k] for k in ("top1", "macro_f1", "log_loss", "precision", "recall", "miss_rate")})
    summ.update({"train_s": train_s, "infer_ms_per_trip": infer_trip, "cv_train_s_mean": cv["train_s"].mean()})
    return cv, pd.DataFrame(per_trip), summ, bundle, p


def pick_winner(summary: pd.DataFrame) -> str:
    s = summary.sort_values(["cv_top1_mean", "cv_log_loss_mean"], ascending=[False, True])
    return s.iloc[0]["model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--no-final", action="store_true", help="do not write artifacts/models/final.joblib")
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    folds = load_folds()
    if not a.plots_only:
        df = clean(load_dataset("train")).reset_index(drop=True)
        assert set(df["trip_id"].unique()) == set(folds["trip_id"]), "folds.csv does not match the train cache"
        st = load_dataset("student_test").reset_index(drop=True)
        for name in a.models.split(","):
            print(f"== {name}", flush=True)
            cv, per_trip, summ, bundle, p = run_model(name, df, st, folds, a.subsample)
            _merge(RES / "cv_results.csv", cv)
            _merge(RES / "student_test_results.csv", per_trip)
            _merge(RES / "model_summary.csv", pd.DataFrame([summ]))
            st_p = p
            _confusion(name, st, st_p)
            if name == "mlp":
                _mlp_curve(bundle["model"])
        summary = pd.read_csv(RES / "model_summary.csv")
        if not a.no_final:
            w = pick_winner(summary)
            b = joblib.load(CACHE / "models" / f"{w}.joblib")
            (ROOT / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
            joblib.dump(b, ROOT / "artifacts" / "models" / "final.joblib", compress=3)
            print("untuned winner:", w, "-> artifacts/models/final.joblib")
    make_plots()


def _confusion(name, st, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from .metrics import _argmax_rows
    c = st[st["label_arc"] >= 0]
    pp = p[(st["label_arc"] >= 0).to_numpy()]
    best, _ = _argmax_rows(c["trip_id"].to_numpy(), c["pt_idx"].to_numpy(), pp)
    pred = np.zeros(len(c), int)
    pred[best] = 1
    cm = confusion_matrix(c["label"].to_numpy(), pred, labels=[0, 1])
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({cm[i, j]:,})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"{name}: student_test (candidate level)")
    fig.tight_layout()
    fig.savefig(PLOTS / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def _mlp_curve(pipe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clf = pipe.named_steps["clf"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(clf.loss_curve_)
    ax[0].set(title="MLP training loss", xlabel="epoch", ylabel="log loss")
    ax[1].plot(clf.validation_scores_)
    ax[1].set(title="MLP validation accuracy (10 % held out)", xlabel="epoch", ylabel="accuracy")
    fig.tight_layout()
    fig.savefig(PLOTS / "mlp_loss_curve.png", dpi=150)
    plt.close(fig)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.read_csv(RES / "model_summary.csv")
    cv = pd.read_csv(RES / "cv_results.csv")
    order = list(s["model"])

    def bar(col, ylabel, fname, hlines=()):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(order, [s.set_index("model").loc[m, f"cv_{col}_mean"] for m in order],
               yerr=[s.set_index("model").loc[m, f"cv_{col}_std"] for m in order], capsize=4)
        for v, lab in hlines:
            ax.axhline(v, color="r", ls="--", lw=1)
            ax.text(len(order) - 0.5, v, lab, ha="right", va="bottom", fontsize=8, color="r")
        ax.set_ylabel(ylabel)
        ax.set_title(f"5-fold grouped CV: {ylabel} (mean, std over folds)")
        fig.tight_layout()
        fig.savefig(PLOTS / fname, dpi=150)
        plt.close(fig)

    bar("macro_f1", "Macro F1", "cv_macro_f1.png")
    bar("top1", "Top-1", "cv_top1.png", [(0.551, "nearest arc 55.1 %"), (0.831, "nearest + heading 83.1 %")])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, s["st_top1"], 0.4, label="Top-1")
    ax.bar(x + 0.2, s["st_macro_f1"], 0.4, label="Macro F1")
    ax.set_xticks(x, order)
    ax.legend()
    ax.set_title("student_test")
    fig.tight_layout()
    fig.savefig(PLOTS / "student_test_top1_f1.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, s["cv_log_loss_mean"], 0.4, label="CV")
    ax.bar(x + 0.2, s["st_log_loss"], 0.4, label="student_test")
    ax.set_xticks(x, order)
    ax.legend()
    ax.set_ylabel("log loss")
    fig.tight_layout()
    fig.savefig(PLOTS / "log_loss.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].bar(order, s["train_s"])
    ax[0].set(yscale="log", ylabel="train s (full refit)")
    ax[1].bar(order, s["infer_ms_per_trip"])
    ax[1].set(yscale="log", ylabel="inference ms / trip")
    fig.tight_layout()
    fig.savefig(PLOTS / "timing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
