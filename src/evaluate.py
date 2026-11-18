"""Classifier-only vs HMM comparison on CV folds and student_test. Owner: B.

python -m src.evaluate --split student_test
python -m src.evaluate --split train [--tune-beta] [--lam-ablation] [--oof cache/oof_lgbm.pkl]
python -m src.evaluate --split student_test --bundle cache/models/rf.joblib --no-hmm
python -m src.evaluate --split evaluator_test      # ONLY at the very end, nothing is tuned on it
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import EXCLUDED_TRIPS, align_labels, list_trips, load_trip
from .features import build_candidates
from .hmm import Graph, route_metrics
from .predict import decode

ROOT = Path(__file__).resolve().parents[1]
RES, PLOTS = ROOT / "artifacts" / "results", ROOT / "artifacts" / "plots"
DEFAULT_BETA = 5.0
BETAS, LAMS = [2, 5, 10, 20], [0.0, 0.5, 1.0]


def top1(label_arc, pred_arc) -> float:
    try:
        from .metrics import point_top1_from_assignment
        return float(point_top1_from_assignment(label_arc, pred_arc))
    except NotImplementedError:  # A's module not merged yet: same definition
        m = label_arc >= 0
        return float((pred_arc[m] == label_arc[m]).mean()) if m.any() else float("nan")


def load_oof(path: Path) -> dict[str, pd.DataFrame]:
    oof = pd.read_pickle(path)
    oof["trip_id"] = oof["trip_id"].astype(str)
    return {k: v for k, v in oof.groupby("trip_id")}


def get_p(trip, cands, bundle, oof):
    """Probabilities for one trip: out-of-fold (train CV) or from a bundle; None = no model."""
    if oof is not None:
        o = oof.get(trip.trip_id)
        if o is None:
            return None
        m = cands[["pt_idx", "arc_id"]].merge(o[["pt_idx", "arc_id", "p"]], on=["pt_idx", "arc_id"], how="left")
        return m["p"].fillna(0.0).to_numpy()
    if bundle is not None:
        from .train import predict_proba
        return predict_proba(bundle, cands)
    return None


def evaluate_trip(split, tid, configs, bundle, oof, thin=None):
    """configs: list of (mode, use_hmm, beta, lam); returns list of metric rows for one trip."""
    trip = load_trip(split, tid, with_route=True)
    label_arc, _ = align_labels(trip)
    cands = build_candidates(trip)
    p = get_p(trip, cands, bundle, oof)
    g = Graph(trip)
    rows = []
    for mode, use_hmm, beta, lam in configs:
        pp = p
        if p is None:
            if mode in ("classifier", "hmm") or lam > 0:
                continue
            pp = np.ones(len(cands))
        t0 = time.perf_counter()
        pred, route, breaks, _ = decode(trip, cands, pp, use_hmm=use_hmm, beta=beta, lam=lam, graph=g)
        ms = (time.perf_counter() - t0) * 1e3
        rows.append({"split": split, "trip_id": tid, "mode": mode, "beta": beta, "lam": lam,
                     "top1": top1(label_arc, pred), **route_metrics(trip, route, trip.route),
                     "n_viterbi_breaks": len(breaks), "ms_total": ms})
    return rows


def run(split, configs, bundle, oof, trips=None):
    ids = trips or list_trips(split)
    if split == "train":
        ids = [t for t in ids if t not in EXCLUDED_TRIPS]
    rows = []
    for i, tid in enumerate(ids):
        rows += evaluate_trip(split, tid, configs, bundle, oof)
        print(f"[{split}] {i + 1}/{len(ids)} {tid}", flush=True)
    return pd.DataFrame(rows)


def summarise(df, by=("split", "mode")):
    cols = ["top1", "route_precision", "route_recall", "route_f1", "continuity", "n_breaks", "ms_total"]
    return df.groupby(list(by), sort=False)[cols].mean().reset_index()


def _bars(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["top1", "route_f1", "continuity"]
    modes = list(summary["mode"])
    x, w = np.arange(len(metrics)), 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(modes):
        ax.bar(x + i * w, summary.loc[summary["mode"] == m, metrics].iloc[0], w, label=m)
    ax.set_xticks(x + w * (len(modes) - 1) / 2, ["Top-1", "Route F1", "Continuity"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Classifier-only vs HMM")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _delta(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv = df.pivot_table(index=["split", "trip_id"], columns="mode", values="top1")
    if "classifier" not in pv or "hmm" not in pv:
        return
    d = (pv["hmm"] - pv["classifier"]).sort_values()
    fig, ax = plt.subplots(figsize=(max(6, len(d) * 0.18), 4))
    ax.bar(range(len(d)), d.values, color=np.where(d.values < 0, "tab:red", "tab:green"))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Top-1 (HMM - classifier)")
    ax.set_xlabel("trip (sorted)")
    ax.set_title("Per-trip effect of HMM decoding")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sweep(df, col, path_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = df.groupby(col)[["top1", "route_f1", "continuity"]].mean()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    s.plot(ax=ax, marker="o")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return s.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--oof", default=None, help="out-of-fold probabilities for train CV")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--tune-beta", action="store_true")
    ap.add_argument("--lam-ablation", action="store_true")
    ap.add_argument("--no-hmm", action="store_true", help="classifier-only (+ route reconstruction)")
    ap.add_argument("--trips", default=None, help="comma separated subset")
    ap.add_argument("--tag", default=None, help="model name for classifier_route_metrics.csv")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    import joblib
    bundle = None
    if a.split != "train" or not a.oof:
        path = a.bundle or (ROOT / "artifacts" / "models" / "final.joblib")
        if Path(path).exists():
            bundle = joblib.load(path)
        elif a.bundle:
            raise SystemExit(f"bundle not found: {path}")
    oof = load_oof(Path(a.oof)) if a.oof else None
    if a.split == "train" and oof is None:
        print("no --oof given: train CV runs with lam=0 only (never final.joblib: leakage)")
        bundle = None
    trips = a.trips.split(",") if a.trips else None
    have_p = bundle is not None or oof is not None
    b = a.beta

    if a.no_hmm:
        df = run(a.split, [("classifier", False, b, 1.0)], bundle, oof, trips)
        name = a.tag or (bundle or {}).get("name", "unknown")
        row = {"model": name, "split": a.split, **summarise(df).drop(columns=["split", "mode"]).iloc[0].to_dict()}
        f = RES / "classifier_route_metrics.csv"
        old = pd.read_csv(f, dtype={"model": str}) if f.exists() else pd.DataFrame()
        old = old[~((old.get("model") == name) & (old.get("split") == a.split))] if len(old) else old
        pd.concat([old, pd.DataFrame([row])]).to_csv(f, index=False)
        print(pd.DataFrame([row]).T)
        return

    if a.tune_beta:
        df = run(a.split, [(f"beta={x}", True, float(x), 1.0 if have_p else 0.0) for x in BETAS], bundle, oof, trips)
        s = _sweep(df.assign(beta=df["beta"]), "beta", PLOTS / "hmm_beta.png", "beta sweep")
        s.insert(0, "split", a.split)
        f = RES / "hmm_beta.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        pd.concat([old[old.get("split") != a.split] if len(old) else old, s]).to_csv(f, index=False)
        print(s)
        return

    if a.lam_ablation:
        df = run(a.split, [(f"lam={x}", True, b, x) for x in LAMS if have_p or x == 0.0], bundle, oof, trips)
        s = _sweep(df, "lam", PLOTS / "hmm_lam.png", "lam ablation (0 = Gaussian HMM)")
        s.insert(0, "split", a.split)
        f = RES / "hmm_lam.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        pd.concat([old[old.get("split") != a.split] if len(old) else old, s]).to_csv(f, index=False)
        print(s)
        return

    configs = [("classifier", False, b, 1.0), ("hmm", True, b, 1.0), ("gauss_hmm", True, b, 0.0)]
    if not have_p:
        configs = [("nearest", False, b, 0.0), ("gauss_hmm", True, b, 0.0)]
    df = run(a.split, configs, bundle, oof, trips)
    cols = ["split", "trip_id", "mode", "top1", "route_precision", "route_recall", "route_f1",
            "continuity", "n_breaks", "ms_total"]
    f = RES / "hmm_comparison.csv"
    old = pd.read_csv(f, dtype={"trip_id": str}) if f.exists() else pd.DataFrame(columns=cols)
    cmp_ = pd.concat([old[old["split"] != a.split], df[cols]])
    cmp_.to_csv(f, index=False)
    summ = summarise(cmp_)
    summ.to_csv(RES / "hmm_summary.csv", index=False)
    s = summ[summ["split"] == a.split]
    _bars(s, PLOTS / "hmm_vs_classifier.png")
    _delta(df, PLOTS / "hmm_per_trip_delta.png")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
