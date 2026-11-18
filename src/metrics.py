"""Point-level metrics shared by training and HMM evaluation. Owner: A."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def _argmax_rows(trip_id: np.ndarray, pt: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row index of the max-p candidate per (trip, point) (first row wins ties) and each row's group id."""
    new = np.r_[True, (trip_id[1:] != trip_id[:-1]) | (pt[1:] != pt[:-1])]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    assert len(starts) == len(set(zip(trip_id[starts], pt[starts]))), "rows must be grouped by (trip_id, pt_idx)"
    is_max = p == np.maximum.reduceat(p, starts)[gid]
    idx = np.flatnonzero(is_max)
    _, first = np.unique(gid[idx], return_index=True)
    return idx[first], gid


def point_metrics(cands: pd.DataFrame, p: np.ndarray) -> dict:
    """Keys: top1, macro_f1, log_loss, miss_rate, n_points (PLAN section 5 definitions).

    Only rows of labelled points (label_arc >= 0) are scored. Rows must be sorted by (trip_id, pt_idx, dist).
    """
    p = np.asarray(p, np.float64)
    keep = (cands["label_arc"].to_numpy() >= 0)
    c, p = cands[keep], p[keep]
    if len(c) == 0:
        return {"top1": np.nan, "macro_f1": np.nan, "log_loss": np.nan, "miss_rate": np.nan, "n_points": 0}
    tid = c["trip_id"].to_numpy()
    pt = c["pt_idx"].to_numpy()
    arc = c["arc_id"].to_numpy()
    label = c["label"].to_numpy().astype(int)
    best, gid = _argmax_rows(tid, pt, p)
    n_points = len(best)
    top1 = float((arc[best] == c["label_arc"].to_numpy()[best]).mean())
    pred = np.zeros(len(c), int)
    pred[best] = 1
    hit = np.bincount(gid, weights=label, minlength=n_points) > 0
    return {
        "top1": top1,
        "macro_f1": float(f1_score(label, pred, average="macro", labels=[0, 1])),
        "log_loss": float(log_loss(label, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "miss_rate": float(1 - hit.mean()),
        "n_points": int(n_points),
        "precision": float(precision_score(label, pred, zero_division=0)),
        "recall": float(recall_score(label, pred, zero_division=0)),
    }


def point_top1_from_assignment(label_arc: np.ndarray, pred_arc: np.ndarray) -> float:
    """Top-1 of a per-point arc assignment, ignoring points with label_arc == -1 (pred -1 counts as wrong)."""
    label_arc, pred_arc = np.asarray(label_arc), np.asarray(pred_arc)
    mask = label_arc >= 0
    return float((pred_arc[mask] == label_arc[mask]).mean()) if mask.any() else float("nan")


if __name__ == "__main__":
    # two labelled points: one hit, one miss (true arc not among candidates); one unlabelled point
    df = pd.DataFrame({
        "trip_id": ["t"] * 7,
        "pt_idx": [0, 0, 1, 1, 2, 2, 2],
        "arc_id": [10, 11, 20, 21, 30, 31, 32],
        "label": [1, 0, 0, 0, 0, 0, 0],
        "label_arc": [10, 10, 99, 99, -1, -1, -1],
    })
    p = np.array([0.9, 0.1, 0.6, 0.4, 0.5, 0.3, 0.2])
    m = point_metrics(df, p)
    assert m["top1"] == 0.5 and m["miss_rate"] == 0.5 and m["n_points"] == 2, m
    assert point_top1_from_assignment(np.array([5, -1, 7, 8]), np.array([5, 1, -1, 8])) == 2 / 3
    print("metrics self-check OK", m)
