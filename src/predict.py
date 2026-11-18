"""End-to-end single-trip inference, used by evaluate.py and the API. Owner: B."""
import time

import numpy as np

from .data import load_trip
from .features import build_candidates
from .hmm import Graph, reconstruct_route, route_lonlat, viterbi


def argmax_per_point(trip, cands, p) -> np.ndarray:
    """Per-point arc with the highest p (first row wins ties), -1 where a point has no candidates."""
    pred = np.full(len(trip.xy), -1, np.int64)
    if len(cands) == 0:
        return pred
    pt = cands["pt_idx"].to_numpy(np.int64)
    p = np.asarray(p, np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(pt)) + 1]
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pt)]))
    is_max = p == np.maximum.reduceat(p, starts)[gid]
    idx = np.flatnonzero(is_max)
    _, first = np.unique(gid[idx], return_index=True)
    sel = idx[first]
    pred[pt[sel]] = cands["arc_id"].to_numpy(np.int64)[sel]
    return pred


def decode(trip, cands, p, use_hmm=True, beta=None, lam=None, graph=None):
    """Shared by predict_trip and evaluate: returns (pred_arc, route, breaks, timing dict in ms)."""
    kw = {} if beta is None else {"beta": beta}
    if lam is not None:
        kw["lam"] = lam
    g = graph or Graph(trip)
    t0 = time.perf_counter()
    if use_hmm:
        pred, breaks = viterbi(trip, cands, p, graph=g, **kw)
    else:
        pred, breaks = argmax_per_point(trip, cands, p), []
    t1 = time.perf_counter()
    route = reconstruct_route(trip, pred, graph=g)
    t2 = time.perf_counter()
    return pred, route, breaks, {"viterbi": (t1 - t0) * 1e3, "reconstruct": (t2 - t1) * 1e3}


def predict_trip(split: str, trip_id: str, bundle: dict, use_hmm: bool = True) -> dict:
    """Return {"trip_id", "pred_arc", "route", "route_lonlat", "timing_ms"}; never reads .route.

    bundle=None runs the Gaussian HMM baseline (lam=0, p = ones).
    """
    T0 = time.perf_counter()
    trip = load_trip(split, trip_id, with_route=False)
    T1 = time.perf_counter()
    cands = build_candidates(trip)
    T2 = time.perf_counter()
    if bundle is None:
        p, lam = np.ones(len(cands)), 0.0
    else:
        from .train import predict_proba
        p, lam = predict_proba(bundle, cands), None
    T3 = time.perf_counter()
    pred, route, _, tm = decode(trip, cands, p, use_hmm=use_hmm, lam=lam)
    ll = route_lonlat(trip, route)
    T4 = time.perf_counter()
    return {
        "trip_id": trip_id,
        "pred_arc": pred.tolist(),
        "route": [int(a) for a in route],
        "route_lonlat": ll,
        "timing_ms": {"load": (T1 - T0) * 1e3, "candidates": (T2 - T1) * 1e3,
                      "classifier": (T3 - T2) * 1e3, "viterbi": tm["viterbi"],
                      "reconstruct": tm["reconstruct"], "total": (T4 - T0) * 1e3},
    }
