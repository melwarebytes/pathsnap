"""Trip loading, local projection and per-point label alignment. Owner: Merull (lead)."""
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

DATA_ROOT = Path(str(os.environ.get("PATHSNAP_DATA", Path(__file__).resolve().parents[2])).strip())
EXCLUDED_TRIPS = {"00000033", "00000054"}  # bad ground truth, see docs/PLAN.md section 1
_M_PER_DEG = np.pi / 180 * 6371008.8


class Trip(NamedTuple):
    trip_id: str
    t: np.ndarray             # (n,) float seconds
    lonlat: np.ndarray        # (n,2) GPS lon,lat
    xy: np.ndarray            # (n,2) metres, local projection
    origin: tuple             # (lat0, lon0)
    nodes_lonlat: np.ndarray  # (N,2)
    nodes_xy: np.ndarray      # (N,2)
    arcs: np.ndarray          # (M,2) int64 from,to
    arc_len: np.ndarray       # (M,) metres
    route: np.ndarray | None  # (R,) int64 arc ids, None if not loaded


def list_trips(split: str) -> list[str]:
    return sorted(p.name for p in (DATA_ROOT / split).iterdir() if p.is_dir())


def _read(path: Path, dtype) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype).to_numpy()


def project(lonlat: np.ndarray, origin: tuple) -> np.ndarray:
    lat0, lon0 = origin
    lonlat = np.asarray(lonlat, dtype=np.float64)
    return np.c_[(lonlat[:, 0] - lon0) * _M_PER_DEG * np.cos(np.radians(lat0)),
                 (lonlat[:, 1] - lat0) * _M_PER_DEG]


def load_trip(split: str, trip_id: str, with_route: bool = True) -> Trip:
    base = DATA_ROOT / split / trip_id / trip_id
    track = _read(base.with_suffix(".track"), np.float64)
    nodes = _read(base.with_suffix(".nodes"), np.float64)
    arcs = _read(base.with_suffix(".arcs"), np.int64)
    route = None
    if with_route and base.with_suffix(".route").exists():
        route = _read(base.with_suffix(".route"), np.int64).ravel()
    lonlat = track[:, :2]
    origin = (float(lonlat[:, 1].mean()), float(lonlat[:, 0].mean()))
    nodes_xy = project(nodes, origin)
    arc_len = np.linalg.norm(nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]], axis=1)
    return Trip(trip_id, track[:, 2], lonlat, project(lonlat, origin), origin,
                nodes, nodes_xy, arcs, arc_len, route)


def seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from points P to segments A->B and projection fraction in [0,1]; broadcasts like numpy."""
    AB = B - A
    L2 = (AB * AB).sum(-1)
    t = np.clip(((P - A) * AB).sum(-1) / np.where(L2 == 0, 1.0, L2), 0.0, 1.0)
    return np.linalg.norm(P - (A + AB * t[..., None]), axis=-1), t


def bearing(d: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(d[:, 0], d[:, 1]))


def kinematics(xy: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (speed m/s, heading deg) from a centred difference (one-sided at the ends).

    speed is 0 where the time step is <= 0 and clipped at 70 m/s.
    """
    i = np.arange(len(xy))
    prv, nxt = np.maximum(i - 1, 0), np.minimum(i + 1, len(xy) - 1)
    dxy, dt = xy[nxt] - xy[prv], t[nxt] - t[prv]
    speed = np.where(dt > 0, np.linalg.norm(dxy, axis=1) / np.where(dt > 0, dt, 1.0), 0.0)
    return np.minimum(speed, 70.0), bearing(dxy)


def align_labels(trip: Trip, max_dist: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic DP alignment of points to route arcs (PLAN section 3).

    Cost = min(dist, max_dist) + 2*(1-cos(heading - arc bearing)) for points moving >= 1 m/s, so that
    reverse twins (identical distance) are told apart at U-turns.
    Returns label_arc (n,) int64, -1 where the aligned arc is farther than max_dist,
    and label_dist (n,) distance to the aligned arc.
    """
    route = trip.route
    n, m = len(trip.xy), len(route)
    if n == 0 or m == 0:
        return np.full(n, -1, np.int64), np.full(n, np.inf)
    A = trip.nodes_xy[trip.arcs[route, 0]].astype(np.float32)
    B = trip.nodes_xy[trip.arcs[route, 1]].astype(np.float32)
    P = trip.xy.astype(np.float32)
    speed, heading = kinematics(trip.xy, trip.t)
    h = np.where(speed >= 1.0, heading, np.nan).astype(np.float32)
    ab = bearing(B - A)
    Cc = np.empty((n, m), np.float32)
    for s in range(0, n, 1024):  # chunked so temporaries stay small on 19k x 2.3k trips
        c = np.minimum(seg_dist(P[s:s + 1024, None, :], A[None], B[None])[0], max_dist)
        hc = h[s:s + 1024, None]
        c += np.where(np.isnan(hc), 0.0, 2 * (1 - np.cos(np.radians(hc - ab[None]))))
        Cc[s:s + 1024] = c

    bp = np.zeros((n, m), np.int16 if m < 32767 else np.int32)
    idx = np.arange(m)
    D = Cc[0].astype(np.float64)
    for i in range(1, n):
        cm = np.minimum.accumulate(D)
        better = np.empty(m, bool)
        better[0] = True
        better[1:] = D[1:] < cm[:-1]  # strict: a new prefix minimum starts here
        bp[i] = np.maximum.accumulate(np.where(better, idx, 0))
        D = Cc[i] + cm
    j = int(D.argmin())
    lab = np.empty(n, np.int64)
    for i in range(n - 1, -1, -1):
        lab[i] = j
        j = bp[i, j]
    dist = seg_dist(trip.xy, trip.nodes_xy[trip.arcs[route[lab], 0]], trip.nodes_xy[trip.arcs[route[lab], 1]])[0]
    return np.where(dist <= max_dist, route[lab], -1), dist
