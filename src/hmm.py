"""HMM/Viterbi map matching, route reconstruction and route metrics. Owner: B."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .data import Trip

U_TURN_PENALTY = 3.0
MAX_GAP_S = 120.0     # dt above this forces a break
LONG_GAP_S = 10.0     # dt above this scales beta by dt
P_FLOOR = 1e-6


class Graph:
    """Per-trip road graph with a shortest-path cache (built once per trip)."""

    def __init__(self, trip: Trip):
        N = len(trip.nodes_xy)
        self.N = N
        self.trip = trip
        frm, to = trip.arcs[:, 0], trip.arcs[:, 1]
        key = frm * N + to
        order = np.lexsort((trip.arc_len, key))  # per (from,to) pair the shortest arc comes first
        ks = key[order]
        first = np.r_[True, ks[1:] != ks[:-1]]
        sel = order[first]
        self.keys, self.arc_of = key[sel], sel
        w = trip.arc_len[sel] + 1e-3  # zero-length arcs stay edges
        self.G = csr_matrix((w, (frm[sel], to[sel])), shape=(N, N))
        self._cache: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}

    def sp_matrix(self, src: np.ndarray, dst: np.ndarray, limit: float) -> np.ndarray:
        """(len(src), len(dst)) shortest-path lengths, inf where farther than limit / unreachable."""
        limit = float(np.ceil(limit / 500.0) * 500.0)
        usrc, inv = np.unique(src, return_inverse=True)
        missing = np.array([s for s in usrc if s not in self._cache or self._cache[s][0] < limit], np.int64)
        for i in range(0, len(missing), 16):
            idx = missing[i:i + 16]
            D = dijkstra(self.G, directed=True, indices=idx, limit=limit)
            D = np.atleast_2d(D)
            for s, row in zip(idx, D):
                fin = np.flatnonzero(np.isfinite(row))
                self._cache[int(s)] = (limit, fin, row[fin])
        out = np.full((len(usrc), len(dst)), np.inf)
        for r, s in enumerate(usrc):
            _, nodes, d = self._cache[int(s)]
            j = np.minimum(np.searchsorted(nodes, dst), len(nodes) - 1)
            ok = nodes[j] == dst
            out[r] = np.where(ok, d[j], np.inf)
        return out[inv]

    def arc_between(self, u: int, v: int) -> int:
        k = u * self.N + v
        i = int(np.searchsorted(self.keys, k))
        return int(self.arc_of[i]) if i < len(self.keys) and self.keys[i] == k else -1

    def path_arcs(self, u: int, v: int, limit: float) -> list[int] | None:
        """Arc ids of a shortest path from node u to node v, [] if u == v, None if unreachable."""
        if u == v:
            return []
        _, pred = dijkstra(self.G, directed=True, indices=u, limit=limit, return_predecessors=True)
        if pred[v] < 0:
            return None
        nodes = [v]
        while nodes[-1] != u:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        return [self.arc_between(a, b) for a, b in zip(nodes[:-1], nodes[1:])]


def _log_trans(trip, g, a, ta, b, tb, gc, dt, beta):
    """(Ku, Kv) log transition scores between the candidates of consecutive kept points."""
    frm, to, ln = trip.arcs[:, 0], trip.arcs[:, 1], trip.arc_len
    fa, ea, fb, eb = frm[a], to[a], frm[b], to[b]
    L = max(500.0, 3 * gc + 200.0, 50.0 * dt)
    sp = g.sp_matrix(ea, fb, L)
    rd = (1 - ta)[:, None] * ln[a][:, None] + sp + (tb * ln[b])[None, :]
    same = (fa[:, None] == fb[None, :]) & (ea[:, None] == eb[None, :])  # same road (duplicates included)
    rd = np.where(same, np.abs(tb[None, :] - ta[:, None]) * ln[a][:, None], rd)
    beta_eff = beta * (dt if dt > LONG_GAP_S else 1.0)
    lt = -np.abs(rd - gc) / beta_eff
    uturn = (fa[:, None] == eb[None, :]) & (ea[:, None] == fb[None, :]) & ~same
    lt = lt - U_TURN_PENALTY * uturn
    return np.where(np.isfinite(rd), lt, -np.inf)


def viterbi(trip: Trip, cands: pd.DataFrame, p: np.ndarray, beta=5.0, lam=1.0, sigma=6.7,
            graph: Graph | None = None) -> tuple[np.ndarray, list[int]]:
    """Return pred_arc (n,) int64 (-1 = no candidates) and the break indices (point indices)."""
    n = len(trip.xy)
    pred = np.full(n, -1, np.int64)
    if len(cands) == 0:
        return pred, []
    pt = cands["pt_idx"].to_numpy(np.int64)
    arc = cands["arc_id"].to_numpy(np.int64)
    dist = cands["dist"].to_numpy(np.float64)
    tf = cands["proj_frac"].to_numpy(np.float64)
    p = np.asarray(p, np.float64)
    emis = lam * np.log(np.clip(p, P_FLOOR, 1.0)) + (1 - lam) * (-0.5 * (dist / sigma) ** 2)
    starts = np.r_[0, np.flatnonzero(np.diff(pt)) + 1]
    ends = np.r_[starts[1:], len(pt)]
    pts = pt[starts]
    m = len(pts)
    g = graph or Graph(trip)

    sl = lambda k: slice(starts[k], ends[k])
    delta = emis[sl(0)].copy()
    bps: list[np.ndarray | None] = [None] * m
    seg_end: dict[int, int] = {}
    breaks: list[int] = []
    for k in range(1, m):
        u, v = pts[k - 1], pts[k]
        dt = float(trip.t[v] - trip.t[u])
        ev = emis[sl(k)]
        new = None
        if dt <= MAX_GAP_S:
            gc = float(np.linalg.norm(trip.xy[v] - trip.xy[u]))
            lt = _log_trans(trip, g, arc[sl(k - 1)], tf[sl(k - 1)], arc[sl(k)], tf[sl(k)], gc, dt, beta)
            score = delta[:, None] + lt
            bp = score.argmax(0)
            best = score[bp, np.arange(score.shape[1])]
            if np.isfinite(best).any():
                new, bps[k] = best + ev, bp
        if new is None:  # disconnected / huge gap: close the segment, restart with emissions only
            seg_end[k - 1] = int(delta.argmax())
            breaks.append(int(v))
            new = ev.copy()
        delta = new

    s = int(delta.argmax())
    for k in range(m - 1, -1, -1):
        pred[pts[k]] = arc[starts[k] + s]
        if k == 0:
            break
        s = int(bps[k][s]) if bps[k] is not None else seg_end[k - 1]
    return pred, breaks


def reconstruct_route(trip: Trip, pred_arc: np.ndarray, graph: Graph | None = None) -> list[int]:
    """Full arc sequence from per-point arcs, with shortest-path gap filling."""
    seq: list[int] = []
    frm, to = trip.arcs[:, 0], trip.arcs[:, 1]
    for a in pred_arc[pred_arc >= 0]:
        a = int(a)
        if seq and (a == seq[-1] or (frm[a] == frm[seq[-1]] and to[a] == to[seq[-1]])):
            continue
        seq.append(a)
    if not seq:
        return []
    g = graph or Graph(trip)
    route = [seq[0]]
    for a in seq[1:]:
        u, v = int(to[route[-1]]), int(frm[a])
        if u != v:
            eu = float(np.linalg.norm(trip.nodes_xy[u] - trip.nodes_xy[v]))
            path = g.path_arcs(u, v, min(max(1000.0, 4 * eu + 500.0), 30000.0))
            if path:
                route.extend(x for x in path if x >= 0)
        route.append(a)
    return route


def route_lonlat(trip: Trip, route: list[int]) -> list[list[float]]:
    if not route:
        return []
    pts = [trip.nodes_lonlat[trip.arcs[route[0], 0]]] + [trip.nodes_lonlat[trip.arcs[a, 1]] for a in route]
    return np.asarray(pts, np.float64).tolist()


def route_metrics(trip: Trip, pred_route: list[int], ref_route: np.ndarray) -> dict:
    """Keys: route_precision, route_recall, route_f1 (length-weighted), continuity, n_breaks."""
    pred = np.unique(np.asarray(pred_route, np.int64))
    ref = np.unique(np.asarray(ref_route, np.int64))
    inter = np.intersect1d(pred, ref)
    ln = trip.arc_len
    lp, lr, li = ln[pred].sum(), ln[ref].sum(), ln[inter].sum()
    prec = li / lp if lp > 0 else 0.0
    rec = li / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    r = np.asarray(pred_route, np.int64)
    if len(r) > 1:
        bad = int((trip.arcs[r[:-1], 1] != trip.arcs[r[1:], 0]).sum())
        cont = 1.0 - bad / (len(r) - 1)
    else:
        bad, cont = 0, 1.0
    return {"route_precision": float(prec), "route_recall": float(rec), "route_f1": float(f1),
            "continuity": float(cont), "n_breaks": bad}


def _self_check() -> None:
    """Toy graph: the HMM must take the connected path over an emission-favoured disconnected arc."""
    nodes = np.array([[0, 0], [100, 0], [200, 0], [120, 0.2], [180, 0.2]], float)
    arcs = np.array([[0, 1], [1, 2], [3, 4]], np.int64)
    ln = np.linalg.norm(nodes[arcs[:, 1]] - nodes[arcs[:, 0]], axis=1)
    xy = np.array([[50, 1.0], [140, 1.0], [160, 1.0]])
    trip = Trip("toy", np.arange(3, dtype=float), xy, xy, (0.0, 0.0), nodes, nodes, arcs, ln, None)
    # point 0 only on arc 0; points 1,2 on arc 1 (dist 1.0) or on disconnected arc 2 (dist 0.8)
    rows = [(0, 0, 1.0, .5), (1, 1, 1.0, .4), (1, 2, 0.8, .33), (2, 2, 0.8, .67), (2, 1, 1.0, .6)]
    c = pd.DataFrame(rows, columns=["pt_idx", "arc_id", "dist", "proj_frac"])
    pa, br = viterbi(trip, c, np.ones(len(c)), lam=0.0)
    assert pa.tolist() == [0, 1, 1], pa
    route = reconstruct_route(trip, np.array([0, 1, 1]))
    assert route == [0, 1], route
    m = route_metrics(trip, route, np.array([0, 1]))
    assert m["route_f1"] == 1.0 and m["continuity"] == 1.0 and m["n_breaks"] == 0, m
    # gap filling: skipping the middle arc must be repaired
    nodes2 = np.array([[0, 0], [100, 0], [200, 0], [300, 0]], float)
    arcs2 = np.array([[0, 1], [1, 2], [2, 3]], np.int64)
    ln2 = np.linalg.norm(nodes2[arcs2[:, 1]] - nodes2[arcs2[:, 0]], axis=1)
    t2 = Trip("toy2", np.arange(2, dtype=float), xy[:2], xy[:2], (0.0, 0.0), nodes2, nodes2, arcs2, ln2, None)
    assert reconstruct_route(t2, np.array([0, 2])) == [0, 1, 2]
    print("hmm self-check OK")


if __name__ == "__main__":
    _self_check()
