"""Folium maps of GPS track and predicted route. Owner: B."""
import numpy as np
import folium
from scipy.spatial import cKDTree

from .data import Trip

LEGEND = """
<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
border:1px solid #888;border-radius:4px;font:12px sans-serif;line-height:1.6">
<b>Legend</b><br>
<span style="color:#1f77b4">&#9679;</span> GPS observations<br>
<span style="color:#999">&#9473;</span> Road network (near track)<br>
{rows}
</div>"""


def _ll(trip: Trip, route) -> list[list[float]]:
    r = np.asarray(route, np.int64)
    if len(r) == 0:
        return []
    nodes = np.r_[trip.arcs[r[0], 0], trip.arcs[r, 1]]
    return trip.nodes_lonlat[nodes][:, ::-1].round(6).tolist()  # folium wants lat, lon


def _near_arcs(trip: Trip, radius=100.0, max_arcs=60000) -> np.ndarray:
    pts = trip.xy[:: max(1, len(trip.xy) // 4000)]
    d, _ = cKDTree(pts).query(trip.nodes_xy, distance_upper_bound=radius)
    near = np.isfinite(d)
    a = np.flatnonzero(near[trip.arcs[:, 0]] & near[trip.arcs[:, 1]])
    lo, hi = np.minimum(trip.arcs[a, 0], trip.arcs[a, 1]), np.maximum(trip.arcs[a, 0], trip.arcs[a, 1])
    _, first = np.unique(lo * len(trip.nodes_xy) + hi, return_index=True)  # draw each road once
    return a[first][:max_arcs]


def trip_map(trip: Trip, pred_route: list[int] | None = None, ref_route: np.ndarray | None = None,
             show_network: bool = True, out_html: str | None = None,
             other_routes: dict[str, list[int]] | None = None) -> folium.Map:
    """GPS + network + (reference) + (other_routes) + predicted route, each a toggleable layer."""
    lat, lon = trip.lonlat[:, 1], trip.lonlat[:, 0]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=15, tiles="OpenStreetMap",
                   control_scale=True)
    rows = ""

    if show_network:
        fg = folium.FeatureGroup(name="Road network", show=True)
        arcs = _near_arcs(trip)
        lines = [trip.nodes_lonlat[trip.arcs[a]][:, ::-1].round(6).tolist() for a in arcs]
        if lines:
            folium.PolyLine(lines, color="#999999", weight=1, opacity=0.7).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="GPS observations", show=True)
    ll = np.c_[lat, lon].round(6)
    folium.PolyLine(ll.tolist(), color="#1f77b4", weight=1, opacity=0.4).add_to(fg)
    step = max(1, len(ll) // 1500)
    for q in ll[::step]:
        folium.CircleMarker(q.tolist(), radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m)

    if ref_route is not None and len(ref_route):
        fg = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_ll(trip, ref_route), color="green", weight=5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        rows += '<span style="color:green">&#9473;&#9473;</span> Reference route<br>'
    palette = ["#ff7f0e", "#9467bd", "#8c564b"]
    for i, (name, r) in enumerate((other_routes or {}).items()):
        fg = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(_ll(trip, r), color=palette[i % 3], weight=3, opacity=0.9, dash_array="6").add_to(fg)
        fg.add_to(m)
        rows += f'<span style="color:{palette[i % 3]}">&#9473;&#9473;</span> {name}<br>'
    if pred_route is not None:
        fg = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_ll(trip, pred_route), color="red", weight=3, opacity=0.95).add_to(fg)
        fg.add_to(m)
        rows += '<span style="color:red">&#9473;&#9473;</span> Predicted route<br>'

    fg = folium.FeatureGroup(name="Start / end", show=True)
    folium.Marker(ll[0].tolist(), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker(ll[-1].tolist(), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)

    m.get_root().html.add_child(folium.Element(LEGEND.format(rows=rows)))
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    if out_html:
        m.save(out_html)
    return m


def make_map(trip: Trip, route: list[int]):
    """Back-compat alias of the stub signature: GPS + network + predicted route, no reference."""
    return trip_map(trip, pred_route=route)


def main() -> None:
    """python -m src.maps SPLIT TRIP_ID [bundle.joblib]: writes artifacts/maps/<split>_<trip>.html
    with reference, classifier-only and HMM layers (evaluation use only: loads the reference route)."""
    import sys
    from pathlib import Path

    import numpy as np

    from .data import load_trip
    from .features import build_candidates
    from .predict import decode

    split, tid = sys.argv[1], sys.argv[2]
    trip = load_trip(split, tid, with_route=True)
    cands = build_candidates(trip)
    bundle = None
    if len(sys.argv) > 3:
        import joblib
        from .train import predict_proba
        bundle = joblib.load(sys.argv[3])
        p = predict_proba(bundle, cands)
    else:
        p = np.ones(len(cands))
    other = {}
    if bundle is not None:
        other["Classifier-only route"] = decode(trip, cands, p, use_hmm=False)[1]
    _, route, _, _ = decode(trip, cands, p, use_hmm=True, lam=None if bundle is not None else 0.0)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{split}_{tid}.html"
    trip_map(trip, pred_route=route, ref_route=trip.route, other_routes=other, out_html=str(f))
    print(f, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
