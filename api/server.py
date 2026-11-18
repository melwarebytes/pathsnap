"""FastAPI server for the PathSnap live demo. Owner: C (PLAN section 6.6).

Ground truth is read only inside `score()`, after prediction, and only scalars
leave this module. The pipeline runs live on every /predict request.
"""
import datetime
import inspect
import math
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data import list_trips, load_trip

SPLIT = os.environ.get("PATHSNAP_SPLIT", "evaluator_test").strip()  # cmd "set X=y && ..." keeps a trailing space
MODEL_PATH = os.environ.get("PATHSNAP_MODEL", "artifacts/models/final.joblib").strip()
STUB = os.environ.get("PATHSNAP_STUB", "0").strip().lower() in {"1", "true", "yes"}
METRIC_KEYS = ("top1", "route_f1", "route_precision", "route_recall", "continuity")


def _load_bundle(path):
    """Load the classifier bundle; None if it is missing or unreadable."""
    try:
        return joblib.load(path)
    except Exception as exc:  # FileNotFoundError, corrupt/pickle errors, ...
        print(f"!! WARNING: model not loaded from {path!r} "
              f"({type(exc).__name__}: {exc}); running with bundle=None", flush=True)
        return None


def _discover_trips():
    """List trip ids for SPLIT; [] (with a warning) if the data is absent."""
    try:
        return list_trips(SPLIT)
    except (FileNotFoundError, OSError) as exc:
        print(f"!! WARNING: no trips for split {SPLIT!r} ({type(exc).__name__}: {exc}); "
              f"/predict will return 404 until the data is in place", flush=True)
        return []


BUNDLE = _load_bundle(MODEL_PATH)
TRIPS = _discover_trips()


def _bundle_name():
    """Human-readable model name for /health and the startup line."""
    if isinstance(BUNDLE, dict):
        return str(BUNDLE["name"]) if BUNDLE.get("name") else "unnamed"
    return None if BUNDLE is None else type(BUNDLE).__name__


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Print one startup line, then serve."""
    try:
        from src.data import DATA_ROOT
        root = str(DATA_ROOT)
    except Exception:
        root = "?"
    cv_top1 = BUNDLE.get("cv_top1") if isinstance(BUNDLE, dict) else None
    cv = f"{float(cv_top1):.4f}" if isinstance(cv_top1, (int, float)) else "n/a"
    print(f"PathSnap API: split={SPLIT}  trips={len(TRIPS)}  model={_bundle_name()}  "
          f"cv_top1={cv}  stub={STUB}  data_root={root}", flush=True)
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log exactly one line per request: [ts] ip  METHOD path  trip  status  latency ms."""
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        client = request.client.host if request.client else "-"
        trip = getattr(request.state, "trip_id", "-") or "-"
        latency = (time.perf_counter() - t0) * 1000
        print(f"[{datetime.datetime.now():%H:%M:%S}] {client}  {request.method} "
              f"{request.url.path}  {trip}  {status}  {latency:.0f} ms", flush=True)


class PredictIn(BaseModel):
    trip_id: str


# --- helpers -------------------------------------------------------------

_PREDICT_FN = None
_CACHE = {}                          # trip_id -> predicted route only (never GT)
_CACHE_LOCK = threading.Lock()


def _predict_fn():
    """Import and cache src.predict.predict_trip; 503 if the module is unusable."""
    global _PREDICT_FN
    if _PREDICT_FN is None:
        try:
            from src.predict import predict_trip
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(503, f"src.predict is unavailable ({type(exc).__name__}: "
                                     f"{exc}); the prediction pipeline is not installed yet.")
        _PREDICT_FN = predict_trip
    return _PREDICT_FN


def _plumbing_stub(trip_id):
    """Fallback result used only when PATHSNAP_STUB=1 and the pipeline is a stub."""
    trip = load_trip(SPLIT, trip_id, with_route=False)
    return {"trip_id": trip_id, "pred_arc": [-1] * len(trip.t), "route": [],
            "route_lonlat": trip.lonlat.tolist(), "timing_ms": {}}


def _run_predict(trip_id):
    """Run the live pipeline. Returns (result, inference_ms, stub)."""
    fn = _predict_fn()
    t0 = time.perf_counter()
    stub = False
    try:
        result = fn(SPLIT, trip_id, BUNDLE)
    except NotImplementedError:
        if not STUB:
            raise HTTPException(503, "the prediction pipeline is not implemented yet; "
                                     "set PATHSNAP_STUB=1 to use the plumbing stub.")
        stub = True
        result = _plumbing_stub(trip_id)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"prediction failed: {type(exc).__name__}: {exc}")
    inference_ms = (time.perf_counter() - t0) * 1000
    if not isinstance(result, dict):
        raise HTTPException(500, f"predict_trip returned {type(result).__name__}, expected dict")
    return result, inference_ms, stub


def _scalar(value):
    """numpy scalar -> finite float, or None for None/NaN/Inf/garbage."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lonlat_list(value):
    """Coerce (n,2) arrays/lists to plain JSON-safe [[lon, lat], ...] floats."""
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return []
    if arr.ndim != 2 or arr.shape[1] != 2:
        return []
    return [[float(lon), float(lat)] for lon, lat in arr
            if math.isfinite(lon) and math.isfinite(lat)]


def score(trip_id, result):
    """Score against ground truth. GT is loaded ONLY here, after prediction.

    Returns exactly the five metric keys; each is a float or None. Scoring
    modules that are still stubs (NotImplementedError) yield None instead of
    breaking the demo.
    """
    out = dict.fromkeys(METRIC_KEYS)
    try:
        gt = load_trip(SPLIT, trip_id, with_route=True)
    except (FileNotFoundError, OSError):
        return out
    if gt.route is None or len(gt.route) == 0:
        return out

    try:
        from src.data import align_labels
        label_arc, _ = align_labels(gt)
    except NotImplementedError:
        return out
    except Exception:
        traceback.print_exc()
        return out

    try:
        from src.metrics import point_top1_from_assignment
        pred_arc = np.asarray(result.get("pred_arc") or [], dtype=np.int64)
        if pred_arc.size == label_arc.size:
            out["top1"] = _scalar(point_top1_from_assignment(label_arc, pred_arc))
    except NotImplementedError:
        pass
    except Exception:
        traceback.print_exc()

    try:
        from src.hmm import route_metrics
        rm = route_metrics(gt, list(result.get("route") or []), gt.route)
        if isinstance(rm, dict):
            for key in ("route_f1", "route_precision", "route_recall", "continuity"):
                out[key] = _scalar(rm.get(key))
    except NotImplementedError:
        pass
    except Exception:
        traceback.print_exc()

    return out


def _remember(trip_id, route):
    """Cache the predicted route (arc ids) for /map. Never called with GT."""
    with _CACHE_LOCK:
        _CACHE[trip_id] = route


def _cached_route(trip_id):
    """Cached route for trip_id, or None when no prediction has run yet."""
    with _CACHE_LOCK:
        return _CACHE.get(trip_id)


def _render_map(trip, route):
    """Render folium HTML via src.maps, supporting trip_map and make_map."""
    try:
        import src.maps as maps
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(503, f"src.maps is unavailable ({type(exc).__name__}: {exc}).")

    helper = getattr(maps, "trip_map", None) or getattr(maps, "make_map", None)
    if helper is None:
        raise HTTPException(503, "src.maps exposes neither trip_map nor make_map.")
    try:
        params = inspect.signature(helper).parameters
    except (TypeError, ValueError):
        params = {}

    try:
        if "pred_route" in params:                 # B's trip_map contract
            folium_map = helper(trip, pred_route=route)
        else:                                      # legacy make_map(trip, route)
            folium_map = helper(trip, route)
    except NotImplementedError:
        raise HTTPException(503, f"src.maps.{helper.__name__} is not implemented yet.")
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"map rendering failed: {type(exc).__name__}: {exc}")

    return folium_map if isinstance(folium_map, str) else folium_map.get_root().render()


# --- endpoints -----------------------------------------------------------

@app.get("/trips")
def trips():
    """Trip ids available for the configured split."""
    return {"split": SPLIT, "trips": TRIPS}


@app.get("/health")
def health():
    """Small liveness/config probe (the UI does not need it)."""
    return {"ok": True, "model": _bundle_name(), "split": SPLIT,
            "trips": len(TRIPS), "stub": STUB}


@app.post("/predict")
def predict(body: PredictIn, request: Request):
    """Run the live pipeline for one trip and score it against server-side GT."""
    trip_id = body.trip_id
    if trip_id not in TRIPS:          # exact match also blocks "../train/00000001"
        raise HTTPException(404, "unknown trip_id")
    request.state.trip_id = trip_id

    result, inference_ms, stub = _run_predict(trip_id)
    _remember(trip_id, list(result.get("route") or []))

    timing = result.get("timing_ms") or {}
    if isinstance(timing, dict):
        timing = {str(k): _scalar(v) for k, v in timing.items()}
    else:
        timing = {}

    return {
        "trip_id": trip_id,
        "route_lonlat": _lonlat_list(result.get("route_lonlat")),
        "gps_lonlat": _lonlat_list(load_trip(SPLIT, trip_id, with_route=False).lonlat),
        "metrics": score(trip_id, result),
        "inference_ms": round(inference_ms, 1),
        "timing_ms": timing,
        "stub": stub,
    }


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def map_trip(trip_id: str, request: Request):
    """Folium HTML of GPS + predicted route (no reference layer)."""
    if trip_id not in TRIPS:
        raise HTTPException(404, "unknown trip_id")
    request.state.trip_id = trip_id

    route = _cached_route(trip_id)
    if route is None:
        result, _, _ = _run_predict(trip_id)
        route = list(result.get("route") or [])
        _remember(trip_id, route)

    trip = load_trip(SPLIT, trip_id, with_route=False)
    return HTMLResponse(_render_map(trip, route))
