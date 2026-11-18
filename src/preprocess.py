"""Picklable preprocessing used inside the LR/MLP pipelines (must live outside the __main__ module)."""
import numpy as np

from .features import FEATURES

HEAVY_TAIL = [FEATURES.index(c) for c in ("dt_prev", "dt_next", "arc_len", "edge_proj")]


def log1p_heavy(X):
    """log1p on the heavy-tailed columns (gaps up to 153,064 s, arcs up to 9 km)."""
    X = np.array(X, np.float32, copy=True)
    X[:, HEAVY_TAIL] = np.log1p(np.maximum(X[:, HEAVY_TAIL], 0))
    return X
