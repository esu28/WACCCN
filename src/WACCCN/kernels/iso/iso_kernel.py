#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def load_coords_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ("barcode", "x", "y")
    if any(k not in cols for k in need):
        raise ValueError("coords CSV must have columns: barcode, x, y (case-insensitive).")
    bcol, xcol, ycol = cols["barcode"], cols["x"], cols["y"]
    barcodes = df[bcol].astype(str).values
    coords = df[[xcol, ycol]].to_numpy(float)
    return coords, barcodes

def median_1nn(coords):
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    return float(np.median(d[:, 1]))

def edge_distances(coords, edges):
    vi = coords[edges[:, 0]]
    vj = coords[edges[:, 1]]
    return np.linalg.norm(vi - vj, axis=1)

def per_source_normalize_K(K, edges, N, eps=1e-12):
    out = K.copy()
    src = edges[:, 0]
    sums = np.bincount(src, weights=K, minlength=N)
    cnts = np.bincount(src, minlength=N).astype(float)
    denom = sums / np.maximum(cnts, 1.0)
    out /= np.maximum(denom[src], eps)
    return out

def build_iso_from_aniso(coords_csv, aniso_npz, out_path, kappa=1.0, normK=False):
    coords, barcodes = load_coords_csv(coords_csv)
    Z = np.load(aniso_npz, allow_pickle=True)
    if "edges" not in Z.files:
        raise ValueError("aniso NPZ must contain 'edges' (E,2) directed.")
    edges = Z["edges"].astype(int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("'edges' must be shape (E,2)")

    N = coords.shape[0]
    h = median_1nn(coords)
    sigma = max(float(kappa) * h, 1e-12)
    d = edge_distances(coords, edges)

    K = np.exp(-(d ** 2) / (2.0 * (sigma ** 2))).astype(np.float64)
    if normK:
        K = per_source_normalize_K(K, edges, N)

    np.savez_compressed(
        out_path,
        edges=edges.astype(np.int32),
        kernel=K.astype(np.float32),
        K=K.astype(np.float32),
        h=float(h),
    )
    return dict(E=int(edges.shape[0]), h=float(h), sigma=float(sigma), normK=bool(normK))
