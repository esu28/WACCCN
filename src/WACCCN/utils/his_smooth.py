#!/usr/bin/env python3
# used to smooth his wav coeff tensor

import argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

def estimate_spacing_global_min(coords_xy):
    D = distance_matrix(coords_xy, coords_xy)
    np.fill_diagonal(D, np.inf)
    return float(np.min(D))

def gaussian_disk_smooth(X, coords_xy, radius, sigma):
    H, W, C = X.shape
    Y = np.zeros_like(X, dtype=np.float32)
    Wm = np.zeros((H, W), dtype=np.float32)
    coords = np.asarray(coords_xy, dtype=float)

    for x, y in coords:
        x = int(x); y = int(y)
        if not (0 <= x < W and 0 <= y < H):
            continue
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        d2 = ys[:, None]**2 + xs[None, :]**2
        disk = d2 <= (radius * radius)
        w = np.exp(-d2 / (2.0 * (sigma**2))).astype(np.float32) * disk
        val = X[y, x, :]
        Y[y0:y1, x0:x1, :] += w[:, :, None] * val[None, None, :]
        Wm[y0:y1, x0:x1] += w

    Wm[Wm == 0] = 1.0
    Y /= Wm[:, :, None]
    return Y
