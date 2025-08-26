import argparse
import numpy as np
import pandas as pd

def load_coords_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for k in ("barcode", "x", "y"):
        if k not in cols:
            raise ValueError("coords CSV must have columns: barcode, x, y")
    barcodes = df[cols["barcode"]].astype(str).to_numpy()
    xs = df[cols["x"]].to_numpy(dtype=float)
    ys = df[cols["y"]].to_numpy(dtype=float)
    return barcodes, xs, ys


def ensure_hwc(T):
    """Return (H,W,C) layout; accept HWC/CHW/HCW."""
    if T.ndim != 3:
        raise ValueError(f"Tensor must be 3D, got shape {T.shape}")

    if T.shape[2] <= 64 and T.shape[0] >= 4 and T.shape[1] >= 4:
        return T, "HWC"
    if T.shape[0] <= 64 and T.shape[1] >= 4 and T.shape[2] >= 4:
        return np.moveaxis(T, 0, -1), "CHW→HWC"
    if T.shape[1] <= 64 and T.shape[0] >= 4 and T.shape[2] >= 4:
        return np.moveaxis(T, 1, -1), "HCW→HWC"
    return T, "assumed-HWC"



def sample_on_grid(T, xs, ys):
    """Sample T[y, x, :] at nearest pixels; clamp to bounds."""
    H, W, C = T.shape
    xi = np.rint(xs).astype(int)
    yi = np.rint(ys).astype(int)
    xic = np.clip(xi, 0, W - 1)
    yic = np.clip(yi, 0, H - 1)
    n_clipped = int((xic != xi).sum() + (yic != yi).sum())
    X = T[yic, xic, :].astype(np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, n_clipped
