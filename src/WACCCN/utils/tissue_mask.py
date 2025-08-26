"""
- Mask: (S >= S_floor) OR (V <= V_ceiling), while rejecting near-white via gray_pct.
- You must tune S_floor / V_ceiling / gray_pct per slide and check the overlay by eye.
- Pass coordinates to rescue missed capture spots locally (small disks) (optional).
"""

import numpy as np
from pathlib import Path
from skimage import io, color, filters, morphology, transform, util, measure

__all__ = ["build_mask_sv", "make_tissue_mask", "overlay_edges", "resize_mask"]

def _ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.dtype.kind == "f":
        x = np.clip(img, 0, 1)
    else:
        x = img.astype(float) / 255.0
    return (x * 255).astype(np.uint8)

def _cleanup(mask, min_obj_frac, min_hole_frac, keep_largest):
    H, W = mask.shape
    area = H * W
    mask = morphology.remove_small_objects(mask, max(1, int(min_obj_frac * area)))
    mask = morphology.remove_small_holes(mask,     max(1, int(min_hole_frac * area)))
    if keep_largest:
        lab = measure.label(mask, connectivity=1)
        if lab.max() > 0:
            sizes = np.bincount(lab.ravel())
            keep = np.argmax(sizes[1:]) + 1
            mask = (lab == keep)
    return mask

def build_mask_sv(
    img_rgb,
    blur_sigma=1.0,
    S_floor=0.18,
    V_ceiling=0.67,
    gray_pct=96.0,
    min_obj_frac=3e-5,
    min_hole_frac=8e-4,
    close_r=2,
    keep_largest=False,
):
    imgf = util.img_as_float(img_rgb[..., :3])
    hsv  = color.rgb2hsv(imgf)
    Sg   = filters.gaussian(hsv[..., 1], sigma=blur_sigma, preserve_range=True)
    Vg   = filters.gaussian(hsv[..., 2], sigma=blur_sigma, preserve_range=True)
    gray = color.rgb2gray(imgf)
    gcut = np.percentile(gray, gray_pct)
    mask = ((Sg > S_floor) | (Vg < V_ceiling)) & (gray < gcut)
    if close_r > 0:
        mask = morphology.binary_closing(mask, morphology.disk(close_r))
    mask = _cleanup(mask, min_obj_frac, min_hole_frac, keep_largest)
    return mask

def resize_mask(mask_src, target_hw):
    Ht, Wt = target_hw
    out = transform.resize(mask_src.astype(float), (Ht, Wt), order=0, preserve_range=True, anti_aliasing=False)
    return out.astype(bool)

def overlay_edges(img_rgb, mask, edge_color=(255, 0, 0)):
    vis = _ensure_uint8(img_rgb.copy())
    if vis.ndim == 2:
        vis = np.repeat(vis[..., None], 3, axis=2)
    edge = morphology.binary_dilation(mask, morphology.disk(1)) ^ mask
    vis[edge] = np.array(edge_color, np.uint8)
    return vis

# ---- optional spot rescue ----
def _load_xy(coords_csv):
    try:
        import pandas as pd
        df = pd.read_csv(coords_csv)
        for cx, cy in [("x","y"),("X","Y"),("img_x","img_y"),("imgX","imgY"),
                       ("col","row"),("Col","Row"),("j","i"),("J","I")]:
            if cx in df.columns and cy in df.columns:
                return df[cx].to_numpy(float), df[cy].to_numpy(float)
        arr = df.to_numpy()
        return arr[:,1].astype(float), arr[:,2].astype(float)
    except Exception:
        arr = np.genfromtxt(coords_csv, delimiter=",", names=True, dtype=None, encoding=None)
        names = list(arr.dtype.names) if arr.dtype.names else []
        for cx, cy in [("x","y"),("X","Y"),("img_x","img_y"),("imgX","imgY"),
                       ("col","row"),("Col","Row"),("j","i"),("J","I")]:
            if cx in names and cy in names:
                return np.asarray(arr[cx], float), np.asarray(arr[cy], float)
        data = np.genfromtxt(coords_csv, delimiter=",", skip_header=1)
        return data[:,1].astype(float), data[:,2].astype(float)

def _map_to_grid(x, y, H, W):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.max() <= 1.0 + 1e-6 and y.max() <= 1.0 + 1e-6:
        x = x * (W - 1); y = y * (H - 1)
    if x.max() > W - 1 or y.max() > H - 1:
        x = (x - x.min()) / (x.max() - x.min() + 1e-9) * (W - 1)
        y = (y - y.min()) / (y.max() - y.min() + 1e-9) * (H - 1)
    return x, y

def _estimate_spacing(x, y):
    xs = np.sort(x); ys = np.sort(y)
    dx = np.diff(xs); dy = np.diff(ys)
    diffs = np.hstack([dx[dx > 0], dy[dy > 0]])
    return float(np.median(diffs)) if diffs.size > 0 else 10.0

def _rescue_spots(mask, x, y, radius_px=None, radius_mul=0.6):
    H, W = mask.shape
    x, y = _map_to_grid(x, y, H, W)
    rr = np.clip(np.round(y).astype(int), 0, H-1)
    cc = np.clip(np.round(x).astype(int), 0, W-1)
    covered = mask[rr, cc]
    if not np.any(~covered):
        return mask
    if radius_px is None:
        R = max(1, int(round(_estimate_spacing(x, y) * radius_mul)))
    else:
        R = max(1, int(radius_px))
    pts = np.zeros_like(mask, dtype=bool)
    pts[rr[~covered], cc[~covered]] = True
    disks = morphology.binary_dilation(pts, morphology.disk(R))
    return mask | disks

def make_tissue_mask(
    he_path,
    out_npy,
    out_overlay,
    target_hw=None,
    blur_sigma=1.0,
    S_floor=0.18,
    V_ceiling=0.67,
    gray_pct=96.0,
    min_obj_frac=3e-5,
    min_hole_frac=8e-4,
    close_r=2,
    keep_largest=False,
    coords_csv=None,
    rescue_spots=False,
    spot_radius_px=None,
    spot_radius_mul=0.6,
):
    img = io.imread(str(he_path))
    mask = build_mask_sv(
        img,
        blur_sigma=blur_sigma,
        S_floor=S_floor,
        V_ceiling=V_ceiling,
        gray_pct=gray_pct,
        min_obj_frac=min_obj_frac,
        min_hole_frac=min_hole_frac,
        close_r=close_r,
        keep_largest=keep_largest,
    )
    img_for_overlay = img
    if target_hw is not None:
        mask = resize_mask(mask, target_hw)
        img_for_overlay = _ensure_uint8(transform.resize(img, mask.shape[:2], order=1, preserve_range=True, anti_aliasing=True))
    if coords_csv is not None and rescue_spots:
        x, y = _load_xy(coords_csv)
        mask = _rescue_spots(mask, x, y, radius_px=spot_radius_px, radius_mul=spot_radius_mul)
    Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, mask.astype(bool))
    vis = overlay_edges(img_for_overlay, mask)
    Path(out_overlay).parent.mkdir(parents=True, exist_ok=True)
    io.imsave(out_overlay, vis)
    return mask, vis
