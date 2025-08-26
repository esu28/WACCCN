import os, re, glob, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

eps = 1e-12

def to_xy(coords):
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] not in (2,3):
        raise ValueError("coords must be (N,2) [x,y] or (N,3) [*,x,y]")
    return arr[:, -2:].astype(float)

def load_coords(path):
    if path.endswith(".npy"):
        return np.load(path)
    import pandas as pd
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    xname = next((lc.get(c) for c in ["x","x_px","coord_x","pos_x","xc","col","imagecol","x_um","xpix","x_pixel"]), None)
    yname = next((lc.get(c) for c in ["y","y_px","coord_y","pos_y","yc","row","imagerow","y_um","ypix","y_pixel"]), None)
    if xname and yname:
        arr = df[[xname, yname]].to_numpy(dtype=float)
    else:
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            raise ValueError(f"No numeric x,y columns found in {path}")
        arr = num.iloc[:, -2:].to_numpy(dtype=float)
    return arr

def median_nn_distance(coords):
    xy = to_xy(coords)
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    return float(np.median(d[:,1]))

def build_radius_graph(coords, R_rel=2.2, deg_min=10, deg_max=18, h=None, Rmax_rel=None):
    xy = to_xy(coords)
    if h is None:
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=2)
        h = float(np.median(d[:,1]))
    R = R_rel * h
    tree = cKDTree(xy)

    neighbors = tree.query_ball_point(xy, r=R)
    dists_all, idxs_all = tree.query(xy, k=min(xy.shape[0], deg_max + 30))
    Lcap = (float(Rmax_rel) * h) if (Rmax_rel is not None) else np.inf
    pairs = []

    for i, idxs in enumerate(neighbors):
        idxs = [j for j in idxs if j != i]
        if idxs:
            diff = xy[np.array(idxs)] - xy[i]
            dij  = np.linalg.norm(diff, axis=1)
            order = np.argsort(dij)
            idxs = list(np.array(idxs)[order])
            dij  = dij[order]
        else:
            dij = np.array([], dtype=float)

        if np.isfinite(Lcap) and len(idxs) > 0:
            keep = dij <= Lcap
            idxs = list(np.array(idxs)[keep])

        if len(idxs) < deg_min:
            extras = []
            for j in idxs_all[i, 1:]:
                if j == i or j in idxs:
                    continue
                d = np.linalg.norm(xy[j] - xy[i])
                if d <= Lcap:
                    extras.append(j)
                if len(idxs) + len(extras) >= deg_min:
                    break
            idxs.extend(extras)

        if len(idxs) > deg_max:
            idxs = idxs[:deg_max]

        for j in idxs:
            pairs.append((i, j))

    return np.asarray(pairs, dtype=int), h, R

def parse_angle_from_name(fn):
    m = re.search(r'ang([\-]?\d+)', os.path.basename(fn))
    if not m: raise ValueError(f"cannot parse angle from {fn}")
    return int(m.group(1))

def p99_median_normalize_map(X, mask=None):
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[..., None]
    H, W, C = X.shape
    if mask is None:
        mask = np.ones((H,W), dtype=bool)
    out = np.empty_like(X)
    for c in range(C):
        vals = X[...,c]
        msel = mask & np.isfinite(vals)
        if not np.any(msel):
            out[...,c] = np.nan
            continue
        hi = np.percentile(vals[msel], 99.0)
        hi = 0.0 if (not np.isfinite(hi)) else hi
        vals = np.minimum(vals, hi)
        med = np.median(vals[msel])
        if (not np.isfinite(med)) or med < 1e-12:
            med = 1.0
        out[...,c] = vals / (med + 1e-12)
    return out if out.shape[2] > 1 else out[...,0]

def halfdisk_pixel_weights(ci, u, r_pool, sigma, H, W, tissue_mask=None):
    cx, cy = ci
    x0 = max(0, int(np.floor(cx - r_pool)))
    x1 = min(W-1, int(np.ceil(cx + r_pool)))
    y0 = max(0, int(np.floor(cy - r_pool)))
    y1 = min(H-1, int(np.ceil(cy + r_pool)))
    xs = np.arange(x0, x1+1)
    ys = np.arange(y0, y1+1)
    if xs.size == 0 or ys.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    XX, YY = np.meshgrid(xs, ys)
    dx = XX - cx
    dy = YY - cy
    dist2 = dx*dx + dy*dy
    forward = (dx*u[0] + dy*u[1]) >= 0.0
    inside = dist2 <= (r_pool * r_pool)
    mask = inside & forward
    if tissue_mask is not None:
        mask &= (tissue_mask[YY, XX])
    if not np.any(mask):
        mask = inside
        if tissue_mask is not None:
            mask &= tissue_mask[YY, XX]
        if not np.any(mask):
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    w = np.exp(-dist2 / (2.0 * sigma * sigma)) * mask
    wsum = w.sum()
    if wsum <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    w = w / (wsum + 1e-12)
    ys_idx = YY[mask].ravel()
    xs_idx = XX[mask].ravel()
    w_flat = w[mask].ravel()
    return ys_idx, xs_idx, w_flat

def corridor_pixel_weights(ci, cj, H, W, tissue_mask=None,
                           sigma_perp=1.0, sigma_para=1.0, lateral_mult=3.0):
    ci = np.asarray(ci, float); cj = np.asarray(cj, float)
    v = cj - ci
    d = float(np.linalg.norm(v))
    if d < 1e-9:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    u = v / d
    e2 = np.array([-u[1], u[0]])

    lat = lateral_mult * sigma_perp + 1.0
    x_min = int(np.floor(min(ci[0], cj[0]) - lat))
    x_max = int(np.ceil (max(ci[0], cj[0]) + lat))
    y_min = int(np.floor(min(ci[1], cj[1]) - lat))
    y_max = int(np.ceil (max(ci[1], cj[1]) + lat))
    x_min = max(0, x_min); y_min = max(0, y_min)
    x_max = min(W-1, x_max); y_max = min(H-1, y_max)
    if x_max < x_min or y_max < y_min:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    xs = np.arange(x_min, x_max+1)
    ys = np.arange(y_min, y_max+1)
    XX, YY = np.meshgrid(xs, ys)
    dx = XX - ci[0]
    dy = YY - ci[1]
    s   = dx * u[0] + dy * u[1]
    ell = dx * e2[0] + dy * e2[1]

    sel = (s >= 0.0) & (s <= d) & (np.abs(ell) <= lateral_mult * sigma_perp)
    if tissue_mask is not None:
        sel &= tissue_mask[YY, XX]
    if not np.any(sel):
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    w = np.exp(- (ell**2) / (2.0 * sigma_perp * sigma_perp))
    if sigma_para > 0:
        w *= np.exp(- ((s - 0.5*d)**2) / (2.0 * sigma_para * sigma_para))
    w *= sel
    wsum = float(w.sum())
    if wsum <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    w = w / (wsum + 1e-12)

    ys_idx = YY[sel].ravel().astype(int)
    xs_idx = XX[sel].ravel().astype(int)
    w_flat = w[sel].ravel().astype(float)
    return ys_idx, xs_idx, w_flat

def receiver_cap_weights(cj, u_ij, r_cap, sigma_cap, half_angle_deg, H, W, tissue_mask=None):
    cx, cy = float(cj[0]), float(cj[1])
    x0 = max(0, int(np.floor(cx - r_cap))); x1 = min(W-1, int(np.ceil(cx + r_cap)))
    y0 = max(0, int(np.floor(cy - r_cap))); y1 = min(H-1, int(np.ceil(cy + r_cap)))
    if x1 < x0 or y1 < y0:
        return np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=float)

    xs = np.arange(x0, x1+1); ys = np.arange(y0, y1+1)
    XX, YY = np.meshgrid(xs, ys)
    dx = XX - cx; dy = YY - cy
    r2 = dx*dx + dy*dy
    inside = r2 <= (r_cap*r_cap)

    vnorm = np.maximum(np.sqrt(r2), 1e-12)
    dir_to_pixels_x = dx / vnorm
    dir_to_pixels_y = dy / vnorm
    mdir = np.array([-u_ij[0], -u_ij[1]], dtype=float)
    cosang = dir_to_pixels_x*mdir[0] + dir_to_pixels_y*mdir[1]
    gate = cosang >= np.cos(np.deg2rad(half_angle_deg))

    sel = inside & gate
    if tissue_mask is not None:
        sel &= tissue_mask[YY, XX]
    if not np.any(sel):
        return np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=float)

    w = np.exp(-r2/(2.0*sigma_cap*sigma_cap)) * sel
    wsum = float(w.sum())
    if wsum <= 0:
        return np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=float)
    w = w/(wsum + 1e-12)

    ys_idx = YY[sel].ravel().astype(int)
    xs_idx = XX[sel].ravel().astype(int)
    w_flat = w[sel].ravel().astype(float)
    return ys_idx, xs_idx, w_flat

def combine_weights(ys1, xs1, w1, ys2, xs2, w2, W):
    if w1.size == 0 and w2.size == 0:
        return ys1, xs1, w1
    if w2.size == 0:
        return ys1, xs1, w1
    if w1.size == 0:
        return ys2, xs2, w2
    ys = np.concatenate([ys1, ys2]); xs = np.concatenate([xs1, xs2]); ww = np.concatenate([w1, w2])
    keys = ys.astype(np.int64)*W + xs.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    wsum = np.zeros_like(uniq, dtype=float)
    np.add.at(wsum, inv, ww)
    ys_u = (uniq // W).astype(int); xs_u = (uniq %  W).astype(int)
    wsum /= (wsum.sum() + 1e-12)
    return ys_u, xs_u, wsum

def apply_sigma_floor(Dpar, Dperp, h, sigma_floor_rel=0.68):
    floor_D = float(sigma_floor_rel**2)
    if Dpar <= Dperp:
        Dpar = max(Dpar, floor_D)
    else:
        Dperp = max(Dperp, floor_D)
    return Dpar, Dperp

def edge_pool_params(ci, cj, h, policy="fixed", r_pool_rel=2.2, sigma_rel=2.2,
                     k_edge=1.0, rmin_rel=0.8, rmax_rel=1.2, c_sigma=0.73,
                     rcap_rel=3.0):
    v = cj - ci
    d = float(np.linalg.norm(v))
    if policy == "fixed":
        r_pool = r_pool_rel * h
        sigma  = sigma_rel  * h
        return d, r_pool, sigma
    elif policy == "edge":
        r_pool = max(k_edge * d, 1e-9)
        sigma  = max(c_sigma * r_pool, 1e-9)
        return d, r_pool, sigma
    elif policy == "edge_clip":
        r_min = rmin_rel * h
        r_max = rmax_rel * h
        r_pool = float(np.clip(k_edge * d, r_min, r_max))
        sigma  = max(c_sigma * r_pool, 1e-9)
        return d, r_pool, sigma
    elif policy == "adaptive":
        r_cap = rcap_rel * h
        r_pool = min(max(k_edge * d, 1e-9), r_cap)
        sigma  = max(c_sigma * r_pool, 1e-9)
        return d, r_pool, sigma
    else:
        raise ValueError(f"unknown r_pool policy: {policy}")

def run_wv2_pixel(coords, mask, LH1, LH2, HL1, HL2, HH1, HH2,
                  out_path, R_rel=2.2, deg_min=10, deg_max=18, Rmax_rel=None,
                  r_pool_rel=2.2, sigma_rel=2.2, w1=0.7, w2=0.3,
                  eta=0.5, a_min=1.2, a_max=3.0, gamma=1.5, sigma_floor_rel=0.68,
                  pool_mode="half", r_pool_policy="fixed", k_edge=1.0, rmin_rel=0.8, rmax_rel=1.2, c_sigma=0.73,
                  tube_perp_rel=0.9, tube_ax_rel=1.0, tube_lateral_mult=3.0,
                  receiver_cap_rel=1.2, receiver_cap_half_angle=20.0):
    xy = to_xy(coords)

    def HW(a):
        a = np.asarray(a)
        if a.ndim == 2: return a.shape
        if a.ndim == 3: return a.shape[:2]
        raise ValueError("subbands must be 2D or 3D (H,W[,3])")
    H, W = HW(LH1)
    for arr in [LH2, HL1, HL2, HH1, HH2]:
        if HW(arr) != (H, W):
            raise ValueError("All subbands must share the same (H,W)")
    if mask is not None:
        if mask.shape != (H, W):
            raise ValueError("mask shape mismatch")
        mask = mask.astype(bool)

    P_LH = w1*(np.asarray(LH1, float)**2) + w2*(np.asarray(LH2, float)**2)
    P_HL = w1*(np.asarray(HL1, float)**2) + w2*(np.asarray(HL2, float)**2)
    P_HH = w1*(np.asarray(HH1, float)**2) + w2*(np.asarray(HH2, float)**2)

    N_LH = p99_median_normalize_map(P_LH, mask=mask)
    N_HL = p99_median_normalize_map(P_HL, mask=mask)
    N_HH = p99_median_normalize_map(P_HH, mask=mask)

    def ch_avg(M):
        M = np.asarray(M)
        return M if M.ndim == 2 else np.nanmean(M, axis=2)
    A_LH = ch_avg(N_LH)
    A_HL = ch_avg(N_HL)
    A_HH = ch_avg(N_HH)

    pairs, h, R_abs = build_radius_graph(xy, R_rel=R_rel, deg_min=deg_min, deg_max=deg_max, Rmax_rel=Rmax_rel)

    E = pairs.shape[0]
    K    = np.zeros(E, dtype=float)
    aout = np.ones(E, dtype=float)
    rout = np.zeros(E, dtype=float)
    theta = np.zeros(E, dtype=float)
    r_edge = np.zeros(E, dtype=float)
    s_edge = np.zeros(E, dtype=float)

    for eidx in range(E):
        i, j = int(pairs[eidx,0]), int(pairs[eidx,1])
        ci = xy[i]; cj = xy[j]
        v  = cj - ci
        un = float(np.linalg.norm(v))
        u  = np.array([1.0, 0.0]) if un < 1e-9 else (v / un)

        d_ij, r_pool_ij, sigma_ij = edge_pool_params(
            ci, cj, h, policy=r_pool_policy, r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
            k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma
        )
        r_edge[eidx] = r_pool_ij
        s_edge[eidx] = sigma_ij

        if pool_mode == "half":
            ys, xs, w = halfdisk_pixel_weights(ci, u, r_pool_ij, sigma_ij, H, W, mask)
        else:
            sigma_perp = max(tube_perp_rel * h, 1e-9)
            sigma_para = max(tube_ax_rel   * d_ij, 1e-9)
            ys, xs, w = corridor_pixel_weights(ci, cj, H, W, tissue_mask=mask,
                                               sigma_perp=sigma_perp, sigma_para=sigma_para,
                                               lateral_mult=tube_lateral_mult)
            if d_ij <= h and w.size:
                r_cap     = receiver_cap_rel * h
                sigma_cap = r_cap
                ys2, xs2, w2 = receiver_cap_weights(cj, u, r_cap, sigma_cap,
                                                    receiver_cap_half_angle, H, W, tissue_mask=mask)
                ys, xs, w = combine_weights(ys, xs, w, ys2, xs2, w2, W)

        if w.size == 0:
            v = cj - ci
            q = (v[0]**2 + v[1]**2)
            K[eidx] = np.exp(-0.5 * q / (h*h))
            aout[eidx] = 1.0; rout[eidx] = 0.0; theta[eidx] = 0.0
            continue

        Eh = np.nansum(w * A_LH[ys, xs])
        Ev = np.nansum(w * A_HL[ys, xs])
        Ed = np.nansum(w * A_HH[ys, xs])
        Sx = Ev + eta*Ed
        Sy = Eh + eta*Ed

        r_axis = abs(Sx - Sy) / (Sx + Sy + 1e-12)
        a = 1.0
        lamx, lamy = 1.0, 1.0
        if (Sx + Sy) > 0:
            a = 1.0 + (a_max - 1.0) * (r_axis ** gamma)
            a = 1.0 if a < a_min else min(a, a_max)
            if a > 1.0:
                if Sx >= Sy:
                    lamx, lamy = 1.0, 1.0 / a; theta[eidx] = 0.0
                else:
                    lamx, lamy = 1.0 / a, 1.0; theta[eidx] = np.pi/2

        lamx, lamy = apply_sigma_floor(lamx, lamy, h, sigma_floor_rel=sigma_floor_rel)
        v = cj - ci
        q = (v[0]*v[0])/(lamx + 1e-12) + (v[1]*v[1])/(lamy + 1e-12)
        K[eidx] = np.exp(-0.5 * q / (h*h))
        aout[eidx] = a
        rout[eidx] = r_axis

    np.savez(out_path,
        edges=pairs, kernel=K, a=aout, r=rout, theta=theta,
        h=h, R_rel=R_rel, R_abs=R_abs,
        r_pool_edge=r_edge, sigma_edge=s_edge,
        pool_mode=pool_mode, r_pool_policy=r_pool_policy,
        r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
        k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma,
        tube_perp_rel=tube_perp_rel, tube_ax_rel=tube_ax_rel, tube_lateral_mult=tube_lateral_mult,
        receiver_cap_rel=receiver_cap_rel, receiver_cap_half_angle=receiver_cap_half_angle,
        deg_min=deg_min, deg_max=deg_max,
        a_min=a_min, a_max=a_max, gamma=gamma,
        eta=eta, sigma_floor_rel=sigma_floor_rel,
        family="wv2"
    )

def run_wv4_pixel(coords, mask, band_glob_level1, meta_npz,
                  out_path, band_glob_level2=None, level_weights=(0.8,0.2),
                  R_rel=2.2, deg_min=10, deg_max=18, Rmax_rel=None,
                  r_pool_rel=2.2, sigma_rel=2.2,
                  tau_kappa=0.2, a_min=1.2, a_max=3.0, gamma=1.5,
                  sigma_floor_rel=0.68,
                  pool_mode="half", r_pool_policy="fixed",
                  k_edge=1.0, rmin_rel=0.8, rmax_rel=1.2, c_sigma=0.73,
                  tube_perp_rel=0.9, tube_ax_rel=1.0, tube_lateral_mult=3.0,
                  receiver_cap_rel=1.2, receiver_cap_half_angle=20.0):
    xy = to_xy(coords)
    files_L1 = sorted(glob.glob(band_glob_level1))
    if len(files_L1) == 0:
        raise FileNotFoundError(f"No bands match {band_glob_level1}")
    arr0 = np.load(files_L1[0])
    if arr0.ndim not in (2,3) or (arr0.ndim==3 and arr0.shape[2] not in (1,3)):
        raise ValueError("4-band arrays must be (H,W) or (H,W,3)")
    H, W = arr0.shape[:2]
    if mask is not None:
        if mask.shape != (H,W):
            raise ValueError("mask shape mismatch")
        mask = mask.astype(bool)

    meta = np.load(meta_npz)
    theta_b = np.asarray(meta["theta"]).astype(float)
    kappa_b = np.asarray(meta["kappa"]).astype(float)
    B = len(files_L1)
    if theta_b.shape[0] != B or kappa_b.shape[0] != B:
        raise ValueError("meta theta/kappa length must match number of bands")

    bands_L1 = [np.load(fp).astype(float) for fp in files_L1]
    norm_L1  = [p99_median_normalize_map(b, mask=mask) for b in bands_L1]
    avg_L1   = [np.nanmean(b, axis=2) if b.ndim==3 else b for b in norm_L1]

    use_L2 = (band_glob_level2 is not None) and (len(glob.glob(band_glob_level2)) == B)
    if use_L2:
        files_L2 = sorted(glob.glob(band_glob_level2))
        bands_L2 = [np.load(fp).astype(float) for fp in files_L2]
        norm_L2  = [p99_median_normalize_map(b, mask=mask) for b in bands_L2]
        avg_L2   = [np.nanmean(b, axis=2) if b.ndim==3 else b for b in norm_L2]
    else:
        avg_L2 = [np.zeros((H,W), dtype=float) for _ in range(B)]

    g1, g2 = level_weights if use_L2 else (1.0, 0.0)

    pairs, h, R_abs = build_radius_graph(xy, R_rel=R_rel, deg_min=deg_min, deg_max=deg_max, Rmax_rel=Rmax_rel)

    E = pairs.shape[0]
    K     = np.zeros(E, dtype=float)
    aout  = np.ones(E, dtype=float)
    rout  = np.zeros(E, dtype=float)
    theta = np.zeros(E, dtype=float)
    r_edge = np.zeros(E, dtype=float)
    s_edge = np.zeros(E, dtype=float)

    cos2 = np.cos(2*theta_b)
    sin2 = np.sin(2*theta_b)

    for eidx in range(E):
        i, j = int(pairs[eidx,0]), int(pairs[eidx,1])
        ci = xy[i]; cj = xy[j]
        v  = cj - ci
        un = float(np.linalg.norm(v))
        u  = np.array([1.0,0.0]) if un < 1e-9 else (v / un)

        d_ij, r_pool_ij, sigma_ij = edge_pool_params(
            ci, cj, h, policy=r_pool_policy, r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
            k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma
        )
        r_edge[eidx] = r_pool_ij
        s_edge[eidx] = sigma_ij

        if pool_mode == "half":
            ys, xs, w = halfdisk_pixel_weights(ci, u, r_pool_ij, sigma_ij, H, W, mask)
        else:
            sigma_perp = max(tube_perp_rel * h, 1e-9)
            sigma_para = max(tube_ax_rel   * d_ij, 1e-9)
            ys, xs, w = corridor_pixel_weights(ci, cj, H, W, tissue_mask=mask,
                                               sigma_perp=sigma_perp, sigma_para=sigma_para,
                                               lateral_mult=tube_lateral_mult)
            if d_ij <= h and w.size:
                r_cap     = receiver_cap_rel * h
                sigma_cap = r_cap
                ys2, xs2, w2 = receiver_cap_weights(cj, u, r_cap, sigma_cap,
                                                    receiver_cap_half_angle, H, W, tissue_mask=mask)
                ys, xs, w = combine_weights(ys, xs, w, ys2, xs2, w2, W)

        if w.size == 0:
            v = cj - ci
            q = (v[0]**2 + v[1]**2)
            K[eidx] = np.exp(-0.5 * q / (h*h))
            aout[eidx] = 1.0; rout[eidx] = 0.0; theta[eidx] = 0.0
            continue

        Csum = 0.0; Ssum = 0.0; Esum = 0.0
        for b in range(B):
            if kappa_b[b] < tau_kappa:
                continue
            pb = g1 * np.nansum(w * avg_L1[b][ys, xs]) + g2 * np.nansum(w * avg_L2[b][ys, xs])
            if not np.isfinite(pb) or pb <= 0:
                continue
            kb = float(kappa_b[b])
            Csum += kb * pb * cos2[b]
            Ssum += kb * pb * sin2[b]
            Esum += kb * pb

        if Esum <= 0:
            a = 1.0; r = 0.0; th = 0.0
        else:
            th = 0.5 * np.arctan2(Ssum, Csum)
            mabs = np.sqrt(Csum*Csum + Ssum*Ssum)
            r = float(mabs / (Esum + 1e-12))
            r = min(r, 0.995)
            a = 1.0 + (a_max - 1.0) * (r ** gamma)
            if a < a_min: a = 1.0
            if a > a_max: a = a_max

        e1 = np.array([np.cos(th), np.sin(th)])
        e2 = np.array([-np.sin(th), np.cos(th)])
        Dpar, Dperp = 1.0, (1.0 / a)
        Dpar, Dperp = apply_sigma_floor(Dpar, Dperp, h, sigma_floor_rel=sigma_floor_rel)
        v = cj - ci
        vpar  = float(v @ e1)
        vperp = float(v @ e2)
        q = (vpar*vpar)/(Dpar + 1e-12) + (vperp*vperp)/(Dperp + 1e-12)
        K[eidx] = np.exp(-0.5 * q / (h*h))
        aout[eidx] = a; rout[eidx] = r; theta[eidx] = th

    np.savez(out_path,
        edges=pairs, kernel=K, a=aout, r=rout, theta=theta,
        h=h, R_rel=R_rel, R_abs=R_abs,
        r_pool_edge=r_edge, sigma_edge=s_edge,
        pool_mode=pool_mode, r_pool_policy=r_pool_policy,
        r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
        k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma,
        tube_perp_rel=tube_perp_rel, tube_ax_rel=tube_ax_rel, tube_lateral_mult=tube_lateral_mult,
        receiver_cap_rel=receiver_cap_rel, receiver_cap_half_angle=receiver_cap_half_angle,
        deg_min=deg_min, deg_max=deg_max,
        a_min=a_min, a_max=a_max, gamma=gamma,
        tau_kappa=tau_kappa, level_weights=(g1,g2),
        sigma_floor_rel=sigma_floor_rel,
        family="wv4"
    )

def load_dtcwt_folder(folder, levels=(1,2), angles=(15,45,75,105,135,165)):
    out = {}
    for L in levels:
        out[L] = {}
        files = glob.glob(os.path.join(folder, f"*L{L}_ang*_upsampled.npy"))
        if not files:
            raise FileNotFoundError(f"No files for level {L} in {folder}")
        by_ang = {}
        for fp in files:
            a = parse_angle_from_name(fp)
            by_ang[a] = fp
        for ang in angles:
            if ang not in by_ang:
                raise FileNotFoundError(f"Missing angle {ang} at L{L}")
            arr = np.load(by_ang[ang], allow_pickle=False)
            if not np.iscomplexobj(arr) or arr.ndim not in (2,3) or (arr.ndim==3 and arr.shape[2] not in (1,3)):
                raise ValueError(f"Bad DTCWT array: {by_ang[ang]}")
            out[L][ang] = arr
    H, W = next(iter(next(iter(out.values())).values())).shape[:2]
    for L in levels:
        for ang in angles:
            if out[L][ang].shape[:2] != (H,W):
                raise ValueError("size mismatch across subbands")
    return out, H, W

def run_dtcwt_pixel(coords, mask, folder, out_path,
                    levels=(1,2), level_weights=(0.7,0.3),
                    angles=(15,45,75,105,135,165),
                    R_rel=2.2, deg_min=10, deg_max=18, Rmax_rel=None,
                    r_pool_rel=2.2, sigma_rel=2.2,
                    beta=2.0, a_min=1.2, a_max=3.0, gamma=1.5,
                    sigma_floor_rel=0.68,
                    pool_mode="half", r_pool_policy="fixed",
                    k_edge=1.0, rmin_rel=0.8, rmax_rel=1.2, c_sigma=0.73,
                    tube_perp_rel=0.9, tube_ax_rel=1.0, tube_lateral_mult=3.0,
                    receiver_cap_rel=1.2, receiver_cap_half_angle=20.0):
    xy = to_xy(coords)
    sub, H, W = load_dtcwt_folder(folder, levels=levels, angles=angles)
    if mask is not None:
        if mask.shape != (H,W):
            raise ValueError("mask shape mismatch")
        mask = mask.astype(bool)

    normP = {}
    meanW = {}
    for L in levels:
        for ang in angles:
            Wc = sub[L][ang]
            if Wc.ndim == 3:
                P  = np.abs(Wc)**2
                Np = p99_median_normalize_map(P, mask=mask)
                normP[(L,ang)] = np.nanmean(Np, axis=2)
                meanW[(L,ang)] = np.nanmean(Wc, axis=2)
            else:
                P  = np.abs(Wc)**2
                Np = p99_median_normalize_map(P, mask=mask)
                normP[(L,ang)] = Np
                meanW[(L,ang)] = Wc

    ang_rad = {ang: np.deg2rad(ang % 180) for ang in angles}
    cos2 = {ang: np.cos(2*ang_rad[ang]) for ang in angles}
    sin2 = {ang: np.sin(2*ang_rad[ang]) for ang in angles}
    lvl_w = {L: level_weights[i] for i, L in enumerate(levels)}

    pairs, h, R_abs = build_radius_graph(xy, R_rel=R_rel, deg_min=deg_min, deg_max=deg_max, Rmax_rel=Rmax_rel)

    E = pairs.shape[0]
    K     = np.zeros(E, dtype=float)
    aout  = np.ones(E, dtype=float)
    rout  = np.zeros(E, dtype=float)
    theta = np.zeros(E, dtype=float)
    r_edge = np.zeros(E, dtype=float)
    s_edge = np.zeros(E, dtype=float)

    for eidx in range(E):
        i, j = int(pairs[eidx,0]), int(pairs[eidx,1])
        ci = xy[i]; cj = xy[j]
        v  = cj - ci
        un = float(np.linalg.norm(v))
        u  = np.array([1.0,0.0]) if un < 1e-9 else (v / un)

        d_ij, r_pool_ij, sigma_ij = edge_pool_params(
            ci, cj, h, policy=r_pool_policy, r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
            k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma
        )
        r_edge[eidx] = r_pool_ij
        s_edge[eidx] = sigma_ij

        if pool_mode == "half":
            ys, xs, w = halfdisk_pixel_weights(ci, u, r_pool_ij, sigma_ij, H, W, mask)
        else:
            sigma_perp = max(tube_perp_rel * h, 1e-9)
            sigma_para = max(tube_ax_rel   * d_ij, 1e-9)
            ys, xs, w = corridor_pixel_weights(ci, cj, H, W, tissue_mask=mask,
                                               sigma_perp=sigma_perp, sigma_para=sigma_para,
                                               lateral_mult=tube_lateral_mult)
            if d_ij <= h and w.size:
                r_cap     = receiver_cap_rel * h
                sigma_cap = r_cap
                ys2, xs2, w2 = receiver_cap_weights(cj, u, r_cap, sigma_cap,
                                                    receiver_cap_half_angle, H, W, tissue_mask=mask)
                ys, xs, w = combine_weights(ys, xs, w, ys2, xs2, w2, W)

        if w.size == 0:
            v = cj - ci
            q = (v[0]**2 + v[1]**2)
            K[eidx] = np.exp(-0.5 * q / (h*h))
            aout[eidx] = 1.0; rout[eidx] = 0.0; theta[eidx] = 0.0
            continue

        Csum = 0.0; Ssum = 0.0; Esum = 0.0
        for L in levels:
            for ang in angles:
                pb = np.nansum(w * normP[(L,ang)][ys, xs])
                if not np.isfinite(pb) or pb <= 0:
                    continue
                Wmap = meanW[(L,ang)][ys, xs]
                num = np.abs(np.nansum(w * Wmap))
                den = np.nansum(w * np.abs(Wmap)) + 1e-12
                rho = float(num / den) if den > 0 else 0.0
                om  = lvl_w[L] * (rho ** beta)
                Csum += om * pb * cos2[ang]
                Ssum += om * pb * sin2[ang]
                Esum += om * pb

        if Esum <= 0:
            a = 1.0; r = 0.0; th = 0.0
        else:
            th = 0.5 * np.arctan2(Ssum, Csum)
            mabs = np.sqrt(Csum*Csum + Ssum*Ssum)
            r = float(mabs / (Esum + 1e-12))
            r = min(r, 0.995)
            a = 1.0 + (a_max - 1.0) * (r ** gamma)
            if a < a_min: a = 1.0
            if a > a_max: a = a_max

        e1 = np.array([np.cos(th), np.sin(th)])
        e2 = np.array([-np.sin(th), np.cos(th)])
        Dpar, Dperp = 1.0, (1.0 / a)
        Dpar, Dperp = apply_sigma_floor(Dpar, Dperp, h, sigma_floor_rel=sigma_floor_rel)
        v = cj - ci
        vpar  = float(v @ e1)
        vperp = float(v @ e2)
        q = (vpar*vpar)/(Dpar + 1e-12) + (vperp*vperp)/(Dperp + 1e-12)
        K[eidx] = np.exp(-0.5 * q / (h*h))
        aout[eidx] = a; rout[eidx] = r; theta[eidx] = th

    np.savez(out_path,
        edges=pairs, kernel=K, a=aout, r=rout, theta=theta,
        h=h, R_rel=R_rel, R_abs=R_abs,
        r_pool_edge=r_edge, sigma_edge=s_edge,
        pool_mode=pool_mode, r_pool_policy=r_pool_policy,
        r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
        k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma,
        tube_perp_rel=tube_perp_rel, tube_ax_rel=tube_ax_rel, tube_lateral_mult=tube_lateral_mult,
        receiver_cap_rel=receiver_cap_rel, receiver_cap_half_angle=receiver_cap_half_angle,
        deg_min=deg_min, deg_max=deg_max,
        a_min=a_min, a_max=a_max, gamma=gamma,
        beta=beta, level_weights=tuple(level_weights),
        sigma_floor_rel=sigma_floor_rel,
        family="dtcwt"
    )

def run_wv4_pixel_from_bundle(coords, mask, bands_l1, theta_b, kappa_b, out_path,
                              bands_l2=None, level_weights=(0.8,0.2),
                              R_rel=2.2, deg_min=10, deg_max=18, Rmax_rel=None,
                              r_pool_rel=2.2, sigma_rel=2.2,
                              tau_kappa=0.2, a_min=1.2, a_max=3.0, gamma=1.5,
                              sigma_floor_rel=0.68,
                              pool_mode="half", r_pool_policy="fixed",
                              k_edge=1.0, rmin_rel=0.8, rmax_rel=1.2, c_sigma=0.73,
                              tube_perp_rel=0.9, tube_ax_rel=1.0, tube_lateral_mult=3.0,
                              receiver_cap_rel=1.2, receiver_cap_half_angle=20.0):
    xy = to_xy(coords)
    bands_l1 = np.asarray(bands_l1)
    if bands_l1.ndim not in (3,4):
        raise ValueError("bands_l1 must be (B,H,W) or (B,H,W,3)")
    B, H, W = bands_l1.shape[:3]
    if mask is not None and mask.shape != (H,W):
        raise ValueError("mask shape mismatch")

    def normalize_and_avg(stack):
        out = np.empty((stack.shape[0], H, W), dtype=float)
        for b in range(stack.shape[0]):
            arr = stack[b]
            norm = p99_median_normalize_map(arr, mask=mask)
            out[b] = np.nanmean(norm, axis=2) if norm.ndim==3 else norm
        return out

    avg_L1 = normalize_and_avg(bands_l1)
    use_L2 = bands_l2 is not None
    if use_L2:
        bands_l2 = np.asarray(bands_l2)
        if bands_l2.shape[:3] != (B,H,W):
            raise ValueError("bands_l2 shape must match bands_l1's (B,H,W[,3])")
        avg_L2 = normalize_and_avg(bands_l2)
        g1, g2 = level_weights
    else:
        avg_L2 = np.zeros_like(avg_L1)
        g1, g2 = 1.0, 0.0

    theta_b = np.asarray(theta_b, float)
    kappa_b = np.asarray(kappa_b, float)
    if theta_b.shape[0] != B or kappa_b.shape[0] != B:
        raise ValueError("theta/kappa length must match number of bands (L1)")

    pairs, h, R_abs = build_radius_graph(xy, R_rel=R_rel, deg_min=deg_min, deg_max=deg_max, Rmax_rel=Rmax_rel)

    E = pairs.shape[0]
    K     = np.zeros(E, dtype=float)
    aout  = np.ones(E, dtype=float)
    rout  = np.zeros(E, dtype=float)
    theta = np.zeros(E, dtype=float)
    r_edge = np.zeros(E, dtype=float)
    s_edge = np.zeros(E, dtype=float)

    cos2 = np.cos(2*theta_b)
    sin2 = np.sin(2*theta_b)

    for eidx in range(E):
        i, j = int(pairs[eidx,0]), int(pairs[eidx,1])
        ci = xy[i]; cj = xy[j]
        v  = cj - ci
        un = float(np.linalg.norm(v))
        u  = np.array([1.0,0.0]) if un < 1e-9 else (v / un)

        d_ij, r_pool_ij, sigma_ij = edge_pool_params(
            ci, cj, h, policy=r_pool_policy, r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
            k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma
        )
        r_edge[eidx] = r_pool_ij
        s_edge[eidx] = sigma_ij

        if pool_mode == "half":
            ys, xs, w = halfdisk_pixel_weights(ci, u, r_pool_ij, sigma_ij, H, W, mask)
        else:
            sigma_perp = max(tube_perp_rel * h, 1e-9)
            sigma_para = max(tube_ax_rel   * d_ij, 1e-9)
            ys, xs, w = corridor_pixel_weights(ci, cj, H, W, tissue_mask=mask,
                                               sigma_perp=sigma_perp, sigma_para=sigma_para,
                                               lateral_mult=tube_lateral_mult)
            if d_ij <= h and w.size:
                r_cap     = receiver_cap_rel * h
                sigma_cap = r_cap
                ys2, xs2, w2 = receiver_cap_weights(cj, u, r_cap, sigma_cap,
                                                    receiver_cap_half_angle, H, W, tissue_mask=mask)
                ys, xs, w = combine_weights(ys, xs, w, ys2, xs2, w2, W)

        if w.size == 0:
            v = cj - ci
            q = (v[0]**2 + v[1]**2)
            K[eidx] = np.exp(-0.5 * q / (h*h))
            aout[eidx] = 1.0; rout[eidx] = 0.0; theta[eidx] = 0.0
            continue

        Csum = 0.0; Ssum = 0.0; Esum = 0.0
        for b in range(B):
            if kappa_b[b] < tau_kappa:
                continue
            pb = g1 * np.nansum(w * avg_L1[b, ys, xs]) + g2 * np.nansum(w * avg_L2[b, ys, xs])
            if not np.isfinite(pb) or pb <= 0:
                continue
            kb = float(kappa_b[b])
            Csum += kb * pb * cos2[b]
            Ssum += kb * pb * sin2[b]
            Esum += kb * pb

        if Esum <= 0:
            a = 1.0; r = 0.0; th = 0.0
        else:
            th = 0.5 * np.arctan2(Ssum, Csum)
            mabs = np.sqrt(Csum*Csum + Ssum*Ssum)
            r = float(mabs / (Esum + 1e-12))
            r = min(r, 0.995)
            a = 1.0 + (a_max - 1.0) * (r ** gamma)
            if a < a_min: a = 1.0
            if a > a_max: a = a_max

        e1 = np.array([np.cos(th), np.sin(th)])
        e2 = np.array([-np.sin(th), np.cos(th)])
        Dpar, Dperp = 1.0, (1.0 / a)
        Dpar, Dperp = apply_sigma_floor(Dpar, Dperp, h, sigma_floor_rel=sigma_floor_rel)
        v = cj - ci
        vpar  = float(v @ e1)
        vperp = float(v @ e2)
        q = (vpar*vpar)/(Dpar + 1e-12) + (vperp*vperp)/(Dperp + 1e-12)
        K[eidx] = np.exp(-0.5 * q / (h*h))
        aout[eidx] = a; rout[eidx] = r; theta[eidx] = th

    np.savez(out_path,
        edges=pairs, kernel=K, a=aout, r=rout, theta=theta,
        h=h, R_rel=R_rel, R_abs=R_abs,
        r_pool_edge=r_edge, sigma_edge=s_edge,
        pool_mode=pool_mode, r_pool_policy=r_pool_policy,
        r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
        k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma,
        tube_perp_rel=tube_perp_rel, tube_ax_rel=tube_ax_rel, tube_lateral_mult=tube_lateral_mult,
        receiver_cap_rel=receiver_cap_rel, receiver_cap_half_angle=receiver_cap_half_angle,
        deg_min=deg_min, deg_max=deg_max,
        a_min=a_min, a_max=a_max, gamma=gamma,
        tau_kappa=tau_kappa, level_weights=(g1,g2),
        sigma_floor_rel=sigma_floor_rel,
        family="wv4"
    )

def run_dtcwt_pixel_from_bundle(coords, mask, levels, angles, level_maps, out_path,
                                level_weights=None,
                                R_rel=2.2, deg_min=10, deg_max=18, Rmax_rel=None,
                                r_pool_rel=2.2, sigma_rel=2.2,
                                beta=2.0, a_min=1.2, a_max=3.0, gamma=1.5,
                                sigma_floor_rel=0.68,
                                pool_mode="half", r_pool_policy="fixed",
                                k_edge=1.0, rmin_rel=0.8, rmax_rel=1.2, c_sigma=0.73,
                                tube_perp_rel=0.9, tube_ax_rel=1.0, tube_lateral_mult=3.0,
                                receiver_cap_rel=1.2, receiver_cap_half_angle=20.0):
    xy = to_xy(coords)
    levels = [int(L) for L in np.asarray(levels)]
    angles = [int(a) for a in np.asarray(angles)]
    B = len(angles)

    normP = {}
    meanW = {}
    H = W = None
    for L in levels:
        A = np.asarray(level_maps[L])
        if A.ndim not in (3,4):
            raise ValueError(f"dtcwt bundle level {L}: expected (B,H,W) or (B,H,W,3)")
        if H is None: H, W = A.shape[1:3]
        if A.shape[1:3] != (H,W):
            raise ValueError("size mismatch across dtcwt levels in bundle")
        for b, ang in enumerate(angles):
            Wc = A[b]
            if not np.iscomplexobj(Wc):
                Wc = Wc.astype(np.complex64) + 0j
            if Wc.ndim == 3 and Wc.shape[2] in (1,3):
                P  = np.abs(Wc)**2
                Np = p99_median_normalize_map(P, mask=None if mask is None else mask)
                normP[(L,ang)] = np.nanmean(Np, axis=2)
                meanW[(L,ang)] = np.nanmean(Wc, axis=2)
            elif Wc.ndim == 2:
                P  = np.abs(Wc)**2
                Np = p99_median_normalize_map(P, mask=None if mask is None else mask)
                normP[(L,ang)] = Np
                meanW[(L,ang)] = Wc
            else:
                raise ValueError(f"Unexpected dtcwt entry shape: {Wc.shape}")

    if level_weights is None:
        level_weights = [0.7, 0.3][:len(levels)] if len(levels) >= 2 else [1.0]
    if len(level_weights) != len(levels):
        raise ValueError("level_weights length must match number of levels")
    lvl_w = {L: float(level_weights[i]) for i, L in enumerate(levels)}

    ang_rad = {ang: np.deg2rad(ang % 180) for ang in angles}
    cos2 = {ang: np.cos(2*ang_rad[ang]) for ang in angles}
    sin2 = {ang: np.sin(2*ang_rad[ang]) for ang in angles}

    pairs, h, R_abs = build_radius_graph(xy, R_rel=R_rel, deg_min=deg_min, deg_max=deg_max, Rmax_rel=Rmax_rel)

    E = pairs.shape[0]
    K     = np.zeros(E, dtype=float)
    aout  = np.ones(E, dtype=float)
    rout  = np.zeros(E, dtype=float)
    theta = np.zeros(E, dtype=float)
    r_edge = np.zeros(E, dtype=float)
    s_edge = np.zeros(E, dtype=float)

    for eidx in range(E):
        i, j = int(pairs[eidx,0]), int(pairs[eidx,1])
        ci = xy[i]; cj = xy[j]
        v  = cj - ci
        un = float(np.linalg.norm(v))
        u  = np.array([1.0,0.0]) if un < 1e-9 else (v / un)

        d_ij, r_pool_ij, sigma_ij = edge_pool_params(
            ci, cj, h, policy=r_pool_policy, r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
            k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma
        )
        r_edge[eidx] = r_pool_ij
        s_edge[eidx] = sigma_ij

        if pool_mode == "half":
            ys, xs, w = halfdisk_pixel_weights(ci, u, r_pool_ij, sigma_ij, H, W, mask)
        else:
            sigma_perp = max(tube_perp_rel * h, 1e-9)
            sigma_para = max(tube_ax_rel   * d_ij, 1e-9)
            ys, xs, w = corridor_pixel_weights(ci, cj, H, W, tissue_mask=mask,
                                               sigma_perp=sigma_perp, sigma_para=sigma_para,
                                               lateral_mult=tube_lateral_mult)
            if d_ij <= h and w.size:
                r_cap     = receiver_cap_rel * h
                sigma_cap = r_cap
                ys2, xs2, w2 = receiver_cap_weights(cj, u, r_cap, sigma_cap,
                                                    receiver_cap_half_angle, H, W, tissue_mask=mask)
                ys, xs, w = combine_weights(ys, xs, w, ys2, xs2, w2, W)

        if w.size == 0:
            v = cj - ci
            q = (v[0]**2 + v[1]**2)
            K[eidx] = np.exp(-0.5 * q / (h*h))
            aout[eidx] = 1.0; rout[eidx] = 0.0; theta[eidx] = 0.0
            continue

        Csum = 0.0; Ssum = 0.0; Esum = 0.0
        for L in levels:
            for ang in angles:
                pb = np.nansum(w * normP[(L,ang)][ys, xs])
                if not np.isfinite(pb) or pb <= 0:
                    continue
                Wmap = meanW[(L,ang)][ys, xs]
                num = np.abs(np.nansum(w * Wmap))
                den = np.nansum(w * np.abs(Wmap)) + 1e-12
                rho = float(num / den) if den > 0 else 0.0
                om  = lvl_w[L] * (rho ** beta)
                Csum += om * pb * cos2[ang]
                Ssum += om * pb * sin2[ang]
                Esum += om * pb

        if Esum <= 0:
            a = 1.0; r = 0.0; th = 0.0
        else:
            th = 0.5 * np.arctan2(Ssum, Csum)
            mabs = np.sqrt(Csum*Csum + Ssum*Ssum)
            r = float(mabs / (Esum + 1e-12))
            r = min(r, 0.995)
            a = 1.0 + (a_max - 1.0) * (r ** gamma)
            if a < a_min: a = 1.0
            if a > a_max: a = a_max

        e1 = np.array([np.cos(th), np.sin(th)])
        e2 = np.array([-np.sin(th), np.cos(th)])
        Dpar, Dperp = 1.0, (1.0 / a)
        Dpar, Dperp = apply_sigma_floor(Dpar, Dperp, h, sigma_floor_rel=sigma_floor_rel)
        v = cj - ci
        vpar  = float(v @ e1)
        vperp = float(v @ e2)
        q = (vpar*vpar)/(Dpar + 1e-12) + (vperp*vperp)/(Dperp + 1e-12)
        K[eidx] = np.exp(-0.5 * q / (h*h))
        aout[eidx] = a; rout[eidx] = r; theta[eidx] = th

    np.savez(out_path,
        edges=pairs, kernel=K, a=aout, r=rout, theta=theta,
        h=h, R_rel=R_rel, R_abs=R_abs,
        r_pool_edge=r_edge, sigma_edge=s_edge,
        pool_mode=pool_mode, r_pool_policy=r_pool_policy,
        r_pool_rel=r_pool_rel, sigma_rel=sigma_rel,
        k_edge=k_edge, rmin_rel=rmin_rel, rmax_rel=rmax_rel, c_sigma=c_sigma,
        tube_perp_rel=tube_perp_rel, tube_ax_rel=tube_ax_rel, tube_lateral_mult=tube_lateral_mult,
        receiver_cap_rel=receiver_cap_rel, receiver_cap_half_angle=receiver_cap_half_angle,
        deg_min=deg_min, deg_max=deg_max,
        a_min=a_min, a_max=a_max, gamma=gamma,
        beta=beta, level_weights=tuple(level_weights),
        sigma_floor_rel=sigma_floor_rel,
        family="dtcwt"
    )
