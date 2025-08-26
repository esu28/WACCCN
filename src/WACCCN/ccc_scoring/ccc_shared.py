import numpy as np

def median_1nn(coords):
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    return float(np.median(d[:, 1]))

def knn_mean_distance(coords, k=1):
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=k + 1)
    return float(d[:, 1:].mean(axis=1).mean())

def edge_distances(coords, edges):
    vi = coords[edges[:, 0]]
    vj = coords[edges[:, 1]]
    return np.linalg.norm(vi - vj, axis=1)

def scatter_to_dense(N, edges, values):
    A = np.zeros((N, N), dtype=np.float32)
    np.add.at(A, (edges[:, 0], edges[:, 1]), values.astype(np.float32))
    return A

def per_source_normalize_K(K, edges, N, eps=1e-12):
    out = np.asarray(K, float).copy()
    src = edges[:, 0]
    sums = np.bincount(src, weights=out, minlength=N)
    cnts = np.bincount(src, minlength=N).astype(float)
    denom = sums / np.maximum(cnts, 1.0)
    out /= np.maximum(denom[src], eps)
    return out

def hist_evidence_edges(W, edges, sigma=1.5):
    W = np.asarray(W)
    if W.ndim == 2 and W.shape[0] == W.shape[1]:
        K = np.where(np.isfinite(W), W, 0.0).astype(float)
        K = 0.5 * (K + K.T)
        N = K.shape[0]
        diag_med = float(np.nanmedian(np.diag(K))) if N > 0 else 0.0
        off = K[~np.eye(N, dtype=bool)]
        off_med = float(np.nanmedian(off)) if off.size else 0.0
        is_distance_like = diag_med <= off_med
        if is_distance_like:
            dmin, dmax = np.nanmin(K), np.nanmax(K)
            if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
                M = np.zeros_like(K, float)
            else:
                M = 1.0 - (K - dmin) / (dmax - dmin)
        else:
            smin, smax = np.nanmin(K), np.nanmax(K)
            if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
                M = np.zeros_like(K, float)
            else:
                M = (K - smin) / (smax - smin)
        return M[edges[:, 0], edges[:, 1]]
    mu = np.nanmean(W, axis=0, keepdims=True)
    sd = np.nanstd(W, axis=0, keepdims=True)
    Y = (W - mu) / (sd + 1e-8)
    Xi = Y[edges[:, 0]]
    Xj = Y[edges[:, 1]]
    d2 = np.einsum("ij,ij->i", Xi - Xj, Xi - Xj)
    M = np.exp(-d2 / (2.0 * sigma * sigma))
    lo, hi = np.nanmin(M), np.nanmax(M)
    return (M - lo) / (max(hi - lo, 1e-8))

def robust_Z_from_similarity(M_edge, q=0.60, zcap=3.0, mad_floor_rel=0.20, target_q=0.90, target_frac=0.95):
    med = np.quantile(M_edge, q)
    q10, q90 = np.quantile(M_edge, [0.10, 0.90])
    span = max(q90 - q10, 1e-12)
    mad = max(np.median(np.abs(M_edge - med)), mad_floor_rel * span, 1e-12)
    Z = (M_edge - med) / mad
    Z = np.clip(Z, -zcap, zcap)
    Zp = np.maximum(Z, 0.0)
    try:
        qv = np.quantile(Zp, target_q, method="higher")
    except TypeError:
        qv = np.quantile(Zp, target_q, interpolation="higher")
    if qv >= target_frac * zcap and qv > 0:
        scale = qv / (target_frac * zcap)
        Z = np.clip(Z / max(scale, 1e-12), -zcap, zcap)
    return Z

def auto_beta_plus(Z, gamma_big=0.10, f_target=0.10, beta_max=0.08, eps=1e-8):
    Zp = np.maximum(Z, 0.0)
    if not np.isfinite(Zp).any() or np.nanmax(Zp) <= eps:
        return 0.0
    try:
        qv = np.quantile(Zp, 1.0 - f_target, method="higher")
    except TypeError:
        qv = np.quantile(Zp, 1.0 - f_target, interpolation="higher")
    if qv <= eps:
        return 0.0
    beta = np.log1p(gamma_big) / max(qv, eps)
    return float(min(beta_max, beta * 1.001))

def infer_mode_from_text(s):
    if not s:
        return None
    t = str(s).lower()
    mb_keys = ["membrane","membranous","juxtacrine","contact","cadherin","integrin","itga","itgb","desmosome","tight junction","gap junction","adhesion","cell-cell","cell–cell","receptor complex"]
    sec_keys = ["secreted","soluble","paracrine","endocrine","chemokine"]
    if any(k in t for k in mb_keys): return "mb"
    if any(k in t for k in sec_keys): return "secreted"
    return None

def load_aniso_npz(path):
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    Z = np.load(path, allow_pickle=True)
    files = set(Z.files)
    if 'edges' in files:
        edges = Z['edges'].astype(int)
    elif 'E' in files:
        edges = Z['E'].astype(int)
    else:
        raise ValueError("NPZ needs 'edges' or 'E'")
    if 'kernel' in files:
        K = Z['kernel'].astype(float).ravel()
    elif 'K' in files:
        K = Z['K'].astype(float).ravel()
    else:
        raise ValueError("NPZ needs 'kernel' or 'K'")
    out = {'edges': edges, 'K': K}
    if 'h' in files:
        try:
            out['h'] = float(np.array(Z['h']).item())
        except Exception:
            out['h'] = float(Z['h'])
    return out

def hill_response(x, K_half):
    x = np.maximum(x, 0.0)
    return x / (K_half + x + 1e-12)
