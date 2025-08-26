import numpy as np, pandas as pd
from scipy.spatial import cKDTree

def load_xy(path):
    if path.endswith(".npy"):
        arr = np.load(path)
        return arr[:, -2:].astype(float)
    try:
        arr = np.loadtxt(path, delimiter=",", usecols=(-2, -1), skiprows=1)
        return arr.reshape(-1, 2).astype(float)
    except Exception:
        df = pd.read_csv(path)
        x_candidates = ["x","x_um","X","coord_x","imagecol","x_pixel","xpix"]
        y_candidates = ["y","y_um","Y","coord_y","imagerow","y_pixel","ypix"]
        xcol = next((c for c in x_candidates if c in df.columns), None)
        ycol = next((c for c in y_candidates if c in df.columns), None)
        if xcol and ycol:
            return df[[xcol, ycol]].to_numpy(dtype=float)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            raise ValueError("Could not find two numeric x,y columns in CSV")
        return num.iloc[:, -2:].to_numpy(dtype=float)

def _stats4(x):
    x = np.asarray(x)
    return float(np.min(x)), float(np.median(x)), float(np.percentile(x,90)), float(np.max(x))

def _weak_cc(n, edges):
    adj = [[] for _ in range(n)]
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    seen = np.zeros(n, dtype=bool)
    comps, largest = 0, 0
    for i in range(n):
        if not seen[i]:
            comps += 1
            s = [i]; seen[i]=True; size=0
            while s:
                u = s.pop(); size += 1
                for w in adj[u]:
                    if not seen[w]:
                        seen[w]=True; s.append(w)
            largest = max(largest, size)
    return comps, largest

def summarize(coords_path, npz_path):
    xy = load_xy(coords_path)
    z = np.load(npz_path)
    edges = z["edges"]; K = z["kernel"]; a = z["a"]; r = z["r"]
    N = xy.shape[0]; E = edges.shape[0]
    out_deg = np.bincount(edges[:,0], minlength=N)
    in_deg  = np.bincount(edges[:,1], minlength=N)

    tree = cKDTree(xy)
    d2, _ = tree.query(xy, k=2)
    h = float(np.median(d2[:,1]))

    v = xy[edges[:,1]] - xy[edges[:,0]]
    elen = np.linalg.norm(v, axis=1)
    elen_over_h = elen / h

    R_abs = z["R_abs"].item() if "R_abs" in z.files else (z["R"].item() if "R" in z.files else None)
    a_cap = z["a_max"].item() if "a_max" in z.files else 3.0

    cc_count, cc_big = _weak_cc(N, edges)

    degR = {}
    for Rmul in [2.0, 2.2]:
        nb = tree.query_ball_point(xy, r=Rmul*h)
        dr = np.array([len([j for j in lst if j!=i]) for i,lst in enumerate(nb)], dtype=int)
        degR[Rmul] = {
            "min": int(np.min(dr)), "med": float(np.median(dr)),
            "p90": float(np.percentile(dr,90)), "max": int(np.max(dr)),
            "frac_lt10": float(np.mean(dr<10)), "frac_gt18": float(np.mean(dr>18))
        }

    report = {
        "graph": {
            "N": int(N), "E": int(E), "k_mean": float(E/max(N,1)),
            "h": h,
            "edge_len_over_h": {"min": _stats4(elen_over_h)[0], "med": _stats4(elen_over_h)[1],
                                "p90": _stats4(elen_over_h)[2], "max": _stats4(elen_over_h)[3]},
            "frac_len_gt_R": float(np.mean(elen > R_abs)) if R_abs is not None else None,
            "frac_len_gt_2h": float(np.mean(elen_over_h > 2.0)),
            "out_deg": {"min": int(np.min(out_deg)), "med": float(np.median(out_deg)),
                        "p90": float(np.percentile(out_deg,90)), "max": int(np.max(out_deg))},
            "in_deg": {"min": int(np.min(in_deg)), "med": float(np.median(in_deg)),
                       "p90": float(np.percentile(in_deg,90)), "max": int(np.max(in_deg))},
            "weak_cc": {"count": int(cc_count), "largest": int(cc_big)},
            "degR": degR
        },
        "kernel": {
            "K": {"min": float(np.min(K)), "med": float(np.median(K)),
                  "p90": float(np.percentile(K,90)), "max": float(np.max(K))},
            "a": {"min": float(np.min(a)), "med": float(np.median(a)),
                  "p90": float(np.percentile(a,90)), "max": float(np.max(a))},
            "r": {"min": float(np.min(r)), "med": float(np.median(r)),
                  "p90": float(np.percentile(r,90)), "max": float(np.max(r))},
            "frac_a_eq_1": float(np.mean(np.isclose(a, 1.0, atol=1e-6))),
            "frac_a_near_cap": float(np.mean(a > (a_cap - 1e-6))),
            "any_nan": {"K": bool(np.isnan(K).any()), "a": bool(np.isnan(a).any()), "r": bool(np.isnan(r).any())},
            "K_within_0_1": bool(float(np.min(K))>=-1e-9 and float(np.max(K))<=1+1e-9)
        },
        "hints": [
            "If many edges have len>2h or degR has big frac_gt18, lower R_rel or deg_max.",
            "If many nodes have deg<10 at R≈2.0–2.2h, increase R_rel or deg_min slightly.",
            "If frac_a_near_cap is high, raise a_max or reduce gamma.",
            "If frac_a_eq_1 ~ 1.0, anisotropy may be muted: try larger sigma_rel or smaller tau_kappa/beta."
        ]
    }
    return report
