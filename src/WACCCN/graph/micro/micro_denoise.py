# post-hoc denoising step (optional)
import os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from collections import Counter, defaultdict

def _read_coords_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    cols = [c.lower() for c in df.columns]
    colmap = dict(zip(cols, df.columns))
    xcol = colmap.get("x") or df.columns[1]
    ycol = colmap.get("y") or df.columns[2]
    spotcol = colmap.get("spot") or colmap.get("barcode") or df.columns[3]
    return pd.DataFrame({
        "spot": df[spotcol].astype(str).values,
        "x": pd.to_numeric(df[xcol]).values,
        "y": pd.to_numeric(df[ycol]).values,
    })

def _read_micro(micro_csv):
    mic = pd.read_csv(micro_csv)
    if "spot" not in mic.columns:
        mic = mic.rename(columns={mic.columns[0]: "spot"})
    mic["spot"] = mic["spot"].astype(str)
    if "micro_id" not in mic.columns: raise ValueError("micro_csv missing 'micro_id'")
    if "macro" not in mic.columns:    raise ValueError("micro_csv missing 'macro'")
    if "micro_name" not in mic.columns:
        mic["micro_name"] = mic["micro_id"].astype(str)
    return mic[["spot","macro","micro_id","micro_name"]].copy()

def _radius_cap(coords, mult):
    t = cKDTree(coords)
    d2, _ = t.query(coords, k=2)
    h = float(np.median(d2[:,1]))
    return mult * h, t

def _majority_vote(df, labels0, macro, k, consensus, r_cap):
    labels1 = labels0.copy()
    for m in pd.unique(macro):
        idx = np.where(macro == m)[0]
        if idx.size == 0: continue
        pts = df.iloc[idx][["x","y"]].to_numpy()
        t = cKDTree(pts)
        for ii, gi in enumerate(idx):
            kq = min(k+1, idx.size)
            d, nloc = t.query(pts[ii], k=kq)
            nloc = np.atleast_1d(nloc).tolist()
            if nloc and nloc[0] == ii: nloc = nloc[1:]
            if not nloc: continue
            dvals = np.atleast_1d(d)[1:] if np.ndim(d) else np.array([d], float)
            keep = [j for j,dist in zip(nloc, dvals) if dist <= r_cap] or nloc
            neigh_glob = [idx[j] for j in keep]
            votes = [labels0[g] for g in neigh_glob]
            if not votes: continue
            mode_label, count = Counter(votes).most_common(1)[0]
            if mode_label != labels0[gi] and (count / len(votes)) >= consensus:
                labels1[gi] = mode_label
    return labels1

def _components_same_label(df, idx_list, labels, macro_vals, k, r_cap):
    pts = df.iloc[idx_list][["x","y"]].to_numpy()
    t = cKDTree(pts)
    kq = min(k+1, len(idx_list))
    _, nn = t.query(pts, k=kq)
    adj = defaultdict(set)
    loc2glob = np.array(idx_list, int)
    for i, neighs in enumerate(np.atleast_2d(nn)):
        gi = loc2glob[i]
        for j in neighs[1:]:
            gj = loc2glob[int(j)]
            if macro_vals[gi] != macro_vals[gj]: continue
            if labels[gi] != labels[gj]: continue
            dx = df.loc[gi,"x"] - df.loc[gj,"x"]; dy = df.loc[gi,"y"] - df.loc[gj,"y"]
            if dx*dx + dy*dy <= r_cap*r_cap:
                adj[gi].add(gj); adj[gj].add(gi)
    seen, comps = set(), []
    for gi in idx_list:
        if gi in seen: continue
        stack = [gi]; seen.add(gi); comp = [gi]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v); comp.append(v)
        comps.append(comp)
    return comps

def _tiny_island_cleanup(df, labels1, macro, min_island, r_cap, global_tree, k_for_cc=8):
    labels2 = labels1.copy()
    for m in pd.unique(macro):
        idx_m = np.where(macro == m)[0]
        if idx_m.size == 0: continue
        labs_m = labels2[idx_m]
        for micro_val in np.unique(labs_m):
            idx_mm = idx_m[labs_m == micro_val]
            if idx_mm.size == 0: continue
            comps = _components_same_label(df, idx_mm, labels2, macro, k_for_cc, r_cap)
            for comp in comps:
                if len(comp) >= min_island: continue
                boundary = set()
                for g in comp:
                    neigh = global_tree.query_ball_point(df.loc[g, ["x","y"]].to_numpy(), r=r_cap)
                    for h in neigh:
                        if macro[h] == m and labels2[h] != micro_val:
                            boundary.add(h)
                if boundary:
                    new_lab, _ = Counter(labels2[list(boundary)]).most_common(1)[0]
                    for g in comp: labels2[g] = new_lab
    return labels2

def denoise_micro(labels_csv, micro_csv, out_prefix, k=7, consensus=0.7, min_island=5, r_mult=2.2):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    base = _read_coords_labels(labels_csv)
    mic  = _read_micro(micro_csv)
    df = base.merge(mic, on="spot", how="inner").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No overlap between labels_csv barcodes and micro_csv spots.")
    coords = df[["x","y"]].to_numpy()
    r_cap, global_tree = _radius_cap(coords, r_mult)
    labels0 = df["micro_id"].to_numpy(int)
    macro   = df["macro"].astype(str).to_numpy()
    labels1 = _majority_vote(df, labels0, macro, k, consensus, r_cap)
    labels2 = _tiny_island_cleanup(df, labels1, macro, min_island, r_cap, global_tree)
    out_csv = f"{out_prefix}_micro_assignments_DENOISED.csv"
    out = df[["spot","macro","micro_name"]].copy()
    out["micro_id"] = labels2
    out.to_csv(out_csv, index=False)
    return out_csv
