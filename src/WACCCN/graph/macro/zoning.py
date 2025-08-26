# zoning — zones (tumor: core/edge; non-tumor: transitory/interior) 
# src/wacccn/segmentation/zoning.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.spatial import cKDTree
from collections import deque

def _kmeans_1d_threshold(v, seed=0):
    v = np.asarray(v, float).reshape(-1, 1)
    v = v[np.isfinite(v).ravel()]
    if v.shape[0] < 2 or np.allclose(v, v[0]): 
        return float(np.median(v)) if v.size else np.nan
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(v)
    c = np.sort(km.cluster_centers_.ravel())
    return float(0.5*(c[0]+c[1])) if c[0]!=c[1] else float(c[0])

def _median_1nn(coords):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(coords)
    d2, _ = nbrs.kneighbors(coords)
    return float(np.median(d2[:,1]))

def _radius_neighbors(coords, R):
    tree = cKDTree(coords)
    return tree.query_ball_point(coords, r=R), tree

def _tumor_islands(idx_tumor, nbr_ids, z):
    m = {u:i for i,u in enumerate(idx_tumor)}
    G = [[] for _ in range(len(idx_tumor))]
    for i, neigh in enumerate(nbr_ids):
        if z[i]!=0: continue
        ii = m[i]
        for j in neigh:
            if j!=i and z[j]==0:
                G[ii].append(m[j])
    seen = np.zeros(len(idx_tumor), bool)
    islands = []
    for s in range(len(idx_tumor)):
        if seen[s]: continue
        q = deque([s]); seen[s]=True; comp=[s]
        while q:
            u = q.popleft()
            for v in G[u]:
                if not seen[v]:
                    seen[v]=True; q.append(v); comp.append(v)
        islands.append(np.asarray(comp, int))
    return islands

def compute_zoning_from_spot_csv(in_csv, alpha=2.2, k_tumor=6, min_island=20, random_state=0,
                                 core_top_q=0.20, core_min_frac=0.08):
    df = pd.read_csv(in_csv)
    idx_str = df.iloc[:,0].astype(str).to_numpy()
    x = df.iloc[:,1].astype(float).to_numpy()
    y = df.iloc[:,2].astype(float).to_numpy()
    barcode = df.iloc[:,3].astype(str).to_numpy()
    label12 = df.iloc[:,4].astype(int).to_numpy()   # 2=tumor, 1=non

    coords = np.c_[x,y].astype(float)
    N = coords.shape[0]
    z = np.where(label12==2, 0, 1).astype(int)

    h = _median_1nn(coords)
    R = alpha * h
    nbr_ids, tree_all = _radius_neighbors(coords, R)

    mix = np.zeros(N, float)
    for i, neigh in enumerate(nbr_ids):
        neigh = [j for j in neigh if j!=i]
        mix[i] = 0.0 if len(neigh)==0 else np.mean(z[neigh] != z[i])

    trees = {
        0: cKDTree(coords[z==0]) if np.any(z==0) else None,
        1: cKDTree(coords[z==1]) if np.any(z==1) else None,
    }
    opp_dist = np.zeros(N, float)
    for i in range(N):
        other = 1 - z[i]
        opp_dist[i] = np.inf if trees[other] is None else trees[other].query(coords[i], k=1)[0]

    t_mix, t_dist = {}, {}
    for g in (0,1):
        idx = np.where(z==g)[0]
        if idx.size:
            t_mix[g]  = _kmeans_1d_threshold(mix[idx], random_state)
            t_dist[g] = _kmeans_1d_threshold(opp_dist[idx], random_state)
        else:
            t_mix[g]  = np.nan
            t_dist[g] = np.nan

    is_core = np.zeros(N, bool)
    is_edge = np.zeros(N, bool)
    idx_t = np.where(z==0)[0]
    if idx_t.size:
        base_core = (mix[idx_t] <= t_mix[0]) & (opp_dist[idx_t] >= t_dist[0])
        islands = _tumor_islands(idx_t, nbr_ids, z)
        ic_local = np.zeros(len(idx_t), bool)
        for comp in islands:
            glob = idx_t[comp]
            if glob.size < min_island:
                pick = glob[np.argmax(opp_dist[glob])]
                ic_local[np.where(idx_t==pick)[0][0]] = True
                continue
            have_base = base_core[comp]
            if have_base.any():
                ic_local[comp[have_base]] = True
            else:
                k_q   = max(1, int(np.ceil(core_top_q * glob.size)))
                k_min = max(1, int(np.ceil(core_min_frac * glob.size)))
                k = max(k_q, k_min)
                order = np.argsort(opp_dist[glob])[::-1]
                chosen = glob[order[:k]]
                for c in chosen:
                    ic_local[np.where(idx_t==c)[0][0]] = True
        is_core[idx_t] = ic_local
        is_edge[idx_t] = ~is_core[idx_t]

    is_transit = np.zeros(N, bool)
    is_interior = np.zeros(N, bool)
    idx_n = np.where(z==1)[0]
    if idx_n.size:
        is_transit[idx_n]  = (mix[idx_n] >= t_mix[1]) | (opp_dist[idx_n] <= t_dist[1])
        is_interior[idx_n] = ~is_transit[idx_n]

    G = 2
    macro_onehot = np.eye(G, dtype=np.int8)[z]
    zone_core     = (np.eye(G, dtype=np.int8)[z] * (is_core[:,None]     & (z[:,None]==0)))
    zone_edge     = (np.eye(G, dtype=np.int8)[z] * (is_edge[:,None]     & (z[:,None]==0)))
    zone_transit  = (np.eye(G, dtype=np.int8)[z] * (is_transit[:,None]  & (z[:,None]==1)))
    zone_interior = (np.eye(G, dtype=np.int8)[z] * (is_interior[:,None] & (z[:,None]==1)))

    thresholds = dict(
        h=float(h), R=float(R), alpha=float(alpha),
        t_mix={0:float(t_mix[0]), 1:float(t_mix[1])},
        t_dist={0:float(t_dist[0]),1:float(t_dist[1])},
        core_policy=dict(top_q=float(core_top_q), min_frac=float(core_min_frac), min_island=int(min_island))
    )

    return dict(
        z=z, coords=coords, barcode=barcode, idx_str=idx_str,
        macro_onehot=macro_onehot, zone_core=zone_core, zone_edge=zone_edge,
        zone_transit=zone_transit, zone_interior=zone_interior,
        mix=mix, opp_dist=opp_dist, thresholds=thresholds
    )
