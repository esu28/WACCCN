#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#refine graph with ccc inference scores
import os, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from collections import defaultdict, deque

eps = 1e-12

def load_coords(path):
    if path.endswith(".npy"):
        arr = np.load(path)
        return arr[:, -2:].astype(float) if arr.shape[1] >= 2 else arr.astype(float)
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    xname = next((lc.get(c) for c in ["x","x_px","coord_x","pos_x","xc","col"]), None)
    yname = next((lc.get(c) for c in ["y","y_px","coord_y","pos_y","yc","row"]), None)
    if xname and yname:
        return df[[xname, yname]].to_numpy(dtype=float)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError("Could not infer x/y columns in CSV.")
    return num.iloc[:, :2].to_numpy(dtype=float)

def load_labels(path):
    if path.endswith(".npy"):
        return np.load(path).astype(int)
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    for key in ["macro","class","label","z","z_i","macro_niche","niche"]:
        if key in lc:
            return df[lc[key]].to_numpy(dtype=int)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().all():
            return s.to_numpy(dtype=int)
    raise ValueError("Could not infer macro labels from CSV.")

def load_matrix(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if "W" in data:
            return data["W"]
        for k in data.files:
            arr = data[k]
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                return arr
    raise ValueError("Provide a dense NxN CCC final matrix (.npy or .npz with key 'W').")

def median_1nn_by_class(coords, labels):
    uniq = np.unique(labels)
    h_g = []
    for g in uniq:
        idx = np.where(labels == g)[0]
        if idx.size < 2:
            continue
        tree = cKDTree(coords[idx])
        d, _ = tree.query(coords[idx], k=2)
        h_g.append(np.median(d[:, 1]))
    if not h_g:
        tree = cKDTree(coords)
        d, _ = tree.query(coords, k=2)
        return float(np.median(d[:, 1])), np.array([float(np.median(d[:, 1]))])
    return float(np.median(h_g)), np.array(h_g, dtype=float)

def pooled_knn_distances(coords, kmin=8, kmax=12):
    tree = cKDTree(coords)
    D = []
    for k in range(kmin, kmax + 1):
        d, _ = tree.query(coords, k=k + 1)
        D.append(d[:, k])
    return np.concatenate(D, axis=0)

def kneedle_alpha_from_cdf(D_pool_over_h, lo=1.2, hi=3.5):
    y = np.sort(D_pool_over_h)
    n = y.size
    if n < 10:
        return None
    x = np.linspace(0.0, 1.0, n)
    x0, y0 = 0.0, y[0]
    x1, y1 = 1.0, y[-1]
    vx, vy = (x1 - x0), (y1 - y0)
    v_norm = (vx*vx + vy*vy)**0.5 + eps
    dx = x - x0
    dy = y - y0
    cross = np.abs(vx * dy - vy * dx)
    diffs = cross / v_norm
    idx = int(np.argmax(diffs))
    alpha = float(y[idx])
    if np.isnan(alpha) or alpha < lo or alpha > hi:
        return None
    return alpha

def build_directed_within_radius(coords, R):
    tree = cKDTree(coords)
    nbrs = tree.query_ball_point(coords, r=R)
    I, J = [], []
    for i, neigh in enumerate(nbrs):
        for j in neigh:
            if j == i:
                continue
            I.append(i); J.append(j)
    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)
    return I, J, np.linalg.norm(coords[I] - coords[J], axis=1)

def weakly_lcc_fraction(n_nodes, I, J):
    G = [[] for _ in range(n_nodes)]
    for a, b in zip(I, J):
        G[a].append(b)
        G[b].append(a)
    seen = np.zeros(n_nodes, dtype=bool)
    best = 0
    for s in range(n_nodes):
        if seen[s]:
            continue
        q = deque([s])
        seen[s] = True
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for v in G[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        if size > best:
            best = size
    return best / float(n_nodes)

def tune_alpha_for_degree(coords, h, alpha0, deg_target=(12, 22), lcc_min=0.9,
                          max_iter=12, bounds=(0.9, 4.0)):
    lo, hi = bounds
    alpha = alpha0
    last_good = None
    for _ in range(max_iter):
        R = alpha * h
        I, J, _ = build_directed_within_radius(coords, R)
        if len(I) == 0:
            alpha = min(max(alpha * 1.25, alpha + 0.1), hi)
            continue
        mean_out = len(I) / float(coords.shape[0])
        lcc_frac = weakly_lcc_fraction(coords.shape[0], I, J)
        if deg_target[0] <= mean_out <= deg_target[1] and lcc_frac >= lcc_min:
            last_good = (alpha, mean_out, lcc_frac)
            break
        if mean_out < deg_target[0] or lcc_frac < lcc_min:
            lo = max(lo, alpha)
            alpha = (alpha + hi) / 2.0
        else:
            hi = min(hi, alpha)
            alpha = (alpha + lo) / 2.0
    if last_good is None:
        R = alpha * h
        I, J, _ = build_directed_within_radius(coords, R)
        mean_out = len(I) / float(coords.shape[0]) if len(I) else 0.0
        lcc_frac = weakly_lcc_fraction(coords.shape[0], I, J) if len(I) else 0.0
        last_good = (alpha, mean_out, lcc_frac)
    return last_good

def refine_with_ccc(I0, J0, W_final, q=0.05, eps_tau=1e-12):
    w0 = np.asarray([W_final[i, j] for i, j in zip(I0, J0)], dtype=float)
    if w0.size == 0:
        return I0, J0, w0, 0.0
    tau = float(np.quantile(w0, q))
    if tau <= eps_tau and np.any(w0 > eps_tau):
        tau = eps_tau
    keep = np.where(w0 >= tau)[0]
    I1 = I0[keep]; J1 = J0[keep]; w1 = w0[keep]
    return I1, J1, w1, tau

def undirected_summary(I, J, w):
    bucket = defaultdict(float)
    for a, b, ww in zip(I, J, w):
        key = (a, b) if a < b else (b, a)
        bucket[key] += ww
    U, V, S = [], [], []
    for (a, b), ssum in bucket.items():
        U.append(a); V.append(b); S.append(0.5 * ssum)
    return np.asarray(U, int), np.asarray(V, int), np.asarray(S, float)

def h_hop_neighborhood(n_nodes, I, J, seeds, h=2, directed_weak=True):
    seeds = np.asarray(seeds, dtype=int)
    Gf = [[] for _ in range(n_nodes)]
    if directed_weak:
        for a, b in zip(I, J):
            Gf[a].append(b)
            Gf[b].append(a)
    else:
        for a, b in zip(I, J):
            Gf[a].append(b)
    seen = set(int(s) for s in seeds)
    frontier = set(int(s) for s in seeds)
    for _ in range(h):
        nxt = set()
        for u in frontier:
            for v in Gf[u]:
                if v not in seen:
                    nxt.add(v)
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    nodes = np.array(sorted(seen), dtype=int)
    mask = np.zeros(len(I), dtype=bool)
    idx = {u: i for i, u in enumerate(nodes)}
    for e, (a, b) in enumerate(zip(I, J)):
        if a in idx and b in idx:
            mask[e] = True
    return nodes, mask

def refine_graph(coords, macro, W_final, kmin=8, kmax=12, deg_min=12.0, deg_max=22.0,
                 lcc_min=0.9, q_prune=0.05, alpha_fallback=2.3):
    N = coords.shape[0]
    if W_final.shape != (N, N):
        raise ValueError("W_final shape must be (N,N) matching coords.")

    h, h_g = median_1nn_by_class(coords, macro)
    Dpool = pooled_knn_distances(coords, kmin, kmax)
    alpha0 = kneedle_alpha_from_cdf(Dpool / (h + eps)) or float(alpha_fallback)

    alpha, mean_out0, lcc0 = tune_alpha_for_degree(
        coords, h, alpha0,
        deg_target=(deg_min, deg_max),
        lcc_min=lcc_min, max_iter=12, bounds=(0.9, 4.0)
    )
    R = alpha * h

    I0, J0, D0 = build_directed_within_radius(coords, R)
    I1, J1, w1, tau = refine_with_ccc(I0, J0, W_final, q=q_prune)
    mean_out1 = len(I1) / float(N) if len(I1) else 0.0
    lcc1 = weakly_lcc_fraction(N, I1, J1) if len(I1) else 0.0
    U, V, S = undirected_summary(I1, J1, w1)

    meta = dict(
        N=int(N),
        h=float(h),
        h_g=h_g.astype(float).tolist(),
        alpha0=float(alpha0),
        alpha=float(alpha),
        R=float(R),
        kmin=int(kmin),
        kmax=int(kmax),
        deg_min=float(deg_min),
        deg_max=float(deg_max),
        lcc_min=float(lcc_min),
        mean_out_A0=float(mean_out0),
        lcc_A0=float(lcc0),
        q_prune=float(q_prune),
        tau=float(tau),
        mean_out_A1=float(mean_out1),
        lcc_A1=float(lcc1),
    )
    return I0, J0, D0, I1, J1, w1, U, V, S, meta
