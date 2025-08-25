import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ARI
import umap
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import os

PATH_HIST_WAV  = ""
TRUTH_CSV_PATH = ""
EXPR_NPZ_PATH  = ""
EXPR_KEY       = "expr"
MASK_KEY       = "mask"
PATH_HIST_WAV  = ""
TRUTH_CSV_PATH = ""
HIST_IMAGE_PATH = None

z = np.load(EXPR_NPZ_PATH, allow_pickle=True)
EXPR_TENSOR = z[EXPR_KEY]
SPOT_MASK   = z[MASK_KEY].astype(bool)
WAV_TENSOR  = np.load(PATH_HIST_WAV)

RANDOM_STATE = 43
GENE_WEIGHT  = 1.0
HIST_WEIGHT  = 1.0

UMAP_N_NEIGHBORS_LIST = [60, 100]
UMAP_MIN_DIST_LIST    = [0.10, 0.20, 0.30]
GRAPH_K_LIST          = [100, 110]
CLUSTER_METHODS       = ["leiden", "louvain"]
LEIDEN_RES_LIST       = [0.3, 0.4, 0.50, 0.8, 1.0]

K_SMOOTH_LIST   = [6, 8, 10]
MIN_FRAC_LIST   = [0.60, 0.70]
MAX_ITER_SMOOTH = 2

def _flatten_tensor(tensor: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray,np.ndarray]]:
    if tensor.ndim != 3:
        raise ValueError("tensor must be 3D (H,W,C)")
    H, W, _ = tensor.shape
    if mask is None:
        mask = np.ones((H, W), dtype=bool)
    else:
        mask = mask.astype(bool)
        if mask.shape != (H, W):
            raise ValueError("mask must be shape (H,W)")
    yy, xx = np.where(mask)
    X = tensor[yy, xx, :]
    coords = np.stack([xx, yy], axis=1)
    return X, coords, (yy, xx)

def _umap_embed(X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, random_state: int):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    return reducer.fit_transform(X)

def _knn_graph_from_embedding(embedding_2d: np.ndarray, n_neighbors: int) -> ig.Graph:
    A = kneighbors_graph(embedding_2d, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    A = A.maximum(A.T)
    src, dst = A.nonzero()
    edges = list({(int(i), int(j)) if i < j else (int(j), int(i)) for i, j in zip(src, dst)})
    g = ig.Graph(n=embedding_2d.shape[0], edges=edges, directed=False)
    g.simplify()
    return g

def _run_clustering(g: ig.Graph, method: str, resolution: float, seed: int) -> np.ndarray:
    if method.lower() == "leiden":
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=resolution, seed=seed)
        return np.array(part.membership, dtype=int)
    elif method.lower() == "louvain":
        cl = g.community_multilevel()
        return np.array(cl.membership, dtype=int)

def knn_label_smooth(labels_1d: np.ndarray,
                     coords_xy_1d: np.ndarray,
                     k: int = 6,
                     min_frac: float = 0.6,
                     max_iter: int = 1) -> np.ndarray:
    if k <= 0:
        return labels_1d.copy()
    N = labels_1d.shape[0]
    k_eff = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(coords_xy_1d)
    neigh_idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    smoothed = labels_1d.copy()
    for _ in range(max_iter):
        flips = 0
        current = smoothed.copy()
        for i in range(N):
            neigh_labels = current[neigh_idx[i]]
            vals, counts = np.unique(neigh_labels, return_counts=True)
            maj = vals[np.argmax(counts)]
            frac = counts.max() / len(neigh_labels)
            if maj != current[i] and frac >= min_frac:
                smoothed[i] = maj
                flips += 1
        if flips == 0:
            break
    return smoothed

z = np.load(EXPR_NPZ_PATH, allow_pickle=True)
EXPR_TENSOR = z[EXPR_KEY]
SPOT_MASK   = z[MASK_KEY].astype(bool)
WAV_TENSOR  = np.load(PATH_HIST_WAV)

H, W, G = EXPR_TENSOR.shape
assert SPOT_MASK.shape == (H, W), f"mask must be {(H,W)}, got {SPOT_MASK.shape}"
assert WAV_TENSOR.shape[:2] == (H, W), f"histo DWT (H,W) must match expr: {WAV_TENSOR.shape[:2]} vs {(H,W)}"
print(f"[shapes] H={H}, W={W}, G={G}, histC={WAV_TENSOR.shape[2]}, spots={SPOT_MASK.sum()}")

X_expr, coords_xy, idx2d = _flatten_tensor(EXPR_TENSOR, SPOT_MASK)
yy, xx = idx2d
X_expr = np.nan_to_num(X_expr, nan=0.0)
gene_std = X_expr.std(axis=0)
keep = gene_std > 0
X_expr = X_expr[:, keep]
X_expr_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X_expr)

X_his, coords_xy_w, idx2d_w = _flatten_tensor(WAV_TENSOR, SPOT_MASK)
X_his = np.nan_to_num(X_his, nan=0.0)
X_his_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X_his)

X_joint = np.hstack([GENE_WEIGHT * X_expr_std, HIST_WEIGHT * X_his_std])
N, F = X_joint.shape
coords_px = np.stack([xx, yy], axis=1)
print(f"[features] N={N}, F={F}")

df = pd.read_csv(TRUTH_CSV_PATH)
def pick(colnames, options):
    for o in options:
        if o in colnames: return o

col_x = pick(df.columns, ["c", "col", "x"])
col_y = pick(df.columns, ["y", "row"])
col_id = pick(df.columns, ["label_id", "labelid", "id"])

df = df[[col_x, col_y, col_id]].copy()
df[col_x] = df[col_x].astype(int)
df[col_y] = df[col_y].astype(int)
coord2id = {(int(r[col_x]), int(r[col_y])): int(r[col_id]) for _, r in df.iterrows()}

true_labels = np.full(N, -1, dtype=int)
for i, (x, y) in enumerate(coords_px):
    true_labels[i] = coord2id.get((int(x), int(y)), -1)

eval_idx = np.where(true_labels >= 0)[0]
print(f"[truth] matched {eval_idx.size}/{N} spots to truth labels")


results = []
best = None

for nn in UMAP_N_NEIGHBORS_LIST:
    for md in UMAP_MIN_DIST_LIST:
        embedding = _umap_embed(X_joint, n_neighbors=nn, min_dist=md, metric="euclidean", random_state=RANDOM_STATE)
        for gk in GRAPH_K_LIST:
            g = _knn_graph_from_embedding(embedding, n_neighbors=gk)
            for method in CLUSTER_METHODS:
                if method == "leiden":
                    res_list = LEIDEN_RES_LIST
                else:
                    res_list = [None]
                for res in res_list:
                    labels_pred = _run_clustering(g, method, resolution=(res if res is not None else 1.0), seed=RANDOM_STATE)
                    ari_pre = ARI(true_labels[eval_idx], labels_pred[eval_idx])
                    for ks in K_SMOOTH_LIST:
                        for mf in MIN_FRAC_LIST:
                            labels_post = knn_label_smooth(labels_pred, coords_px, k=ks, min_frac=mf, max_iter=MAX_ITER_SMOOTH)
                            ari_post = ARI(true_labels[eval_idx], labels_post[eval_idx])
                            row = {
                                "umap_neighbors": nn,
                                "umap_min_dist": md,
                                "graph_k": gk,
                                "method": method,
                                "leiden_res": res if res is not None else np.nan,
                                "k_smooth": ks,
                                "min_frac": mf,
                                "ari_pre": ari_pre,
                                "ari_post": ari_post,
                            }
                            results.append(row)
                            if (best is None) or (ari_post > best["ari_post"]):
                                best = {
                                    **row,
                                    "embedding": embedding,
                                    "labels_pre": labels_pred,
                                    "labels_post": labels_post,
                                }
