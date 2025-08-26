# src/wacccn/segmentation/micro.py
import numpy as np, pandas as pd, umap, igraph as ig, leidenalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

def build_umap(X, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0, pca_dim=50):
    Xz = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=pca_dim, random_state=random_state).fit_transform(Xz) if (pca_dim and pca_dim < Xz.shape[1]) else Xz
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state, n_components=2, verbose=False)
    return reducer.fit_transform(Xp)

def leiden_on_embedding(emb2d, k=15, resolution=1.0, random_state=0):
    A = kneighbors_graph(emb2d, n_neighbors=k, mode="connectivity", include_self=False)
    A = A.maximum(A.T)
    src, dst = A.nonzero()
    g = ig.Graph(n=emb2d.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=int(random_state))
    return np.array(part.membership, dtype=int)

def microniche_segment(expr_df, wavelet_arr, macro_series,
                       umap_neighbors=15, umap_min_dist=0.1, umap_metric="euclidean", pca_dim=50,
                       leiden_k=15, resolution=1.0, min_cluster_size=10, seed=0):
    X_fused = np.hstack([expr_df.to_numpy(), wavelet_arr])
    emb = build_umap(X_fused, n_neighbors=umap_neighbors, min_dist=umap_min_dist, metric=umap_metric, random_state=seed, pca_dim=pca_dim)

    spots = expr_df.index.to_numpy()
    N = spots.shape[0]
    macros = macro_series.values.astype(str)
    uniq = np.unique(macros).tolist()

    labels = np.full(N, -1, dtype=int)
    names = np.full(N, "", dtype=object)
    blocks = []
    next_global = 0

    for g in uniq:
        idx = np.where(macros == g)[0]
        if idx.size < max(leiden_k + 1, min_cluster_size):
            labels[idx] = next_global; names[idx] = f"{g}:0"; blocks.append((g, 0, idx)); next_global += 1; continue

        local = leiden_on_embedding(emb[idx], k=leiden_k, resolution=resolution, random_state=seed)
        unique_local = np.unique(local)
        sizes = {c: int(np.sum(local == c)) for c in unique_local}
        keep = [c for c, s in sizes.items() if s >= min_cluster_size]

        if keep:
            kept_centroids = {c: emb[idx[local == c]].mean(axis=0) for c in keep}
            local_to_global = {}
            for lc in sorted(keep):
                ii = idx[local == lc]
                local_to_global[lc] = next_global
                labels[ii] = next_global
                names[ii] = f"{g}:{lc}"
                blocks.append((g, lc, ii))
                next_global += 1
            tiny = [c for c in unique_local if c not in keep]
            for lc in tiny:
                ii = idx[local == lc]
                if ii.size == 0: continue
                c_centroid = emb[ii].mean(axis=0)
                best = min(keep, key=lambda kk: np.linalg.norm(c_centroid - kept_centroids[kk]))
                glab = local_to_global[best]
                labels[ii] = glab
                names[ii] = f"{g}:{best}"
        else:
            labels[idx] = next_global; names[idx] = f"{g}:0tiny"; blocks.append((g, 0, idx)); next_global += 1

    L = int(labels.max() + 1)
    onehot = np.zeros((N, L), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab >= 0: onehot[i, lab] = 1.0

    out = {
        "embedding": emb,             # (N,2)
        "labels": labels,             # (N,)
        "names": names,               # (N,)
        "onehot": onehot,             # (N,L)
        "blocks": blocks,             # list of (macro, local_id, idx_array)
        "spots": spots,               # (N,)
    }
    return out
