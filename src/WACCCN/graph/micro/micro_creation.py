# microniches 
import numpy as np, pandas as pd, umap, igraph as ig, leidenalg as la
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

def _looks_like_genes_x_spots(df):
    first = df.columns[0]
    col0_nonnum = not pd.api.types.is_numeric_dtype(df[first])
    other_numeric = df.drop(columns=[first]).select_dtypes(include=[np.number]).shape[1]
    return col0_nonnum and other_numeric >= max(5, int(0.5 * (df.shape[1]-1)))

def load_expr(expr_csv, spot_col=None, log1p=False, expr_format="auto"):
    df_raw = pd.read_csv(expr_csv)
    if expr_format not in {"auto","spots_by_genes","genes_by_spots"}:
        raise ValueError("expr_format must be auto/spots_by_genes/genes_by_spots")

    if expr_format == "spots_by_genes":
        df = df_raw.set_index(spot_col) if (spot_col and spot_col in df_raw.columns) else df_raw
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0: raise ValueError("No numeric gene columns (spots_by_genes).")
        out = num.copy()

    elif expr_format == "genes_by_spots":
        gene_col = df_raw.columns[0]
        df = df_raw.set_index(gene_col).apply(pd.to_numeric, errors="coerce")
        out = df.T.dropna(axis=0, how="all").dropna(axis=1, how="all")

    else:
        if spot_col and spot_col in df_raw.columns:
            df = df_raw.set_index(spot_col)
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] > 0:
                out = num.copy()
            else:
                if _looks_like_genes_x_spots(df_raw):
                    gene_col = df_raw.columns[0]
                    df2 = df_raw.set_index(gene_col).apply(pd.to_numeric, errors="coerce")
                    out = df2.T.dropna(axis=0, how="all").dropna(axis=1, how="all")
                else:
                    raise ValueError("Cannot auto-parse expression CSV; set expr_format.")
        else:
            if _looks_like_genes_x_spots(df_raw):
                gene_col = df_raw.columns[0]
                df = df_raw.set_index(gene_col).apply(pd.to_numeric, errors="coerce")
                out = df.T.dropna(axis=0, how="all").dropna(axis=1, how="all")
            else:
                num = df_raw.select_dtypes(include=[np.number])
                if num.shape[1] == 0: raise ValueError("Expression CSV not understood.")
                out = num.copy()

    if log1p: out = np.log1p(out)
    out.index = out.index.astype(str)
    return out

def load_wavelet(path, spot_col=None):
    if path.endswith(".npy"):
        arr = np.load(path); return arr, None
    df = pd.read_csv(path)
    if spot_col is not None:
        if spot_col not in df.columns: raise ValueError(f"Wavelet CSV missing spot column '{spot_col}'.")
        ids = df[spot_col].astype(str).values
        X = df.drop(columns=[spot_col]).select_dtypes(include=[np.number]).to_numpy()
        return X, ids
    X = df.select_dtypes(include=[np.number]).to_numpy()
    return X, None

def load_macro(macro_csv, macro_col, spot_col):
    df = pd.read_csv(macro_csv)
    if spot_col not in df.columns: raise ValueError("macro_csv missing spot_col.")
    if macro_col not in df.columns: raise ValueError("macro_csv missing macro_col.")
    s = df[[spot_col, macro_col]].set_index(spot_col)[macro_col]
    s.index = s.index.astype(str)
    return s.astype(str)

def align_by_index(expr_df, macro_series, wavelet_arr, wavelet_spot_ids, out_dir=None):
    expr_df = expr_df.copy(); expr_df.index = expr_df.index.astype(str)
    macro_series = macro_series.copy(); macro_series.index = macro_series.index.astype(str)

    expr_ids, macro_ids = set(expr_df.index), set(macro_series.index)
    overlap = len(expr_ids & macro_ids)
    if overlap == 0:
        ex_expr = list(sorted(expr_df.index))[:5]; ex_macro = list(sorted(macro_series.index))[:5]
        raise ValueError(f"No overlap between expr and macro IDs. expr~{ex_expr} vs macro~{ex_macro}")

    frac = overlap / max(1, len(expr_ids))
    if frac < 0.9:
        msg = (f"Low ID overlap: {overlap}/{len(expr_ids)} ({frac:.1%}). Likely cross-slide mismatch or wrong spot_col.")
        try:
            if out_dir: Path(out_dir, "alignment_warning.txt").write_text(msg)
        except Exception: pass
        raise ValueError(msg)

    missing = expr_ids - macro_ids
    if missing:
        ex = sorted(list(missing))[:5]
        raise ValueError(f"{len(missing)} expr spots missing in macro labels. Example: {ex}")

    macro_series = macro_series.loc[expr_df.index]

    if wavelet_spot_ids is None:
        if wavelet_arr.shape[0] != expr_df.shape[0]:
            raise ValueError("Wavelet rows != #expr spots and no spot IDs in wavelet file.")
        Xw = wavelet_arr
    else:
        id_to_row = {str(s): i for i, s in enumerate(wavelet_spot_ids)}
        idxs, miss_w = [], []
        for s in expr_df.index:
            i = id_to_row.get(s, None)
            (idxs if i is not None else miss_w).append(i if i is not None else s)
        if miss_w: raise ValueError(f"Wavelet missing {len(miss_w)} expr spots. Example: {miss_w[:5]}")
        Xw = wavelet_arr[np.asarray(idxs)]

    return expr_df, macro_series, Xw

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
    macros = macro_series.values.astype(str)
    uniq = np.unique(macros).tolist()

    N = spots.shape[0]
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
                labels[ii] = next_global; names[ii] = f"{g}:{lc}"; blocks.append((g, lc, ii)); next_global += 1
            tiny = [c for c in unique_local if c not in keep]
            for lc in tiny:
                ii = idx[local == lc]
                if ii.size == 0: continue
                c_centroid = emb[ii].mean(axis=0)
                best = min(keep, key=lambda kk: np.linalg.norm(c_centroid - kept_centroids[kk]))
                glab = local_to_global[best]
                labels[ii] = glab; names[ii] = f"{g}:{best}"
        else:
            labels[idx] = next_global; names[idx] = f"{g}:0tiny"; blocks.append((g, 0, idx)); next_global += 1

    L = int(labels.max() + 1)
    onehot = np.zeros((N, L), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab >= 0: onehot[i, lab] = 1.0

    return {"embedding": emb, "labels": labels, "names": names, "onehot": onehot, "blocks": blocks, "spots": spots}
