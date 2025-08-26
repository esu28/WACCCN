# marker genes
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (n / rank) * ranked[i]
        cummin = min(cummin, val)
        q[i] = min(cummin, 1.0)
    out = np.empty_like(q); out[order] = q
    return out

def _log2fc_pos_vs_rest(a, b, eps=1e-9):
    return np.log2((float(np.mean(a)) + eps) / (float(np.mean(b)) + eps))

def de_for_cluster(expr_df, in_mask):
    in_idx = np.where(in_mask)[0]; out_idx = np.where(~in_mask)[0]
    if in_idx.size == 0 or out_idx.size == 0: raise ValueError("Empty group for DE.")
    X = expr_df.to_numpy(); G = X.shape[1]
    pvals = np.empty(G); l2fc = np.empty(G)
    for g in range(G):
        u = mannwhitneyu(X[in_idx, g], X[out_idx, g], alternative="two-sided")
        pvals[g] = u.pvalue
        l2fc[g] = _log2fc_pos_vs_rest(X[in_idx, g], X[out_idx, g])
    padj = bh_fdr(pvals); genes = np.array(expr_df.columns)
    return pd.DataFrame({"gene": genes, "log2FC": l2fc, "pval": pvals, "padj": padj}) \
             .sort_values(["padj","log2FC"], ascending=[True, False], kind="mergesort")

def markers_from_labels(expr_df, labels, min_cluster_size=10, top_per_cluster=50, padj_thresh=0.05, final_markers=200):
    L = int(labels.max() + 1)
    gene_markers = []
    for k in range(L):
        in_mask = (labels == k)
        if in_mask.sum() < min_cluster_size: continue
        df = de_for_cluster(expr_df, in_mask)
        sig = df[df["padj"] < padj_thresh].sort_values(["log2FC"], ascending=False)
        if sig.shape[0] > top_per_cluster: sig = sig.iloc[:top_per_cluster]
        sig = sig.copy(); sig["best_cluster"] = k
        gene_markers.append(sig)
    if not gene_markers:
        raise RuntimeError("No significant markers. Relax thresholds or cluster size.")
    cat = pd.concat(gene_markers, ignore_index=True)

    def _best_idx(gdf): return int(gdf["log2FC"].values.argmax())

    agg = (
        cat.groupby("gene", group_keys=False)
        .apply(lambda g: pd.Series({
            "max_log2FC": float(g["log2FC"].max()),
            "min_padj": float(g["padj"].min()),
            "best_cluster": int(g.iloc[_best_idx(g)]["best_cluster"]),
        }))
        .reset_index()
        .sort_values(["max_log2FC","min_padj"], ascending=[False, True])
    )
    if agg.shape[0] > final_markers: agg = agg.iloc[:final_markers]
    marker_genes = agg["gene"].tolist()
    missing = [g for g in marker_genes if g not in expr_df.columns]
    if missing: raise RuntimeError(f"Missing {len(missing)} selected markers in expr. Example: {missing[:5]}")
    X_mark = expr_df[marker_genes].to_numpy()
    X_mark_z = StandardScaler().fit_transform(X_mark)
    return agg, marker_genes, X_mark_z
