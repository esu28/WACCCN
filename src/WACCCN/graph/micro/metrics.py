# marker genes 
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

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

def _log2fc_pos_vs_rest(group_vals, rest_vals, eps=1e-9):
    mu_g = float(np.mean(group_vals)); mu_r = float(np.mean(rest_vals))
    return np.log2((mu_g + eps) / (mu_r + eps))

def de_for_cluster(expr_df, in_mask):
    in_idx = np.where(in_mask)[0]; out_idx = np.where(~in_mask)[0]
    if in_idx.size == 0 or out_idx.size == 0:
        raise ValueError("Empty group for DE.")
    X = expr_df.to_numpy(); G = X.shape[1]
    pvals = np.empty(G, dtype=float); l2fc = np.empty(G, dtype=float)
    for g in range(G):
        u = mannwhitneyu(X[in_idx, g], X[out_idx, g], alternative="two-sided")
        pvals[g] = u.pvalue
        l2fc[g] = _log2fc_pos_vs_rest(X[in_idx, g], X[out_idx, g])
    padj = bh_fdr(pvals)
    genes = np.array(expr_df.columns)
    return pd.DataFrame({"gene": genes, "log2FC": l2fc, "pval": pvals, "padj": padj}) \
             .sort_values(["padj", "log2FC"], ascending=[True, False], kind="mergesort")
