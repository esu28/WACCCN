import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ccc_shared import (
    median_1nn, knn_mean_distance, edge_distances, scatter_to_dense,
    per_source_normalize_K, hist_evidence_edges, robust_Z_from_similarity,
    auto_beta_plus, infer_mode_from_text, load_aniso_npz, hill_response
)

def load_coords_csv(path):
    df = pd.read_csv(path)
    need = ["barcode","x","y"]
    got = [str(c).strip().lower() for c in df.columns[:3]]
    if got != need:
        raise ValueError(f"coords CSV first three columns must be {need}. got {got}")
    bcol, xcol, ycol = df.columns[:3]
    barcodes = df[bcol].astype(str).values
    coords = df[[xcol, ycol]].to_numpy(float)
    xy_keys = (np.round(coords[:, 0]).astype(int).astype(str) + "_" +
               np.round(coords[:, 1]).astype(int).astype(str))
    return coords, barcodes, xy_keys

def load_expr_gene_rows_csv(path):
    df = pd.read_csv(path)
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    barcodes = df.columns.astype(str)
    X = df.apply(pd.to_numeric, errors='coerce').T
    X.index = X.index.astype(str)
    Xv = X.to_numpy(float)
    p99 = np.nanpercentile(Xv, 99, axis=0)
    Xv = np.minimum(Xv, p99)
    lo = np.nanmin(Xv, axis=0)
    hi = np.nanmax(Xv, axis=0)
    den = np.maximum(hi - lo, 1e-12)
    Xv = np.clip((Xv - lo) / den, 0.0, 1.0)
    df01 = pd.DataFrame(Xv, index=X.index, columns=X.columns.astype(str))
    return df01, barcodes

def load_lr(paths):
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [str(c).strip().lower() for c in df.columns]
        need = {"ligand","receptor","pathway","ligand_annotation"}
        if not need.issubset(set(df.columns)):
            missing = sorted(list(need - set(df.columns)))
            raise ValueError(f"LR CSV missing {missing} in {p}")
        rows.append(df)
    LR = pd.concat(rows, axis=0, ignore_index=True)
    ann = (LR["ligand_annotation"].astype(str) + " " + LR["pathway"].astype(str))
    mode = ann.apply(lambda s: infer_mode_from_text(s) or "secreted")
    mode = mode.replace({"sec":"secreted","secret":"secreted","soluble":"secreted","membrane":"mb","membranous":"mb","juxtacrine":"mb"})
    mode = mode.where(mode.isin(["secreted","mb"]), "secreted")
    out = pd.DataFrame({
        "ligand": LR["ligand"].astype(str).str.strip(),
        "receptor": LR["receptor"].astype(str).str.strip(),
        "mode": mode,
        "pathway": LR["pathway"].astype(str).str.strip(),
    })
    out = out[(out["ligand"]!="") & (out["receptor"]!="")].reset_index(drop=True)
    return out

def load_tf_auc(tf_csv, coords_barcodes, xy_keys):
    if tf_csv is None:
        return None
    df = pd.read_csv(tf_csv)
    if df.shape[1] < 2:
        raise ValueError("TF CSV must have a 'spot' column and TF columns.")
    cols = [c.lower() for c in df.columns]
    scol = df.columns[cols.index('spot')] if 'spot' in cols else df.columns[0]
    df[scol] = df[scol].astype(str)
    df = df.set_index(scol)
    tfmat = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    idx = pd.Index(xy_keys.astype(str))
    Delta = tfmat.sum(axis=1).reindex(idx, fill_value=0.0)
    Delta.name = 'Delta'
    return Delta

def default_params():
    return dict(
        sigma_sim=1.5, q_robust=0.60, zcap=3.0,
        gamma_big=0.10, f_target=0.10, beta_max=0.08, beta_neg=0.0,
        K_half=0.01,
        R_factor=None, deg_min=None, deg_max=None,
        Wr_factor=0.50, k_for_Wr=1,
        n_null=1000, pool_size=1000, seed=0,
        penalty=0.0, prune_frac=0.0
    )

def merge_params(base, updates):
    cfg = dict(base)
    if updates:
        for k, v in updates.items():
            if k in cfg:
                cfg[k] = v
    return cfg

def score_all_pairs(coords, expr, lr_df, W_his, tf_auc_series, aniso_npz, params,
                    save_dense_pair=False, dense_pair_max_floats=50_000_000,
                    save_dense_pathway=True, dense_pathway_max_floats=50_000_000):
    rng = np.random.default_rng(int(params.get('seed', 0)))
    N = coords.shape[0]

    Z = load_aniso_npz(aniso_npz)
    edges = Z['edges'].astype(int)
    K_edge = np.asarray(Z['K'], float).ravel()
    if K_edge.size != edges.shape[0]:
        raise ValueError("K length mismatch")

    if params.get('R_factor') is not None and params['R_factor'] > 0:
        h = median_1nn(coords)
        d_all = edge_distances(coords, edges)
        keep = d_all <= (params['R_factor'] * h)
        if keep.sum() > 0:
            edges = edges[keep]
            K_edge = K_edge[keep]

    if params.get('deg_max') is not None and params['deg_max'] > 0:
        d_cur = edge_distances(coords, edges)
        order = np.argsort(d_cur)
        edges = edges[order]; K_edge = K_edge[order]; d_cur = d_cur[order]
        src = edges[:, 0]
        keep_mask = np.zeros(edges.shape[0], dtype=bool)
        seen = set()
        for s in src:
            if s in seen: continue
            seen.add(int(s))
            sel = np.where(src == s)[0]
            if sel.size <= params['deg_max']:
                keep_mask[sel] = True
            else:
                keep_mask[sel[:params['deg_max']]] = True
        if (~keep_mask).sum() > 0:
            edges = edges[keep_mask]; K_edge = K_edge[keep_mask]

    K_edge = per_source_normalize_K(K_edge, edges, N)

    M = hist_evidence_edges(W_his, edges, sigma=params['sigma_sim'])
    Zg = robust_Z_from_similarity(M, q=params['q_robust'], zcap=params['zcap'])
    beta_p = auto_beta_plus(Zg, gamma_big=params['gamma_big'], f_target=params['f_target'], beta_max=params['beta_max'])
    G_edge = np.exp(beta_p * np.maximum(Zg, 0.0) - params['beta_neg'] * np.maximum(-Zg, 0.0))

    Wr = params['Wr_factor'] * knn_mean_distance(coords, k=params['k_for_Wr'])
    R_contact = 0.9 * median_1nn(coords)

    if tf_auc_series is not None:
        Delta = tf_auc_series.values
        S_D = 1.0 + (Delta / (0.5 + Delta + 1e-12))
        S_D = np.clip(S_D, 1.0, 2.0)
    else:
        S_D = np.ones(N, float)

    d = edge_distances(coords, edges)
    radial = np.exp(-(d / max(Wr, 1e-9))**2)
    contact_mask = (d <= R_contact).astype(float)

    expr_means = expr.mean(axis=0)
    genes = np.array(expr.columns)

    E = edges.shape[0]
    S_total_edge = np.zeros(E, dtype=np.float64)
    S_by_pair = []
    meta_L = []
    meta_R = []
    meta_mode = []
    meta_pathway = []

    for _, row in lr_df.iterrows():
        L = row['ligand']; R = row['receptor']; mode = str(row['mode']).lower()
        pathway = str(row.get('pathway', 'unknown'))
        if L not in expr.columns or R not in expr.columns:
            continue
        EL = np.asarray(expr[L].values, float); EL = np.clip(EL, 0.0, 1.0)
        ER = np.asarray(expr[R].values, float); ER = np.clip(ER, 0.0, 1.0)
        if mode in {'mb','membrane','membrane-bound'}:
            C_base = EL[edges[:, 0]] * contact_mask
        else:
            C_base = EL[edges[:, 0]] * K_edge * radial
        S_raw = hill_response(C_base * G_edge * ER[edges[:, 1]], params['K_half'])

        if params.get('n_null', 0):
            diffs_L = np.abs(expr_means - expr_means[L]); pool_L = genes[np.argsort(diffs_L.values)[:params['pool_size']]]
            diffs_R = np.abs(expr_means - expr_means[R]); pool_R = genes[np.argsort(diffs_R.values)[:params['pool_size']]]
            null_scores = np.zeros((int(params['n_null']), E), dtype=np.float32)
            cache = {}
            def get01(g):
                if g not in cache:
                    v = np.asarray(expr[g].values, float)
                    cache[g] = np.clip(v, 0.0, 1.0)
                return cache[g]
            for t in range(int(params['n_null'])):
                gL = rng.choice(pool_L); gR = rng.choice(pool_R)
                ELn = get01(str(gL)); ERn = get01(str(gR))
                if mode in {'mb','membrane','membrane-bound'}:
                    Cn = ELn[edges[:, 0]] * contact_mask
                else:
                    Cn = ELn[edges[:, 0]] * K_edge * radial
                null_scores[t, :] = hill_response(Cn * G_edge * ERn[edges[:, 1]], params['K_half'])
            tau = np.quantile(null_scores, 0.95, axis=0)
            surplus = S_raw - tau
            S_thr = np.maximum(surplus - max(params.get('penalty', 0.0), 0.0), 0.0)
        else:
            S_thr = np.maximum(S_raw - max(params.get('penalty', 0.0), 0.0), 0.0)

        S_final = S_thr * S_D[edges[:, 1]]
        S_total_edge += S_final
        S_by_pair.append(S_final.astype(np.float32))
        meta_L.append(L); meta_R.append(R)
        meta_mode.append('mb' if mode in {'mb','membrane','membrane-bound'} else 'secreted')
        meta_pathway.append(pathway)

    if len(S_by_pair) > 0:
        S_edge_by_pair = np.stack(S_by_pair, axis=1).astype(np.float32)
        pair_path = np.array(meta_pathway, dtype='<U64')
        uniq_paths, inv = np.unique(pair_path, return_inverse=True)
        Kp = len(uniq_paths)
        S_edge_by_pathway = np.zeros((E, Kp), dtype=np.float32)
        for k in range(Kp):
            cols = np.where(inv == k)[0]
            if cols.size:
                S_edge_by_pathway[:, k] = S_edge_by_pair[:, cols].sum(axis=1).astype(np.float32)
    else:
        S_edge_by_pair = np.zeros((E, 0), dtype=np.float32)
        uniq_paths = np.array([], dtype='<U1')
        S_edge_by_pathway = np.zeros((E, 0), dtype=np.float32)

    q = float(params.get('prune_frac', 0.0) or 0.0)
    if q > 0:
        q = float(np.clip(q, 0.0, 0.999))
        thr = np.quantile(S_total_edge, q)
        keep_mask = S_total_edge >= thr
        if (~keep_mask).sum() > 0:
            S_total_edge[~keep_mask] = 0.0
            if S_edge_by_pair.size:
                S_edge_by_pair[~keep_mask, :] = 0.0
            if S_edge_by_pathway.size:
                S_edge_by_pathway[~keep_mask, :] = 0.0

    S_total_edge = S_total_edge.astype(np.float32)
    A_final = scatter_to_dense(N, edges, S_total_edge)

    A_by_pathway = None
    if save_dense_pathway and S_edge_by_pathway.size > 0:
        Kp = S_edge_by_pathway.shape[1]
        need = Kp * (N * N)
        if need <= int(dense_pathway_max_floats):
            A_by_pathway = np.zeros((Kp, N, N), dtype=np.float32)
            for k in range(Kp):
                A_by_pathway[k] = scatter_to_dense(N, edges, S_edge_by_pathway[:, k])

    A_by_pair = None
    P = S_edge_by_pair.shape[1]
    if save_dense_pair and P > 0:
        need = P * (N * N)
        if need <= int(dense_pair_max_floats):
            A_by_pair = np.zeros((P, N, N), dtype=np.float32)
            for pidx in range(P):
                A_by_pair[pidx] = scatter_to_dense(N, edges, S_edge_by_pair[:, pidx])

    out = dict(
        edges=edges.astype(np.int32),
        S_edge=S_total_edge,
        S_edge_by_pair=S_edge_by_pair,
        pair_ligand=np.array(meta_L, dtype='<U64'),
        pair_receptor=np.array(meta_R, dtype='<U64'),
        pair_mode=np.array(meta_mode, dtype='<U16'),
        pair_pathway=np.array(meta_pathway, dtype='<U64'),
        pathway_names=uniq_paths,
        S_edge_by_pathway=S_edge_by_pathway,
        A_final=A_final,
        Wr=float(Wr),
        R_contact=float(R_contact),
        h=float(median_1nn(coords)),
        params_json=None
    )
    if A_by_pathway is not None:
        out['A_by_pathway'] = A_by_pathway
    if A_by_pair is not None:
        out['A_by_pair'] = A_by_pair
    return out
