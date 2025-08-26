#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd

from ccc_hparm import run_ccc_sanity_tests, CCCParams
from ccc_score import load_coords_csv, load_lr, load_tf_auc


def load_expr_matrix_csv(expr_csv, coords_barcodes):
    df = pd.read_csv(expr_csv)
    if df.shape[1] < 2:
        raise ValueError("expression needs a gene column plus barcode columns")
    mat = df.iloc[:, 1:].copy()
    genes = df.iloc[:, 0].astype(str)
    mat.index = genes
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors='coerce')
    mat = mat.fillna(0.0)
    missing = [bc for bc in coords_barcodes if bc not in mat.columns]
    for bc in missing:
        mat[bc] = 0.0
    mat = mat.reindex(columns=list(coords_barcodes), fill_value=0.0)
    mat = mat.groupby(level=0).max()
    expr_df = mat.T
    if list(expr_df.index) != list(coords_barcodes):
        expr_df = expr_df.reindex(coords_barcodes).fillna(0.0)
    return expr_df


def scale_expr01(expr_df):
    X = expr_df.to_numpy(float)
    p99 = np.nanpercentile(X, 99.0, axis=0)
    X = np.minimum(X, p99[None, :])
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    den = np.maximum(hi - lo, 1e-12)
    X01 = np.clip((X - lo) / den, 0.0, 1.0)
    return pd.DataFrame(X01, index=expr_df.index, columns=expr_df.columns)


def objective(metrics):
    m = metrics or {}
    pen = 0.0
    conn = m.get('test1_connectivity', {})
    comps = conn.get('n_components', 1)
    mean_deg = conn.get('mean_degree', np.nan)
    pen += 5.0 * max(0, comps - 1)
    if not np.isnan(mean_deg):
        if mean_deg < 12: pen += (12 - mean_deg) / 5.0
        if mean_deg > 22: pen += (mean_deg - 22) / 5.0
    t2 = m.get('test2_null_pass_rate_mean', np.nan)
    if not np.isnan(t2):
        pen += abs(t2 - 0.05) / 0.05
    t3 = m.get('test3_spearman_dist_ccc_mean', np.nan)
    if not np.isnan(t3):
        pen += max(0.0, t3 + 0.20) * 6.0
    gate_big = m.get('test4_gate_big_frac', np.nan)
    if not np.isnan(gate_big):
        pen += abs(gate_big - 0.10) / 0.10
        if gate_big == 0.0:
            pen += 0.5
    mb_corr = m.get('test4_corr_noGate_vs_Gate', np.nan)
    if not np.isnan(mb_corr):
        pen += max(0.0, 0.98 - mb_corr) * 2.0
    t5_above = m.get('test5_hill_frac_aboveK_mean', np.nan)
    if not np.isnan(t5_above):
        pen += abs(t5_above - 0.20) / 0.20
    keep = m.get('test7_edge_keep_rate_after_prune', np.nan)
    if not np.isnan(keep):
        pen += abs(keep - 0.30) / 0.30
    nz = m.get('test7_frac_edges_nonzero', np.nan)
    if not np.isnan(nz):
        pen += 0.5 * abs(nz - 0.5)
    return float(pen)


def sample_params(rng, prune_percentile, normK_mode="on"):
    def U(a,b): return float(rng.uniform(a,b))
    def C(choices, p=None):
        i = rng.choice(len(choices), p=p)
        return choices[int(i)]
    if normK_mode == "search":
        normalize_K = bool(rng.choice([True, False], p=[0.8, 0.2]))
    elif normK_mode == "off":
        normalize_K = False
    else:
        normalize_K = True
    beta_choices = [0.06, 0.07, 0.08, 0.10]
    beta_probs   = [0.10, 0.25, 0.45, 0.20]
    return CCCParams(
        sigma_sim    = U(1.0, 2.0),
        q_robust     = U(0.58, 0.70),
        zcap         = C([2.0, 2.5, 3.0], p=[0.25, 0.5, 0.25]),
        gamma_big    = 0.10,
        f_target     = 0.10,
        beta_max     = C(beta_choices, p=beta_probs),
        beta_neg     = 0.0,
        K_half       = U(0.0035, 0.0080),
        R_factor     = U(2.0, 2.4),
        deg_min      = int(C([8,10,12], p=[0.2,0.6,0.2])),
        deg_max      = int(C([16,18,20], p=[0.4,0.4,0.2])),
        Wr_factor    = U(0.30, 0.55),
        k_for_Wr     = int(C([1,3], p=[0.7,0.3])),
        prune_percentile = prune_percentile if prune_percentile is not None else U(0.65, 0.85),
        n_null       = 200,
        pool_size    = 1000,
        random_state = int(rng.integers(0, 10_000)),
        use_directed = True,
        normalize_K  = normalize_K,
    )


def refine_params(base):
    return CCCParams(
        sigma_sim    = base.sigma_sim,
        q_robust     = base.q_robust,
        zcap         = base.zcap,
        gamma_big    = getattr(base, 'gamma_big', 0.10),
        f_target     = getattr(base, 'f_target', 0.10),
        beta_max     = base.beta_max,
        beta_neg     = base.beta_neg,
        K_half       = base.K_half,
        R_factor     = base.R_factor,
        deg_min      = base.deg_min,
        deg_max      = base.deg_max,
        Wr_factor    = base.Wr_factor,
        k_for_Wr     = base.k_for_Wr,
        prune_percentile = base.prune_percentile,
        n_null       = 1000,
        pool_size    = 1200,
        random_state = base.random_state,
        use_directed = True,
        normalize_K  = base.normalize_K,
    )


def run_search(coords_csv, expr_path, lr_paths, W_his_npy, tf_csv=None,
               aniso_npz=None, n_fast=40, n_refine=8, seed=0,
               n_pairs_sample=20, out_csv="ccc_tuning_results.csv",
               prune_override=None, normK_mode="on", no_scale=False):
    if aniso_npz is None:
        raise ValueError("aniso npz is required")
    coords, barcodes, xy_keys = load_coords_csv(coords_csv)
    expr_df_raw = load_expr_matrix_csv(expr_path, barcodes)
    expr_df = expr_df_raw if no_scale else scale_expr01(expr_df_raw)
    lr_df = load_lr(lr_paths)
    W_his = np.load(W_his_npy)
    if W_his.shape[0] != coords.shape[0]:
        raise ValueError("W_his rows must equal N")
    tf_series = load_tf_auc(tf_csv, coords_barcodes=barcodes, xy_keys=xy_keys) if tf_csv else None
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_fast):
        hp = sample_params(rng, prune_percentile=prune_override, normK_mode=normK_mode)
        if hp.deg_min >= hp.deg_max:
            hp.deg_min, hp.deg_max = int(min(hp.deg_min, 12)), int(max(hp.deg_max, 16))
        try:
            m = run_ccc_sanity_tests(
                coords, expr_df, lr_df, W_his, tf_series,
                params=hp, edges=None, n_pairs_sample=n_pairs_sample,
                aniso_npz=aniso_npz,
            )
            pen = objective(m)
        except Exception as e:
            m = {"error": str(e)}
            pen = 1e9
        trials.append(dict(mode="aniso", penalty=pen, params=hp, metrics=m))
    trials.sort(key=lambda x: x["penalty"])
    top = trials[:n_refine]
    refined = []
    for rec in top:
        hp2 = refine_params(rec["params"])
        try:
            m2 = run_ccc_sanity_tests(
                coords, expr_df, lr_df, W_his, tf_series,
                params=hp2, edges=None, n_pairs_sample=n_pairs_sample,
                aniso_npz=aniso_npz,
            )
            pen2 = objective(m2)
        except Exception as e:
            m2 = {"error": str(e)}
            pen2 = 1e9
        refined.append(dict(mode="aniso", penalty=pen2, params=hp2, metrics=m2))
    all_recs = refined + trials
    all_recs.sort(key=lambda x: x["penalty"])
    rows = []
    for rec in all_recs:
        p = rec["params"]; m = rec["metrics"] or {}
        conn = m.get("test1_connectivity", {})
        row = dict(
            penalty=rec["penalty"],
            sigma_sim=p.sigma_sim, q_robust=p.q_robust, zcap=p.zcap,
            beta_max=p.beta_max, beta_neg=p.beta_neg,
            gamma_big=getattr(p, 'gamma_big', np.nan), f_target=getattr(p, 'f_target', np.nan),
            K_half=p.K_half, Wr_factor=p.Wr_factor, k_for_Wr=p.k_for_Wr,
            R_factor=p.R_factor, deg_min=p.deg_min, deg_max=p.deg_max,
            prune_percentile=p.prune_percentile, n_null=p.n_null,
            t2_null=m.get("test2_null_pass_rate_mean", np.nan),
            t3_rho=m.get("test3_spearman_dist_ccc_mean", np.nan),
            t4_gate_big=m.get("test4_gate_big_frac", np.nan),
            t4_corr=m.get("test4_corr_noGate_vs_Gate", np.nan),
            t5_aboveK=m.get("test5_hill_frac_aboveK_mean", np.nan),
            t7_keep=m.get("test7_edge_keep_rate_after_prune", np.nan),
            t7_frac_nz=m.get("test7_frac_edges_nonzero", np.nan),
            mean_deg=conn.get("mean_degree", np.nan),
            n_comp=conn.get("n_components", np.nan),
            aux_Wr=m.get("aux_Wr", np.nan),
            aux_R_contact=m.get("aux_R_contact", np.nan),
            aux_beta_plus=m.get("aux_beta_plus", np.nan),
            normalize_K=p.normalize_K,
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    best = all_recs[0]
    best_bundle = {
        "penalty": float(best["penalty"]),
        "params": best["params"].__dict__,
        "metrics_keys": list((best["metrics"] or {}).keys()),
    }
    with open(out_csv.replace('.csv', '_best.json'), 'w') as f:
        json.dump(best_bundle, f, indent=2)
