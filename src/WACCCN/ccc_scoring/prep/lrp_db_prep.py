import re
from pathlib import Path
import pandas as pd

def normalize_gene_names(series):
    s = series.astype(str).str.strip().str.upper()
    return s.str.replace(r"\.\d+$", "", regex=True)

def load_expr_genes(expr_csv):
    df = pd.read_csv(expr_csv)
    if df.shape[1] < 1:
        raise ValueError("Expression CSV must have first column = gene names.")
    return set(normalize_gene_names(df.iloc[:, 0]).values)

def _map_lr_columns_with_headers(df):
    lc = [str(c).strip().lower() for c in df.columns]
    want = ["ligand", "receptor", "pathway", "ligand_annotation", "tf"]
    col_map = {}
    for k in want:
        if k in lc:
            col_map[k] = df.columns[lc.index(k)]
    if ("ligand" not in col_map) or ("receptor" not in col_map):
        return {}
    return col_map

def load_lr_db(lr_csv):
    raw = pd.read_csv(lr_csv)
    col_map = _map_lr_columns_with_headers(raw)
    if col_map:
        lr = pd.DataFrame({
            "ligand": raw[col_map["ligand"]],
            "receptor": raw[col_map["receptor"]],
            "pathway": raw[col_map["pathway"]] if "pathway" in col_map else "",
            "ligand_annotation": raw[col_map["ligand_annotation"]] if "ligand_annotation" in col_map else "",
            "TF": raw[col_map["tf"]] if "tf" in col_map else "",
        })
    else:
        if raw.shape[1] < 5:
            raise ValueError(f"LR file must have ≥5 columns or labeled ligand/receptor. Got {raw.shape[1]}.")
        lr = raw.iloc[:, -5:].copy()
        lr.columns = ["ligand", "receptor", "pathway", "ligand_annotation", "TF"]

    lr["ligand_raw"] = lr["ligand"]
    lr["receptor_raw"] = lr["receptor"]

    lr["ligand"] = normalize_gene_names(lr["ligand"])
    lr["receptor"] = normalize_gene_names(lr["receptor"])
    for c in ["pathway", "ligand_annotation", "TF"]:
        lr[c] = lr[c].fillna("").astype(str).str.strip()
    return lr

def remove_complexes(lr):
    pattern = re.compile(r"(?:^COMPLEX:)|[_/+\s;]")
    mask = lr["ligand"].str.contains(pattern) | lr["receptor"].str.contains(pattern)
    return lr[~mask].copy(), int(mask.sum())

def filter_lrp(expr_csv, lr_csv, out_csv):
    genes = load_expr_genes(expr_csv)
    lr = load_lr_db(lr_csv)

    n_input = lr.shape[0]
    nL_in = lr["ligand"].nunique()
    nR_in = lr["receptor"].nunique()

    lr_nocomp, n_complex = remove_complexes(lr)

    maskL = lr_nocomp["ligand"].isin(genes)
    maskR = lr_nocomp["receptor"].isin(genes)
    kept = lr_nocomp[maskL & maskR].copy()

    kept = kept.drop_duplicates(subset=["ligand", "receptor", "pathway", "ligand_annotation", "TF"]).reset_index(drop=True)
    kept_out = kept[["ligand", "receptor", "pathway", "ligand_annotation", "TF"]]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    kept_out.to_csv(out_csv, index=False)

    missL = sorted(set(lr_nocomp.loc[~maskL, "ligand"].tolist()))[:10]
    missR = sorted(set(lr_nocomp.loc[~maskR, "receptor"].tolist()))[:10]

    report = {
        "n_expr_genes": len(genes),
        "rows_input": n_input,
        "uniq_ligands_in": nL_in,
        "uniq_receptors_in": nR_in,
        "removed_complex_rows": n_complex,
        "rows_kept": kept_out.shape[0],
        "uniq_ligands_kept": kept_out["ligand"].nunique(),
        "uniq_receptors_kept": kept_out["receptor"].nunique(),
        "examples_missing_ligand": missL,
        "examples_missing_receptor": missR,
        "top_pathways": kept_out["pathway"].value_counts().head(15),
        "out_csv": out_csv,
    }
    return report
