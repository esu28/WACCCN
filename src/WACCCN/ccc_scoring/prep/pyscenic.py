!pip install numpy==1.23.5
!pip install llvmlite==0.40.1 numba==0.57.1
!pip install arboreto

!pip install pyscenic==0.11.2 --no-deps
# or !pip install pyscenic==0.11.2

import pandas as pd

#transpose the expr matrix first
expr = pd.read_csv("expr.csv", index_col=0)
exprT = expr.T
exprT.to_csv("exprT.csv")

#GRN
from multiprocessing import Pool, cpu_count
import pandas as pd

from arboreto.algo import _prepare_input
from arboreto.core import infer_partial_network, to_tf_matrix, target_gene_indices, SGBM_KWARGS

expr_path = "exprT.csv"
expr_df = pd.read_csv(expr_path, index_col=0)

gene_names = expr_df.columns


with open("tf_list.txt") as f:
    tf_list = [line.strip() for line in f]

expression_matrix, gene_names, tf_list = _prepare_input(expr_df, gene_names, tf_list)
tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_list)

def run_partial_net(i):
    target_gene = gene_names[i]
    target_expression = expression_matrix[:, i]
    return infer_partial_network(
        regressor_type='GBM',
        regressor_kwargs=SGBM_KWARGS,
        tf_matrix=tf_matrix,
        tf_matrix_gene_names=tf_matrix_gene_names,
        target_gene_name=target_gene,
        target_gene_expression=target_expression,
        include_meta=False,
        early_stop_window_length=5,
        seed=0
    )


with Pool(cpu_count()) as pool:
    adj_list = pool.map(run_partial_net, target_gene_indices(gene_names, 'all'))

adj_df = pd.concat(adj_list).sort_values(by="importance", ascending=False)

adj_df.to_csv(f"{base}/adj.tsv", sep="\t", index=False)

#ctx
!pyscenic ctx \
  "adj.tsv" \
  "hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather" \
  --annotations_fname "motifs-v9-nr.hgnc-m0.001-o0.0.tbl.txt" \
  --expression_mtx "exprT.csv" \
  --mode custom_multiprocessing \
  --mask_dropouts \
  -o "regulons.csv"

#aucell
!pyscenic aucell \
  "exprT.csv" \
  "egulons.csv" \
  --num_workers 1 \
  -o "tf_activity.csv"


