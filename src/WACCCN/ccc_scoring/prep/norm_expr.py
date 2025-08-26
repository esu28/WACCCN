#!/usr/bin/env python3
# normalize expression matrix (winsorize + min–max [0,1])
import pandas as pd
import numpy as np

INPUT_PATH  = ""
OUTPUT_PATH = ""

df_all = pd.read_csv(INPUT_PATH)

first = df_all.columns[0]
if not np.issubdtype(df_all.dtypes[first], np.number):
    barcodes = df_all[first].astype(str)
    df = df_all.drop(columns=[first])
else:
    barcodes = None
    df = df_all

upper = df.quantile(0.99)  # winsorize at 99%
df = df.clip(upper=upper, axis=1)

gmin = df.min()
gmax = df.max()
scale = (gmax - gmin).replace(0, 1)  # avoid div by 0
df_norm = (df - gmin) / scale
df_norm = df_norm.fillna(0.0).clip(0,1)

if barcodes is not None:
    df_out = pd.concat([barcodes.rename(first), df_norm], axis=1)
else:
    df_out = df_norm

df_out.to_csv(OUTPUT_PATH, index=False)

print(f"Normalized expression saved to {OUTPUT_PATH}")
