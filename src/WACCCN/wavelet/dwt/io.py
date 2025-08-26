#load data

import os, math
import numpy as np       
import pandas as pd        
from PIL import Image       

__all__ = ["load_image_or_csv"]


def load_image_or_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        with np.load(path) as z:
            key = "arr_0" if "arr_0" in z.files else z.files[0]
            arr = z[key].astype(np.float32)
    elif ext == ".npy":
        arr = np.load(path).astype(np.float32)
    elif ext == ".csv":
        df = pd.read_csv(path, index_col=0)
        arr = df.values.astype(np.float32)
    else:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32)

    # Auto-reshape 1D vectors to square if possible
    if arr.ndim == 1:
        n = arr.size
        s = int(math.isqrt(n))
        if s * s == n:
            arr = arr.reshape(s, s)
        else:
            raise ValueError(
                f"Loaded 1D array of length {n}; cannot reshape to square."
            )
    return arr
