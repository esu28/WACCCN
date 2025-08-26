#plotting helpers

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["show_wavelet_decomposition", "show_wavelet_grid"]


def _norm01(x):
    x = np.asarray(x)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _to_display(sub):  # sub: (h,w) or (C,h,w)
    if sub.ndim == 2:
        return _norm01(sub)
    # C,h,w -> image
    C = sub.shape[0]
    if C == 3:
        img = np.moveaxis(sub, 0, -1)
    elif C == 1:
        img = sub[0]
    else:
        img = np.mean(sub, axis=0)  # robust default for C!=1,3
    return _norm01(img)


def show_wavelet_decomposition(coeffs_list):
    # Each subband in its own figure
    first = coeffs_list[0]
    M = first.shape[1] if first.ndim == 5 else first.shape[0]

    for lvl, coeff in enumerate(coeffs_list, start=1):
        for i in range(M):
            for j in range(M):
                plt.figure()
                if coeff.ndim == 5:  # (C,M,M,h,w)
                    sub = coeff[:, i, j, :, :]
                else:                # (M,M,h,w)
                    sub = coeff[i, j]
                img = _to_display(sub)
                if img.ndim == 2:
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)
                label = "LL" if (i == 0 and j == 0) else f"B{i}{j}"
                plt.title(f"{label} L{lvl}")
                plt.axis("off")
    plt.show()


def show_wavelet_grid(coeffs_list):
    # Grid: rows = levels, cols = M*M subbands
    first = coeffs_list[0]
    M = first.shape[1] if first.ndim == 5 else first.shape[0]
    L = len(coeffs_list)
    ncols = M * M
    fig, axes = plt.subplots(nrows=L, ncols=ncols, figsize=(ncols * 2.2, L * 2.2), squeeze=False)

    for lvl, coeff in enumerate(coeffs_list, start=1):
        for i in range(M):
            for j in range(M):
                ax = axes[lvl - 1, i * M + j]
                if coeff.ndim == 5:
                    sub = coeff[:, i, j, :, :]
                else:
                    sub = coeff[i, j]
                img = _to_display(sub)
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
                label = "LL" if (i == 0 and j == 0) else f"B{i}{j}"
                ax.set_title(f"{label} L{lvl}")
                ax.axis("off")

    plt.tight_layout()
    plt.show()
