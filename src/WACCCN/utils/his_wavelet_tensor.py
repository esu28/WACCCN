import os, re
import numpy as np

def _ext_priority(n):
    return 2 if n.lower().endswith(".npy") else 1 if n.lower().endswith(".npz") else 0

def _discover(base_dir, excludes):
    pat = re.compile(r"L(\d+)_B(\d+)", re.IGNORECASE)
    picked = {}
    for root, _, files in os.walk(base_dir):
        root_tag = pat.search(os.path.basename(root))
        for name in files:
            if not name.lower().endswith((".npy", ".npz")): continue
            path = os.path.join(root, name)
            m = pat.search(name) or root_tag
            if not m: continue
            lvl, band = int(m.group(1)), int(m.group(2))
            if (lvl, band) in excludes: continue
            key = (lvl, band)
            if key not in picked or _ext_priority(name) > _ext_priority(os.path.basename(picked[key])):
                picked[key] = path
    out = [(k[0], k[1], v) for k, v in picked.items()]
    out.sort(key=lambda t: (t[0], t[1]))
    return out

def _load(path, keep_rgb):
    if path.lower().endswith(".npy"):
        a = np.load(path, allow_pickle=False)
    else:
        z = np.load(path, allow_pickle=False)
        key = "arr" if "arr" in z else list(z.keys())[0]
        a = z[key]
    a = np.asarray(a)
    if a.ndim == 2: return a
    if a.ndim == 3: return a if keep_rgb else a.mean(axis=-1)
    raise ValueError(f"bad shape {a.shape} in {path}")

def make_histology_wavelet_tensor(base_dir, out_npy, out_csv, keep_rgb=True, excludes=None):
    if excludes is None: excludes = {(1, 0)}
    cands = _discover(base_dir, excludes)
    if not cands: raise ValueError("no subbands found (need names like L{level}_B{band}).")

    first = _load(cands[0][2], keep_rgb)
    H, W = first.shape[:2]
    stacks = [first[:, :, None] if first.ndim == 2 else first]
    names = []
    if first.ndim == 2:
        names.append(f"L{cands[0][0]}_B{cands[0][1]}")
    else:
        for c in range(first.shape[-1]):
            names.append(f"L{cands[0][0]}_B{cands[0][1]}_c{c}")

    for lvl, band, path in cands[1:]:
        a = _load(path, keep_rgb)
        if a.shape[0] != H or a.shape[1] != W:
            raise ValueError(f"size mismatch {path}: {a.shape[:2]} vs {(H,W)}")
        if a.ndim == 2:
            stacks.append(a[:, :, None])
            names.append(f"L{lvl}_B{band}")
        else:
            stacks.append(a)
            for c in range(a.shape[-1]):
                names.append(f"L{lvl}_B{band}_c{c}")

    X = np.concatenate(stacks, axis=-1)
    eps = 1e-8
    mu = np.nanmean(X, axis=(0,1), keepdims=True)
    sd = np.nanstd(X,  axis=(0,1), keepdims=True)
    Xz = (X - mu) / (sd + eps)
    np.nan_to_num(Xz, copy=False)

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, Xz)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("index,channel\n")
        for i, nm in enumerate(names): f.write(f"{i},{nm}\n")
    return Xz.shape, names
