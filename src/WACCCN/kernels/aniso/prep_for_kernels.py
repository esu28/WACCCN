#!/usr/bin/env python3
import os, re, glob, numpy as np, pandas as pd

def find_all(folder, pattern):
    return sorted(glob.glob(os.path.join(folder, pattern)))

def find_first(folder, patterns):
    for pat in patterns:
        hits = find_all(folder, pat)
        if hits:
            return hits[0]
    return None

def load_coords_auto(folder, override):
    def _load_coords_path(p):
        if p.endswith(".npy"):
            arr = np.load(p)
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] in (2,3):
                return arr
            raise ValueError(f"{p}: expected (N,2|3), got {arr.shape}")
        df = pd.read_csv(p)
        x_candidates = ["x","x_um","X","coord_x","imagecol","x_pixel","xpix","pos_x","xc","col"]
        y_candidates = ["y","y_um","Y","coord_y","imagerow","y_pixel","ypix","pos_y","yc","row"]
        xcol = next((c for c in x_candidates if c in df.columns), None)
        ycol = next((c for c in y_candidates if c in df.columns), None)
        if xcol and ycol:
            return df[[xcol, ycol]].to_numpy(dtype=float)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            raise ValueError(f"{p}: could not find two numeric x,y columns")
        return num.iloc[:, -2:].to_numpy(dtype=float)
    if override:
        return _load_coords_path(override)
    candidates = (find_all(folder, "coords*.npy") +
                  find_all(folder, "coordinates*.npy") +
                  find_all(folder, "coords*.csv") +
                  find_all(folder, "coordinates*.csv"))
    for p in candidates:
        try:
            return _load_coords_path(p)
        except Exception:
            continue
    return None

def load_mask_auto(folder, override):
    if override:
        return np.load(override).astype(bool)
    p = find_first(folder, ["mask*.npy", "*_bool.npy", "tissue_mask*.npy"])
    return np.load(p).astype(bool) if p else None

def ensure_equal_hw(arrs, name):
    H, W = np.asarray(arrs[0]).shape[:2]
    for a in arrs[1:]:
        if np.asarray(a).shape[:2] != (H, W):
            raise ValueError(f"{name}: size mismatch; expected (H,W)={H,W}")
    return H, W

def pack_wv2(folder, coords=None, mask=None):
    rx = re.compile(r"^L(?P<L>\d+)_B(?P<B>[A-Za-z0-9]{2})_.*\.npy$")
    files = [f for f in find_all(folder, "L*_B??_*.npy") if "B00" not in os.path.basename(f)]
    by_level = {}
    for fp in files:
        m = rx.match(os.path.basename(fp))
        if not m: continue
        L = int(m.group("L")); Btag = m.group("B")
        by_level.setdefault(L, []).append((Btag, fp))
    def classify(tag, fp):
        name = os.path.basename(fp).lower()
        if "lh" in name: return "LH"
        if "hl" in name: return "HL"
        if "hh" in name: return "HH"
        if tag.upper() == "01": return "LH"
        if tag.upper() == "10": return "HL"
        if tag.upper() == "11": return "HH"
        return None
    def pick_level(level_items, level_name):
        got = {"LH":None, "HL":None, "HH":None}
        for tag, fp in sorted(level_items, key=lambda t: t[1]):
            cls = classify(tag, fp)
            if cls and got[cls] is None: got[cls] = np.load(fp)
        ensure_equal_hw([got["LH"], got["HL"], got["HH"]], f"wv2 {level_name}")
        return got["LH"], got["HL"], got["HH"]
    LH1, HL1, HH1 = pick_level(by_level[1], "L1")
    LH2, HL2, HH2 = pick_level(by_level[2], "L2") if 2 in by_level else (LH1, HL1, HH1)
    bundle = {"family":"wv2","LH1":LH1,"LH2":LH2,"HL1":HL1,"HL2":HL2,"HH1":HH1,"HH2":HH2}
    c = load_coords_auto(folder, coords); m = load_mask_auto(folder, mask)
    if c is not None: bundle["coords"] = c
    if m is not None: bundle["mask"] = m
    return bundle

def pack_wv4(folder, meta_path=None, coords=None, mask=None):
    files_L1 = sorted([f for f in find_all(folder, "L1_B??_*.npy") if "B00" not in os.path.basename(f)])
    bands_l1 = np.stack([np.load(f) for f in files_L1], axis=0)
    files_L2 = sorted([f for f in find_all(folder, "L2_B??_*.npy") if "B00" not in os.path.basename(f)])
    bands_l2 = np.stack([np.load(f) for f in files_L2], axis=0) if files_L2 else None
    meta_fp = meta_path
    if not meta_fp:
        for p in find_all(folder, "*.npz"):
            z = np.load(p)
            if "theta" in z.files and "kappa" in z.files: meta_fp = p; break
    meta = np.load(meta_fp)
    bundle = {"family":"wv4","bands_l1":bands_l1,"theta":meta["theta"],"kappa":meta["kappa"]}
    if bands_l2 is not None: bundle["bands_l2"] = bands_l2
    c = load_coords_auto(folder, coords); m = load_mask_auto(folder, mask)
    if c is not None: bundle["coords"] = c
    if m is not None: bundle["mask"] = m
    return bundle

def pack_dtcwt(folder, coords=None, mask=None):
    rx = re.compile(r"L(?P<L>\d+)_ang(?P<A>-?\d+)_upsampled\.npy$")
    all_files = find_all(folder, "*L*_ang*_upsampled.npy")
    by_level = {}
    for fp in all_files:
        m = rx.search(os.path.basename(fp))
        if not m: continue
        L = int(m.group("L")); A = int(m.group("A"))
        by_level.setdefault(L, []).add(A)
    levels = sorted(by_level.keys())
    common = set.intersection(*[by_level[L] for L in levels]) if len(levels)>1 else by_level[levels[0]]
    angles = sorted(common)
    maps = {}
    for L in levels:
        arrs = []
        for A in angles:
            fp = glob.glob(os.path.join(folder, f"*L{L}_ang{A}_upsampled.npy"))[0]
            arrs.append(np.load(fp))
        maps[L] = np.stack(arrs, axis=0)
    bundle = {"family":"dtcwt","levels":np.array(levels),"angles":np.array(angles)}
    for L in levels: bundle[f"dtcwt_L{L}"] = maps[L]
    c = load_coords_auto(folder, coords); m = load_mask_auto(folder, mask)
    if c is not None: bundle["coords"] = c
    if m is not None: bundle["mask"] = m
    return bundle
