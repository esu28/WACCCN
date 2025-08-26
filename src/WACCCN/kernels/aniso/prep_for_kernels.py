import os, re, glob, numpy as np, pandas as pd

def find_all(folder, pattern):
    return sorted(glob.glob(os.path.join(folder, pattern)))

def find_first(folder, patterns):
    for pat in patterns:
        hits = find_all(folder, pat)
        if hits:
            return hits[0]
    return None

def load_coords_auto(folder, override=None):
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

def load_mask_auto(folder, override=None):
    if override:
        return np.load(override).astype(bool)
    p = find_first(folder, ["mask*.npy", "*_bool.npy", "tissue_mask*.npy"])
    return np.load(p).astype(bool) if p else None

def ensure_equal_hw(arrs, name):
    H, W = np.asarray(arrs[0]).shape[:2]
    for a in arrs[1:]:
        if np.asarray(a).shape[:2] != (H, W):
            raise ValueError(f"{name}: size mismatch")
    return H, W

def pack_wv2(folder, coords=None, mask=None):
    rx = re.compile(r"^L(?P<L>\d+)_B(?P<B>[A-Za-z0-9]{2})_.*\.npy$")
    files = [f for f in find_all(folder, "L*_B??_*.npy") if "B00" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No HP files like 'L*_B??_*.npy' (excluding B00) in {folder}")

    by_level = {}
    for fp in files:
        m = rx.match(os.path.basename(fp))
        if not m:
            continue
        L = int(m.group("L"))
        Btag = m.group("B")
        by_level.setdefault(L, []).append((Btag, fp))

    if 1 not in by_level:
        raise FileNotFoundError("Missing level-1 HP bands (L1_B??_*.npy).")
    have_L2 = 2 in by_level

    def classify(tag, fp):
        name = os.path.basename(fp).lower()
        if "lh" in name: return "LH"
        if "hl" in name: return "HL"
        if "hh" in name: return "HH"
        tag = tag.upper()
        if tag == "01": return "LH"
        if tag == "10": return "HL"
        if tag == "11": return "HH"
        return None

    def pick_level(level_items, level_name):
        got = {"LH":None, "HL":None, "HH":None}
        for tag, fp in sorted(level_items, key=lambda t: t[1]):
            cls = classify(tag, fp)
            if cls and got[cls] is None:
                got[cls] = np.load(fp)
        missing = [k for k,v in got.items() if v is None]
        if missing:
            raise FileNotFoundError(
                f"{level_name}: need LH/HL/HH; missing {missing}. "
                "Include 'LH/HL/HH' in names or use numeric B01/B10/B11."
            )
        ensure_equal_hw([got["LH"], got["HL"], got["HH"]], f"wv2 {level_name}")
        return got["LH"], got["HL"], got["HH"]

    LH1, HL1, HH1 = pick_level(by_level[1], "L1")
    if have_L2:
        LH2, HL2, HH2 = pick_level(by_level[2], "L2")
    else:
        LH2, HL2, HH2 = LH1.copy(), HL1.copy(), HH1.copy()

    bundle = {"family":"wv2","LH1":LH1,"LH2":LH2,"HL1":HL1,"HL2":HL2,"HH1":HH1,"HH2":HH2}
    c = load_coords_auto(folder, coords)
    if c is not None: bundle["coords"] = c
    m = load_mask_auto(folder, mask)
    if m is not None: bundle["mask"] = m
    return bundle

def pack_wv4(folder, meta_path=None, coords=None, mask=None):
    files_L1 = [f for f in find_all(folder, "L1_B??_*.npy") if "B00" not in os.path.basename(f)]
    if not files_L1:
        raise FileNotFoundError(f"No L1 HP files (L1_B??_*.npy) in {folder}")
    files_L1 = sorted(files_L1)
    bands_l1 = np.stack([np.load(f) for f in files_L1], axis=0)

    files_L2 = [f for f in find_all(folder, "L2_B??_*.npy") if "B00" not in os.path.basename(f)]
    bands_l2 = None
    if files_L2 and len(files_L2) == len(files_L1):
        files_L2 = sorted(files_L2)
        bands_l2 = np.stack([np.load(f) for f in files_L2], axis=0)

    if meta_path:
        meta_fp = meta_path
    else:
        meta_fp = None
        for p in find_all(folder, "*.npz"):
            try:
                z = np.load(p)
                if ("theta" in z.files) and ("kappa" in z.files):
                    meta_fp = p; break
            except Exception:
                pass
        if not meta_fp:
            raise FileNotFoundError("No meta npz with 'theta' and 'kappa' found.")
    meta = np.load(meta_fp)
    theta = np.asarray(meta["theta"], float)
    kappa = np.asarray(meta["kappa"], float)
    if theta.shape[0] != bands_l1.shape[0] or kappa.shape[0] != bands_l1.shape[0]:
        raise ValueError("theta/kappa length must equal number of L1 bands (sorted).")

    bundle = {"family":"wv4","bands_l1":bands_l1,"theta":theta,"kappa":kappa}
    if bands_l2 is not None: bundle["bands_l2"] = bands_l2
    c = load_coords_auto(folder, coords)
    if c is not None: bundle["coords"] = c
    m = load_mask_auto(folder, mask)
    if m is not None: bundle["mask"] = m
    return bundle

def pack_dtcwt(folder, coords=None, mask=None):
    rx = re.compile(r"L(?P<L>\d+)_ang(?P<A>-?\d+)_upsampled\.npy$")
    all_files = find_all(folder, "*L*_ang*_upsampled.npy")
    by_level = {}
    for fp in all_files:
        m = rx.search(os.path.basename(fp))
        if not m:
            continue
        L = int(m.group("L")); A = int(m.group("A"))
        by_level.setdefault(L, set()).add(A)
    if not by_level:
        raise FileNotFoundError(f"No DTCWT subbands '*L*_ang*_upsampled.npy' in {folder}")

    levels = sorted(by_level.keys())
    common = set.intersection(*[by_level[L] for L in levels]) if len(levels) > 1 else by_level[levels[0]]
    angles = sorted(common) if common else sorted(by_level[levels[0]])

    def file_for(L, A):
        for pat in (f"*L{L}_ang{A}_upsampled.npy", f"L{L}_ang{A}_upsampled.npy"):
            hits = find_all(folder, pat)
            if hits: return hits[0]
        return None

    maps = {}
    H = W = None
    for L in levels:
        arrs = []
        for A in angles:
            fp = file_for(L, A)
            if fp is None:
                raise FileNotFoundError(f"Missing DTCWT subband for L{L}, ang{A}")
            X = np.load(fp)
            if not np.iscomplexobj(X) or X.ndim not in (2,3) or (X.ndim==3 and X.shape[2] not in (1,3)):
                raise ValueError(f"Expected complex (H,W) or (H,W,3) at {fp}, got {X.dtype}, shape {X.shape}")
            if H is None: H, W = X.shape[:2]
            elif (H, W) != X.shape[:2]: raise ValueError(f"Size mismatch at {fp}: {X.shape[:2]} vs {(H,W)}")
            arrs.append(X)
        maps[L] = np.stack(arrs, axis=0)

    bundle = {"family":"dtcwt", "levels":np.array(levels, int), "angles":np.array(angles, int)}
    for L in levels:
        bundle[f"dtcwt_L{L}"] = maps[L]
    c = load_coords_auto(folder, coords)
    if c is not None: bundle["coords"] = c
    m = load_mask_auto(folder, mask)
    if m is not None: bundle["mask"] = m
    return bundle
