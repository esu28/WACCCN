import numpy as np  

ANGLES_DEG = [15, 45, 75, 105, 135, 165]  # band order (mod 180)

def ensure_dtcwt():
    try:
        from dtcwt.numpy import Transform2d  
    except Exception:
        from dtcwt import Transform2d
    return Transform2d

def normalize_rgb(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):          # HWC
        if x.shape[-1] == 1: x = x[..., 0][..., None]
        if x.shape[-1] == 4: x = x[..., :3]
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):         # CHW
        c = x.shape[0]
        if c == 1: x = x[0][..., None]
        else:      x = np.moveaxis(x[:3], 0, -1)
    else:
        raise ValueError("expected (H,W,3) or (3,H,W)")
    x = x.astype(np.float32, copy=False)
    if float(np.nanmax(x)) > 1.5: x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def prepad_pow2L(img, levels):
    m = 2 ** int(levels)
    H, W = img.shape[:2]
    ph = (-H) % m
    pw = (-W) % m
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return img.astype(np.float32, copy=False), int(ph), int(pw), m

def max_levels(H, W):
    return max(1, int(np.floor(np.log2(min(H, W))) - 1))

def run_dtcwt_rgb(img_rgb, levels):
    Transform2d = ensure_dtcwt()
    H, W, _ = img_rgb.shape
    Lreq = int(levels)
    Lcap = max_levels(H, W)
    L = min(Lreq, Lcap)

    T = Transform2d()
    Yh_list = [None] * L  # each -> (h_l, w_l, 6, 3) complex64

    for c in range(3):
        res = T.forward(img_rgb[..., c], nlevels=L, include_scale=False)
        Yh = res.highpasses if hasattr(res, "highpasses") else res[1]
        for l in range(L):
            band = np.asarray(Yh[l]).astype(np.complex64)  # (h_l, w_l, 6)
            if Yh_list[l] is None:
                h, w, _ = band.shape
                Yh_list[l] = np.zeros((h, w, 6, 3), dtype=np.complex64)
            Yh_list[l][..., c] = band
    return Yh_list, L
