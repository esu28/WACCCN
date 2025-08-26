#Core DWT helpers: matrix build, padding/cropping, single- and multi-level forward/inverse. 


import numpy as np 

__all__ = [
    "dwt_matrix",
    "pad_for_levels",
    "crop_subband",
    "get_max_levels",
    "tDdwt",
    "multi_level_dwt",
    "inverse_tDdwt",
    "multi_level_idwt",
]


def dwt_matrix(filter_bank, rows_per_filter):
    # Polyphase-style analysis matrix H (downsample-by-M), N = M*rows_per_filter
    M, K = filter_bank.shape
    N = M * rows_per_filter
    H = np.zeros((N, N), dtype=filter_bank.dtype)
    row = 0
    for fb in filter_bank:
        offset = 0
        for _ in range(rows_per_filter):
            for k, coeff in enumerate(fb):
                H[row, (offset + k) % N] = coeff
            offset += M
            row += 1
    return H


def pad_for_levels(image, M, levels, mode="reflect"):
    # Pad so both dims divisible by M**levels; return padded and (top,bottom,left,right)
    Hdim, Wdim = image.shape[:2]
    factor = M ** levels
    pad_h = (-Hdim) % factor
    pad_w = (-Wdim) % factor
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    pad_width = ((top, bottom), (left, right)) + ((0, 0),) * (image.ndim - 2)
    padded = np.pad(image, pad_width, mode=mode)
    return padded, (top, bottom, left, right)


def crop_subband(sub, pads, level, M):
    # Crop a subband (…×H×W) back to original support at a given level
    top_pad, bottom_pad, left_pad, right_pad = pads
    ch = (top_pad + bottom_pad) // (M ** level)
    cw = (left_pad + right_pad) // (M ** level)
    t = ch // 2
    b = ch - t
    l = cw // 2
    r = cw - l
    return sub[..., t: sub.shape[-2] - b, l: sub.shape[-1] - r]


def get_max_levels(shape, M):
    # Largest L with dims divisible by M**L
    Hdim, Wdim = shape[:2]
    L = 0
    while Hdim % (M ** (L + 1)) == 0 and Wdim % (M ** (L + 1)) == 0:
        L += 1
    return L


def tDdwt(wavelet, image):
    # Single-level separable DWT via analysis matrices.
    # Returns:
    #   single-channel -> (M, M, h, w)
    #   multi-channel  -> (C, M, M, h, w)
    if image.ndim == 3:  # C-last -> stack C-first
        chans = [tDdwt(wavelet, image[..., c]) for c in range(image.shape[2])]
        return np.stack(chans, axis=0)

    M = wavelet.shape[0]
    Hdim, Wdim = image.shape
    assert Hdim % M == 0 and Wdim % M == 0, "Dims must be divisible by M"
    h = Hdim // M
    w = Wdim // M
    Hy = dwt_matrix(wavelet, h)
    Hx = dwt_matrix(wavelet, w)
    T = Hy @ image @ Hx.T
    return T.reshape(M, h, M, w).transpose(0, 2, 1, 3)


def multi_level_dwt(image, wavelet, levels, pad_mode="reflect"):
    # Mallat scheme: transform LL recursively; return [level1, level2, ...], pads
    M = wavelet.shape[0]
    padded, pads = pad_for_levels(image, M, levels, mode=pad_mode)
    coeffs_list = []
    cur = padded
    for _ in range(levels):
        coeffs = tDdwt(wavelet, cur)
        coeffs_list.append(coeffs)
        if cur.ndim == 3:
            cur = np.stack([coeffs[c, 0, 0] for c in range(coeffs.shape[0])], axis=-1)
        else:
            cur = coeffs[0, 0]
    return coeffs_list, pads


def inverse_tDdwt(coeffs, wavelet):
    # Inverse of single-level tDdwt (channel-aware)
    if coeffs.ndim == 5:  # (C,M,M,h,w)
        chans = [inverse_tDdwt(coeffs[c], wavelet) for c in range(coeffs.shape[0])]
        return np.stack(chans, axis=-1)

    M1, M2, h, w = coeffs.shape
    assert M1 == M2, "Non-square subband grid"
    arr = coeffs.transpose(0, 2, 1, 3)  # (M,h,M,w)
    Hy = dwt_matrix(wavelet, h)
    Hx = dwt_matrix(wavelet, w)
    return Hy.T @ arr.reshape(M1 * h, M1 * w) @ Hx  # (M*h, M*w)


def multi_level_idwt(coeffs_list, wavelet, pads):
    # Coarsest -> finest: set LL of each finer level to current image, then invert
    img = inverse_tDdwt(coeffs_list[-1], wavelet)
    for level in range(len(coeffs_list) - 2, -1, -1):
        coeffs = coeffs_list[level].copy()
        if coeffs.ndim == 5:
            C = coeffs.shape[0]
            for c in range(C):
                coeffs[c, 0, 0] = img[..., c]
        else:
            coeffs[0, 0] = img
        img = inverse_tDdwt(coeffs, wavelet)

    top, bottom, left, right = pads
    if img.ndim == 3:
        return img[top: img.shape[0] - bottom, left: img.shape[1] - right, :]
    return img[top: img.shape[0] - bottom, left: img.shape[1] - right]
