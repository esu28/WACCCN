# Dominant orientation (theta) and concentration (kappa) for separable 2D subbands.

import numpy as np

EPS = 1e-12

def dominant_orientation_2d_separable(h_filt, v_filt, n_fft=4096,
                                      rho_min=0.2*np.pi, rho_max=np.pi):
    Hx = np.fft.fftshift(np.fft.fft(h_filt, n=n_fft))
    Hy = np.fft.fftshift(np.fft.fft(v_filt, n=n_fft))
    H2d = np.outer(Hy, Hx)
    mag2 = np.abs(H2d)**2

    wx = np.linspace(-np.pi, np.pi, n_fft, endpoint=False)
    wy = np.linspace(-np.pi, np.pi, n_fft, endpoint=False)
    WX, WY = np.meshgrid(wx, wy, indexing="xy")

    phi = np.arctan2(WY, WX)
    rho = np.hypot(WX, WY)

    band = (rho >= rho_min) & (rho <= rho_max)
    wts = mag2 * band

    num = np.sum(wts * np.exp(1j * 2 * phi))
    den = np.sum(wts) + EPS
    v = num / den

    theta = 0.5 * np.angle(v)
    if theta < 0: theta += np.pi
    kappa = np.abs(v)
    return float(theta), float(kappa)

def _build_names(M, N, skip_ll=True):
    names, pairs = [], []
    for i in range(M):
        for j in range(N):
            if skip_ll and i == 0 and j == 0: continue
            names.append(f"B{i}{j}")
            pairs.append((i, j))
    return names, pairs

def compute_bank_theta_kappa(hbank, vbank=None, n_fft=4096,
                             rho_min=0.2*np.pi, rho_max=np.pi, skip_ll=True):
    if vbank is None: vbank = hbank
    if hbank.ndim != 2 or vbank.ndim != 2:
        raise ValueError("banks must be 2D (num_filters, taps)")
    M, N = hbank.shape[0], vbank.shape[0]
    names, pairs = _build_names(M, N, skip_ll=skip_ll)
    thetas, kappas = [], []
    for i, j in pairs:
        th, kp = dominant_orientation_2d_separable(
            hbank[i], vbank[j], n_fft=n_fft, rho_min=rho_min, rho_max=rho_max
        )
        thetas.append(np.degrees(th))
        kappas.append(kp)
    return names, np.asarray(thetas, float), np.asarray(kappas, float)
