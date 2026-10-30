"""Feature extraction (numpy/scipy only): MFCC, spectral, pitch, jitter, shimmer, pause cues."""
import numpy as np
from scipy.fft import dct

SR = 16000
N_FFT = 512


def _mel_fb(n_mels=26):
    h2m = lambda f: 2595 * np.log10(1 + f / 700)
    m2h = lambda m: 700 * (10 ** (m / 2595) - 1)
    bins = np.floor((N_FFT + 1) * m2h(np.linspace(0, h2m(SR / 2), n_mels + 2)) / SR).astype(int)
    fb = np.zeros((n_mels, N_FFT // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        fb[i, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[i, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = _mel_fb()
FREQS = np.linspace(0, SR / 2, N_FFT // 2 + 1)


def frames(x, n=N_FFT, h=160):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, h)[:, None] + np.arange(n)[None]
    return x[idx]


def extract(x):
    """Return (feature_vector, meta) for ~1 s of 16 kHz mono audio, or None if silent."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        return None
    if np.sqrt(np.mean(x ** 2)) < 0.003:
        return None
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    F = frames(x)
    rms = np.sqrt((F ** 2).mean(1) + 1e-12)
    act = rms > 0.1 * rms.max()
    if act.sum() < 3:
        act[:] = True
    P = np.abs(np.fft.rfft(F * np.hanning(N_FFT), N_FFT)) ** 2 + 1e-10
    mf = dct(np.log(P @ FB.T + 1e-10), type=2, axis=1, norm="ortho")[:, 1:14]
    mfa = mf[act] if act.sum() > 3 else mf
    flat = (np.exp(np.log(P).mean(1)) / P.mean(1))[act].mean()
    cent = ((P * FREQS).sum(1) / P.sum(1))[act]
    hf4 = (P[:, 128:].sum(1) / P.sum(1))[act].mean()
    hf6 = (P[:, 192:].sum(1) / P.sum(1))[act].mean()
    zcr = (np.diff(np.sign(F), axis=1) != 0).mean()

    # pitch by autocorrelation
    W = F * np.hanning(N_FFT)
    ac = np.fft.irfft(np.abs(np.fft.rfft(W, 1024)) ** 2, 1024)[:, :512]
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = SR // 400, SR // 70
    seg = ac[:, lo:hi]
    lag = seg.argmax(1) + lo
    voiced = (seg.max(1) > 0.4) & (rms > 0.1 * rms.max())
    f0 = SR / lag
    vr, jit, f0std, shim = voiced.mean(), 0.0, 0.0, 0.0
    f0m = 0.0
    if voiced.sum() >= 5:
        f = f0[voiced]
        f0m = float(f.mean())
        jit = float(np.abs(np.diff(f)).mean() / f.mean())
        f0std = float(f.std() / f.mean())
        a = rms[voiced]
        shim = float(np.abs(np.diff(a)).mean() / a.mean())
    vec = np.concatenate([
        mfa.mean(0), mfa.std(0),
        [flat, cent.mean() / 1000, cent.std() / 1000, hf4, hf6, zcr,
         (~act).mean(), rms.std() / rms.mean(), vr, f0std, jit, shim],
    ]).astype(np.float32)
    return vec, {"f0": f0m, "jitter": jit, "voiced": float(vr)}
