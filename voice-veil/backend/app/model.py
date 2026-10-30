"""Synthetic training-data generator, dataset builder, training and loading."""
from pathlib import Path
import json
import numpy as np
import joblib
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .features import SR, extract

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "voice_veil.joblib"
META_PATH = ROOT / "models" / "voice_veil.meta.json"
LAST_SOURCE = "synthetic"


def _res(x, f, b):
    r = np.exp(-np.pi * b / SR)
    return lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], x)


def synth(rng, fake, dur=1.0, base=None):
    """Speech-like signal. 'fake' = over-smooth pitch, low jitter/shimmer, few pauses,
    low breath noise and a periodic vocoder-style tone. Ranges overlap on purpose."""
    n = int(dur * SR); t = np.arange(n) / SR
    base = base or rng.uniform(90, 260)
    walk = np.convolve(np.cumsum(rng.normal(0, 1, n)), np.ones(800) / 800, "same")
    walk /= np.abs(walk).max() + 1e-9
    wob = rng.uniform(0.004, 0.04) if fake else rng.uniform(0.03, 0.09)
    f0 = base * (1 + wob * walk + 0.03 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))
    jit = rng.uniform(0.0005, 0.008) if fake else rng.uniform(0.008, 0.025)
    shim = rng.uniform(0.005, 0.06) if fake else rng.uniform(0.05, 0.15)
    p = np.zeros(n); pos = 0.0
    while pos < n - 1:
        p[int(pos)] = max(0.2, 1 + shim * rng.normal())
        pos += SR / f0[int(pos)] * (1 + jit * rng.normal())
    g = lfilter([1], [1, -0.97], lfilter([1], [1, -0.9], p))
    out = np.zeros(n); i = 0
    while i < n:
        L = int(rng.uniform(0.08, 0.25) * SR)
        s = g[i:i + L]
        for fr, bw in zip((rng.uniform(300, 800), rng.uniform(900, 2300), rng.uniform(2400, 3000)), (80, 110, 160)):
            s = _res(s, fr, bw)
        out[i:i + L] = s; i += L
    out /= np.abs(out).max() + 1e-9
    ph = rng.uniform(0, 3)
    env = (0.85 + 0.15 * np.sin(2 * np.pi * 3 * t + ph)) if fake else (0.55 + 0.45 * np.abs(np.sin(2 * np.pi * rng.uniform(2, 5) * t + ph)))
    out *= env
    for _ in range(rng.integers(0, 2) if fake else rng.integers(0, 3)):
        s0 = rng.integers(0, n - 4000); out[s0:s0 + rng.integers(1200, 4000)] *= 0.03
    out += (rng.uniform(0.001, 0.02) if fake else rng.uniform(0.01, 0.06)) * lfilter([1, -0.95], [1], rng.normal(0, 1, n))
    if fake and rng.random() < 0.85:
        out += rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * rng.uniform(5500, 7500) * t) * (1 + np.sin(2 * np.pi * rng.uniform(50, 200) * t)) / 2
    out = out / (np.abs(out).max() + 1e-9) * 0.5 * rng.uniform(0.2, 1.0)
    snr = rng.uniform(15, 45)
    out += rng.normal(0, np.sqrt(np.mean(out ** 2)) / 10 ** (snr / 20), n)
    return out.astype(np.float32)


def read_wav(src):
    sr, x = wavfile.read(src)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max if x.dtype.kind == "i" else 255)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR); x = resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


def _from_folder(folder):
    X, G = [], []
    for f in sorted(Path(folder).glob("*.wav")):
        x = read_wav(f)
        for i in range(0, len(x) - SR + 1, SR):
            r = extract(x[i:i + SR])
            if r: X.append(r[0]); G.append(f"{Path(folder).name}/{f.name}")
    return X, G


def eer(y, score):
    """Equal error rate (lower is better) for label 1 = real, score = P(real)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, score); fnr = 1 - tpr; i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


LAST_SPLIT = {}


def build_dataset(n_per_class=800, seed=0):
    """Returns X, y, groups. groups = source file, so train/test can be split BY FILE (no window leakage)."""
    global LAST_SOURCE
    (real, gr), (fake, gf) = _from_folder(ROOT / "data" / "real"), _from_folder(ROOT / "data" / "fake")
    if len(real) >= 50 and len(fake) >= 50:
        LAST_SOURCE = "user_dataset"
        print(f"Using your dataset: {len(real)} real / {len(fake)} fake windows from {len(set(gr))}+{len(set(gf))} files")
        return np.array(real + fake), np.array([1] * len(real) + [0] * len(fake)), np.array(gr + gf)
    LAST_SOURCE = "synthetic"
    print("No (or too little) data in data/real + data/fake -> using SYNTHETIC training data")
    rng = np.random.default_rng(seed); X, y = [], []
    for label in (0, 1):
        c = 0
        while c < n_per_class:
            r = extract(synth(rng, fake=(label == 0)))
            if r: X.append(r[0]); y.append(label); c += 1
    return np.array(X), np.array(y), np.arange(len(y)).astype(str)


def train(n_per_class=800):
    from sklearn.model_selection import GroupShuffleSplit
    X, y, g = build_dataset(n_per_class)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(X, y, g))
    if LAST_SOURCE == "user_dataset":                      # leakage guard: no recording is on both sides
        assert not (set(g[tr]) & set(g[te]))
    LAST_SPLIT.update(train_files=sorted(set(g[tr])), test_files=sorted(set(g[te])))
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=1, n_jobs=-1).fit(X[tr], y[tr])
    acc = clf.score(X[te], y[te]); e = eer(y[te], clf.predict_proba(X[te])[:, 1])
    note = "held-out FILES" if LAST_SOURCE == "user_dataset" else "synthetic, same generator (NOT real-world)"
    print(f"Hold-out accuracy: {acc:.3f}  EER: {e:.3f}  [{note}]  (label 1 = real, 0 = synthetic)")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    import sklearn
    byfile = LAST_SOURCE == "user_dataset"
    META_PATH.write_text(json.dumps({
        "trained_on": LAST_SOURCE, "holdout_accuracy": round(float(acc), 3), "holdout_eer": round(e, 3),
        "split": "by_file" if byfile else "synthetic_random",
        "split_note": ("whole recordings are held out: no file contributes windows to both train and test" if byfile else
                       "no recordings exist: every window is an independently generated synthetic sample, so the split is random "
                       "per window. It measures separability of the generator's own two classes, NOT generalisation to real speech or real clones"),
        "train_windows": int(len(tr)), "test_windows": int(len(te)), "sklearn": sklearn.__version__}))
    return clf, acc


class FastForest:
    """Low-overhead single-window inference for a fitted RandomForestClassifier.

    sklearn's predict_proba spends ~10 ms per call on input validation and joblib dispatch, which dominates for one
    row. This wrapper averages the trees' own predictions directly. It is verified against sklearn when constructed
    and falls back to sklearn automatically if the two ever disagree (e.g. after a scikit-learn upgrade changes the
    tree layout), so correctness never depends on the shortcut."""

    def __init__(self, clf, probe=None):
        clf.n_jobs = 1                                   # one row: a thread pool per call is pure overhead
        self.clf, self.trees, self.fast = clf, [e.tree_ for e in clf.estimators_], False
        if probe is not None and len(probe):
            try:
                self.fast = bool(np.allclose(self._fast(probe), clf.predict_proba(probe), atol=1e-9, rtol=0))
            except Exception:
                self.fast = False

    def _fast(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        acc = 0.0
        for t in self.trees:
            v = t.predict(X)
            acc = acc + (v[:, 0, :] if v.ndim == 3 else v)
        return acc / len(self.trees)

    def predict_proba(self, X):
        return self._fast(X) if self.fast else self.clf.predict_proba(X)

    def __getattr__(self, name):                         # score(), n_estimators, ... behave like the wrapped model
        return getattr(self.clf, name)


def _probe(n=24):
    rng = np.random.default_rng(12345)
    return np.array([extract(synth(rng, i % 2 == 0))[0] for i in range(n)])


def load_or_train():
    clf = None
    if MODEL_PATH.exists():
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")           # an InconsistentVersionWarning means the pickle came from another scikit-learn
                clf = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Saved model unusable ({type(e).__name__}) - retraining for this machine...")
    if clf is None:
        print("Training the prototype model (~30-60 s)...")
        clf = train(800)[0]
    return FastForest(clf, _probe())


def model_info():
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"trained_on": "unknown"}
    return {"name": "RandomForest (scikit-learn, CPU)", "validated_on_real_cloned_voices": False, **meta}
