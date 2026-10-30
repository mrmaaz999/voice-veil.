"""Measure per-decision COMPUTE latency (feature extraction + inference + fusion). Usage: python bench.py [n]
End-to-end delay is larger: a decision needs a 1 s audio window and runs every 0.5 s."""
import sys, time, numpy as np
from app.main import Session, SR
from app.model import synth
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(0); wins = [synth(rng, i % 2 == 0) for i in range(n)]
s = Session(); s.step(wins[0]); lat, feat, inf = [], [], []
for w in wins:
    t = time.perf_counter(); r = s.step(w); lat.append((time.perf_counter() - t) * 1000)
    if 'feature_ms' in r: feat.append(r['feature_ms']); inf.append(r['infer_ms'])
lat = np.array(lat)
print(f"n={n} compute latency ms: median={np.median(lat):.1f} p95={np.percentile(lat,95):.1f} max={lat.max():.1f}")
print(f"  of which feature extraction median={np.median(feat):.1f} ms, model inference median={np.median(inf):.1f} ms")
print("Decision cadence: every 0.5 s on a 1 s window -> audio-to-verdict delay is >= ~1 s, not < 200 ms.")
