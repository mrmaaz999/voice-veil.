"""Run any WAV through the exact streaming pipeline (1 s window, decision every 0.5 s, same fusion/smoothing) and print the outcome.
Usage: python check_wav.py file.wav [more.wav ...]
Use it to see how the PROTOTYPE behaves on natural speech vs text-to-speech vs generated test audio. A handful of files proves nothing about accuracy."""
import sys
from collections import Counter
import numpy as np
from app.main import Session, SR, HOP, WIN
from app.model import read_wav

for path in sys.argv[1:]:
    x = read_wav(path)
    print(f"\n{path}: {len(x) / SR:.1f} s")
    for sens in (50, 100):
        s = Session(); s.control({"sensitivity": sens}); res = [s.step(x[i:i + WIN]) for i in range(0, len(x) - WIN + 1, HOP)]
        c = Counter(r["state"] for r in res); p = [r["signals"]["acoustic"] for r in res if r.get("signals")]
        first = next((i * HOP / SR + 1.0 for i, r in enumerate(res) if r["state"] == "HIGH"), None)
        print(f"  sensitivity {sens:3d}: {len(res)} decisions {dict(c)} | model P(real) median {np.median(p) if p else float('nan'):.2f} "
              f"[{min(p) if p else float('nan'):.2f}-{max(p) if p else float('nan'):.2f}] | first HIGH at {'never' if first is None else f'{first:.1f} s'}")
