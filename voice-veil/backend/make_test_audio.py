"""Create fake-microphone WAV files for the browser E2E test (GENERATED SYNTHETIC audio, for plumbing tests only).
Usage: python make_test_audio.py [out_dir]   -> natural_48000.wav, synthetic_48000.wav"""
import sys
from math import gcd
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from app.model import synth

out = Path(sys.argv[1] if len(sys.argv) > 1 else "../e2e/audio"); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(21)
for name, fake in (("natural", False), ("synthetic", True)):
    x = np.concatenate([synth(rng, fake, base=130) for _ in range(14)]); g = gcd(48000, 16000)
    y = resample_poly(x, 48000 // g, 16000 // g); wavfile.write(out / f"{name}_48000.wav", 48000, (y / np.abs(y).max() * 0.6 * 32767).astype(np.int16))
print("wrote", [p.name for p in out.glob("*.wav")], "in", out.resolve())
