"""Evaluate the current model on LABELED audio it was NOT trained on.
Usage: python evaluate.py <folder_of_real_wavs> <folder_of_cloned_wavs>
Use recordings/speakers that are not in data/real or data/fake. Reports window- and file-level accuracy and EER."""
import sys
from pathlib import Path
import numpy as np
from app.model import load_or_train, _from_folder, eer

if len(sys.argv) != 3:
    sys.exit(__doc__)
m = load_or_train()
(R, gr), (F, gf) = _from_folder(sys.argv[1]), _from_folder(sys.argv[2])
if not R or not F:
    sys.exit("No usable 16-bit WAV speech (>=1 s) found in one of the folders.")
X, y, g = np.array(R + F), np.array([1] * len(R) + [0] * len(F)), np.array(gr + gf)
p = m.predict_proba(X)[:, 1]
fa = {k: (float(p[g == k].mean() >= 0.5), int(y[g == k][0])) for k in set(g)}
print(f"files: {len(set(gr))} real, {len(set(gf))} cloned | windows: {len(R)} real, {len(F)} cloned")
print(f"window accuracy @0.5: {((p >= 0.5) == y).mean():.3f} | EER: {eer(y, p):.3f} | file-level accuracy: {np.mean([a == b for a, b in fa.values()]):.3f}")
if len(set(gr)) < 20 or len(set(gf)) < 20:
    print("WARNING: fewer than 20 files per class - these numbers are not statistically meaningful.")
