"""Usage: python train.py [windows_per_class]   (run from the backend/ folder)
Put your own WAV files in data/real/ and data/fake/ to train on real data."""
import sys
from app.model import train
train(int(sys.argv[1]) if len(sys.argv) > 1 else 800)
