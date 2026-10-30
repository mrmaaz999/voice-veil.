"""Facts about the audio the backend ACTUALLY receives (rate, level, clipping), so the UI can show them."""
import numpy as np

EXPECTED_RATE = 16000


class AudioStats:
    def __init__(self):
        self.t0 = self.last = None
        self.first = self.total = self.chunks = self.dropped = 0
        self.rms = self.peak = self.clip = 0.0
        self.client = {}

    def feed(self, chunk, now):
        c = np.nan_to_num(np.asarray(chunk, np.float32))
        if self.t0 is None:
            self.t0, self.first = now, len(c)
        self.last = now
        self.total += len(c); self.chunks += 1
        a = np.abs(c)
        self.rms, self.peak, self.clip = float(np.sqrt(np.mean(c ** 2))), float(a.max()), float(np.mean(a > 0.99))

    def set_client(self, d):
        """Allow-listed, sanitized browser capture info (no device names: those can identify a person)."""
        keep = {}
        for k in ("ctx_rate", "channels", "input_channels"):
            v = d.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 400000: keep[k] = int(v)
        for k in ("echo_cancellation", "noise_suppression", "auto_gain"):
            if isinstance(d.get(k), bool): keep[k] = d[k]
        self.client = keep

    def snapshot(self):
        dur = (self.last - self.t0) if self.t0 is not None else 0.0
        rate = (self.total - self.first) / dur if dur >= 1.0 else None   # samples/s actually delivered (wall clock)
        return {"type": "diag", "format": "float32 mono", "expected_rate_hz": EXPECTED_RATE,
                "rate_hz": None if rate is None else round(rate),
                "rate_ok": None if rate is None else abs(rate - EXPECTED_RATE) <= 0.08 * EXPECTED_RATE,
                "chunks": self.chunks, "dropped_frames": self.dropped, "rms": round(self.rms, 4),
                "peak": round(self.peak, 3), "clipped_pct": round(100 * self.clip, 2), "client": self.client}
