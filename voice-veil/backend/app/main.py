"""Voice Veil API. All audio is processed in memory; only {time, score} of high-risk events is written to disk."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # must precede the numpy import: tiny matrices,
    os.environ.setdefault(_v, "1")                                          # so BLAS worker threads only add wake-up cost
import asyncio, io, json, sys, time
from collections import deque
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from .audiostats import AudioStats
from .features import SR, extract
from .model import load_or_train, synth, read_wav, model_info

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "events.jsonl"
WIN, HOP = SR, SR // 2                     # 1 s window, one decision every 0.5 s
MAX_CHUNK_BYTES, PROFILE_DIM = SR * 4 * 2, 14
MODEL, INFO = load_or_train(), model_info()
INFO["fast_inference"] = bool(getattr(MODEL, "fast", False))
app = FastAPI(title="Voice Veil", version="3.1")
app.mount("/static", StaticFiles(directory=str(ROOT / "frontend")), name="static")
app.mount("/frontend", StaticFiles(directory=str(ROOT / "frontend")), name="frontend")

ADVICE = {
    "MONITORING": "Listening. Waiting for speech.",
    "INCONCLUSIVE": "Not enough reliable audio to judge. No verdict.",
    "LOW_RISK": "No strong risk indicators. This is not proof the voice is genuine.",
    "ELEVATED": "Some indicators look unusual. Verify the caller through a second channel.",
    "HIGH": "Possible cloned or synthetic voice. Hang up and call back on a number you already trust. Never share OTPs or PINs.",
}


class Session:
    """Fuse cues -> rolling median -> HIGH only after 3 consecutive low windows."""
    W = np.r_[np.ones(13), 6.0]            # pitch weighs more than one spectral dimension

    def __init__(self, log=False):
        self.hist, self.low, self.log, self.sens = deque(maxlen=5), 0, log, 50.0
        self.profile, self.enrolling, self.samples = None, False, []
        self.stats = AudioStats()
        self.in_high = False

    @property
    def threshold(self):
        return 0.3 + 0.4 * self.sens / 100  # sensitivity 0..100 -> threshold 0.30..0.70 (real effect)

    def control(self, d):
        if not isinstance(d, dict):
            raise ValueError("control message must be a JSON object")
        if "sensitivity" in d:
            self.sens = float(np.clip(float(d["sensitivity"]), 0, 100))
        if "client" in d:
            if not isinstance(d["client"], dict):
                raise ValueError("client info must be an object")
            self.stats.set_client(d["client"])
        if "profile" in d:
            p = d["profile"]
            if p is None:
                self.profile = None
            else:
                m, sd = np.array(p["mean"], float), np.array(p["sd"], float)
                if m.shape != (PROFILE_DIM,) or sd.shape != (PROFILE_DIM,) or not np.isfinite(m).all() or (sd <= 0).any():
                    raise ValueError("invalid voice profile")
                self.profile = {"mean": m, "sd": sd}
            self.hist.clear()
        if d.get("mode") == "enroll":
            self.enrolling, self.samples = True, []
        if d.get("mode") == "finish_enroll":
            self.enrolling = False
            if len(self.samples) < 8:
                return {"type": "error", "msg": "Not enough speech heard. Speak clearly for 10+ seconds and try again."}
            S = np.array(self.samples); m = S.mean(0)
            sd = np.maximum(S.std(0) * 1.5, 0.15 * np.abs(m) + 0.3); sd[-1] = max(S.std(0)[-1] * 1.5, 0.25)
            self.profile = {"mean": m, "sd": sd}; self.hist.clear()
            return {"type": "profile", "profile": {k: [round(float(x), 4) for x in v] for k, v in self.profile.items()}}

    def _res(self, state, **kw):
        if state in ("MONITORING", "INCONCLUSIVE"):
            self.in_high = False                       # episode ended (silence / unusable audio)
        base = {"state": state, "trust": None, "advice": ADVICE.get(state, ""), "reasons": [], "signals": {},
                "mode": "prototype", "threshold": round(self.threshold, 2)}
        return {**base, **kw}

    def step(self, win, quality=1.0):
        try:
            return self._step(np.asarray(win, np.float32), quality)
        except Exception as e:                       # never report "safe" because analysis failed
            print("analysis error:", type(e).__name__, file=sys.stderr)
            return self._res("INCONCLUSIVE", reasons=["Analysis failed on this window."], quality=round(quality, 2))

    def _step(self, win, quality):
        t0 = time.perf_counter()
        if not np.isfinite(win).all():
            return self._res("INCONCLUSIVE", reasons=["Audio contained invalid samples."], quality=round(quality, 2))
        if float(np.mean(np.abs(win) > 0.99)) > 0.05:
            return self._res("INCONCLUSIVE", reasons=["Audio is clipping. Lower the input volume."], quality=round(quality, 2))
        r = extract(win)
        if r is None:
            return self._res("MONITORING", reasons=["No speech detected in the last second."], quality=round(quality, 2))
        t_feat = (time.perf_counter() - t0) * 1000
        vec, meta = r
        fv = np.r_[vec[:13], meta["f0"] / 25]
        if self.enrolling:
            self.samples.append(fv)
            return self._res("ENROLLING", count=len(self.samples), quality=round(quality, 2))
        match = None
        if self.profile:
            d = float((np.abs(fv - self.profile["mean"]) / self.profile["sd"] * self.W).sum() / self.W.sum())
            match = float(np.exp(-0.5 * d * d))
        t1 = time.perf_counter()
        p = float(MODEL.predict_proba(vec[None])[0, 1])                     # acoustic cue (prototype model)
        t_inf = (time.perf_counter() - t1) * 1000
        pitch = None if meta["voiced"] < 0.1 else float(np.clip((meta["jitter"] - 0.003) / 0.02, 0, 1))
        score = 0.9 * p + 0.1 * (0.5 if pitch is None else pitch)
        self.hist.append(0.3 * score + 0.7 * match if match is not None else score)
        sm = float(np.median(self.hist))
        trust = 0.5 + (sm - 0.5) * quality           # unstable chunk timing -> pull toward 0.5 (less confident)
        thr = self.threshold
        self.low = self.low + 1 if trust < thr else 0
        state = "HIGH" if self.low >= 3 else "ELEVATED" if (self.low > 0 or trust < thr + 0.1) else "LOW_RISK"
        why = []
        if p < 0.4: why.append("Spectral and prosody features resemble the synthetic class of the prototype model.")
        if pitch is not None and pitch < 0.2: why.append(f"Pitch is unusually smooth (jitter {meta['jitter']*100:.2f}%).")
        if match is not None and match < 0.4: why.append("Voice does not match the enrolled trusted voice.")
        if quality < 0.8: why.append("Chunk timing is unstable, so confidence was reduced.")
        if not why: why.append("No individual cue stood out.")
        enter_high, self.in_high = state == "HIGH" and not self.in_high, state == "HIGH"
        if enter_high and self.log:                     # ONE entry per warning episode, not one per window
            LOG.parent.mkdir(exist_ok=True)
            with open(LOG, "a") as f:                # metadata only: never audio
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "score": round(trust, 3)}) + "\n")
        return self._res(state, trust=round(trust, 3), reasons=why, quality=round(quality, 2), f0=round(meta["f0"], 1),
                         signals={"acoustic": round(p, 3), "pitch": None if pitch is None else round(pitch, 3),
                                  "timing": round(quality, 3), "voice_id": None if match is None else round(match, 3)},
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1), feature_ms=round(t_feat, 1), infer_ms=round(t_inf, 1))


def analyse_signal(x, demo=False):
    s, out = Session(), []
    for i in range(0, max(len(x) - WIN, 0) + 1, HOP):
        r = s.step(x[i:i + WIN]); r["t"] = round(i / SR, 2); out.append(r)
    sc = [r["trust"] for r in out if r["trust"] is not None]
    if not sc:
        return {"verdict": "INCONCLUSIVE (no usable speech)", "windows": out, "demo": demo}
    lat = [r["latency_ms"] for r in out if "latency_ms" in r]
    return {"verdict": ("LIKELY SYNTHETIC" if np.median(sc) < s.threshold else "LIKELY NATURAL") + " (prototype)",
            "mean_trust": round(float(np.mean(sc)), 3), "avg_latency_ms": round(float(np.mean(lat)), 1),
            "windows": out, "demo": demo}


@app.get("/api/health")
def health():
    return {"ok": True, "model": INFO, "analysis": {"window_s": 1.0, "decision_every_s": 0.5, "feature_frame_ms": 32,
            "feature_hop_ms": 10, "audio_chunk_ms_approx": 128}, "audio_stored": False}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read(20_000_001)
    if len(raw) > 20_000_000:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        x = read_wav(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Please upload a valid WAV file")
    if len(x) < SR:
        raise HTTPException(400, "Audio must be at least 1 second long")
    return await asyncio.to_thread(analyse_signal, x)


@app.get("/api/demo/{kind}")
async def demo(kind: str, seed: int = 0):
    if kind not in ("real", "fake"):
        raise HTTPException(404, "kind must be 'real' or 'fake'")
    rng = np.random.default_rng(seed or int(time.time()))
    x = np.concatenate([synth(rng, kind == "fake") for _ in range(8)])
    return await asyncio.to_thread(analyse_signal, x, True)   # demo=True -> UI labels it DEMO


@app.get("/api/events")
def events():
    return [json.loads(l) for l in LOG.read_text().splitlines()][-50:][::-1] if LOG.exists() else []


@app.delete("/api/events")
def clear_events():
    LOG.unlink(missing_ok=True)
    return {"cleared": True}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    s, buf, new, arr = Session(log=True), np.zeros(0, np.float32), 0, deque(maxlen=24)
    nchunks, last_diag = 0, time.perf_counter()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    out = s.control(json.loads(msg["text"]))
                except (ValueError, KeyError, TypeError):
                    out = {"type": "error", "msg": "Invalid control message."}
                if out: await ws.send_json(out)
                continue
            data = msg.get("bytes")
            if not data or len(data) % 4 or len(data) > MAX_CHUNK_BYTES:   # malformed frame: ignore, keep session alive
                s.stats.dropped += 1
                continue
            t_rx = time.perf_counter(); arr.append(t_rx); nchunks += 1
            chunk = np.frombuffer(data, dtype=np.float32)
            s.stats.feed(chunk, t_rx)
            if t_rx - last_diag >= 2.0:
                last_diag = t_rx; await ws.send_json(s.stats.snapshot())
            buf = np.concatenate([buf, chunk])[-WIN:]; new += len(chunk)
            if new >= HOP and len(buf) >= WIN // 2:
                new, q = 0, 1.0
                if len(arr) >= 6:                     # timing cue = browser/loopback chunk-arrival stability, NOT carrier network jitter
                    d = np.diff(arr); q = 1 - float(np.clip(d.std() / d.mean() - 0.2, 0, 0.5))
                res = await asyncio.to_thread(s.step, buf.copy(), q)
                res["chunks"] = nchunks                               # lets the browser match this verdict to the chunk it sent
                res["server_ms"] = round((time.perf_counter() - t_rx) * 1000, 1)   # chunk received -> reply ready
                await ws.send_json(res)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({"name": "Voice Veil", "short_name": "VoiceVeil", "start_url": "/", "display": "standalone",
        "background_color": "#0b1020", "theme_color": "#0b1020", "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}]})


@app.get("/icon.svg")
def icon():
    return Response('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" rx="22" fill="#0b1020"/>'
        '<path d="M50 14 80 26v22c0 20-13 33-30 38C33 81 20 68 20 48V26z" fill="#7c8cff"/><path d="M36 50h6l4-10 8 22 4-12h6" stroke="#0b1020" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>', media_type="image/svg+xml")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
