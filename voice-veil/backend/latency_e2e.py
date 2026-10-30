"""Capture-to-verdict latency over the real WebSocket, with real-time paced audio replay.
Usage (server running):  python latency_e2e.py [ws_url] [runs]
Measures, per chunk: chunk-sent -> verdict-received, and for a natural->synthetic switch: audio onset -> first ELEVATED / HIGH.
NOTE: replayed synthetic test audio over loopback. It measures system latency, NOT detection accuracy, and not microphone hardware latency."""
import asyncio, json, sys, time
import numpy as np, websockets
from app.model import synth

CH = 2048; D = CH / 16000


async def once(url, pre, post, sens, seed):
    rng = np.random.default_rng(seed)
    nat = np.concatenate([synth(rng, False, base=130) for _ in range(pre)])
    syn = np.concatenate([synth(rng, True, base=130) for _ in range(post)])
    x = np.concatenate([nat, syn]).astype(np.float32); sent, res = {}, []
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"sensitivity": sens}))
        async def rx():
            async for m in ws:
                d = json.loads(m)
                if "state" in d and "chunks" in d: res.append((time.perf_counter(), d))
        t0 = time.perf_counter(); task = asyncio.create_task(rx())
        for i in range(len(x) // CH):
            await asyncio.sleep(max(0, t0 + (i + 1) * D - time.perf_counter()))   # chunk i is fully "captured" now
            sent[i + 1] = time.perf_counter(); await ws.send(x[i * CH:(i + 1) * CH].tobytes())
        await asyncio.sleep(1.0); task.cancel()
    onset = t0 + len(nat) / 16000
    first = lambda ok: next(((t - onset) * 1000 for t, d in res if t > onset and ok(d)), None)
    return {"rtt": [(t - sent[d["chunks"]]) * 1000 for t, d in res if d["chunks"] in sent],
            "to_elevated": first(lambda d: d["state"] in ("ELEVATED", "HIGH")), "to_high": first(lambda d: d["state"] == "HIGH"),
            "false_high_before_onset": sum(1 for t, d in res if t <= onset and d["state"] == "HIGH"), "decisions": len(res)}


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765/ws/stream"; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for sens in (50, 100):
        out = [await once(url, 6, 6, sens, 40 + k) for k in range(runs)]
        rtt = np.array([v for o in out for v in o["rtt"]]); th = [o["to_high"] for o in out]; te = [o["to_elevated"] for o in out]
        f = lambda a: "n/a (never)" if not [v for v in a if v is not None] else f"{np.median([v for v in a if v is not None]):.0f} ms (runs {[None if v is None else round(v) for v in a]})"
        print(f"sensitivity {sens}: chunk-sent->verdict median {np.median(rtt):.0f} ms, p95 {np.percentile(rtt, 95):.0f} ms, n={len(rtt)}")
        print(f"   onset->first ELEVATED: {f(te)}\n   onset->HIGH warning:   {f(th)}\n   HIGH before onset (false alarms): {[o['false_high_before_onset'] for o in out]}")
asyncio.run(main())
