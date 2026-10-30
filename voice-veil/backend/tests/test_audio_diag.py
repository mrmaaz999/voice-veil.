import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from app.audiostats import AudioStats
from app.main import app, Session, SR
from app.model import synth

c = TestClient(app)

def test_audiostats_measures_real_rate_from_arrival_times():
    s = AudioStats(); t = 100.0
    for i in range(20): s.feed(np.zeros(2048, np.float32) + 0.1, t + i * 2048 / 16000)   # paced exactly at 16 kHz
    snap = s.snapshot(); assert snap["rate_ok"] and abs(snap["rate_hz"] - 16000) < 400 and snap["format"] == "float32 mono"

def test_audiostats_flags_wrong_rate():
    s = AudioStats()                                   # 48 kHz worth of samples arriving in real time = resampler bug
    for i in range(20): s.feed(np.zeros(6144, np.float32), 50.0 + i * 2048 / 16000)
    assert s.snapshot()["rate_ok"] is False

def test_audiostats_handles_nan_and_sanitizes_client_info():
    s = AudioStats(); s.feed(np.full(100, np.nan, np.float32), 1.0)
    s.set_client({"ctx_rate": 48000, "input_channels": 2, "device_label": "Alice's AirPods", "channels": "x", "auto_gain": True})
    snap = s.snapshot(); assert snap["client"] == {"ctx_rate": 48000, "input_channels": 2, "auto_gain": True}   # no device names stored

def test_decision_has_timing_breakdown():
    r = Session().step(np.concatenate([synth(np.random.default_rng(1), False)]))
    assert r["feature_ms"] >= 0 and r["infer_ms"] >= 0 and r["latency_ms"] >= r["infer_ms"]

def test_ws_decisions_carry_chunk_counter_and_server_ms_and_bad_client_info_rejected():
    x = synth(np.random.default_rng(2), False, 3.0).astype(np.float32)
    with c.websocket_connect("/ws/stream") as ws:
        ws.send_text('{"client": 5}'); assert ws.receive_json()["type"] == "error"
        ws.send_text('{"client": {"ctx_rate": 48000, "input_channels": 1}}')
        n = 0
        for i in range(0, len(x), 2048): ws.send_bytes(x[i:i + 2048].tobytes()); n += 1
        r = ws.receive_json()
        assert 1 <= r["chunks"] <= n and r["server_ms"] >= 0 and "state" in r

def test_frontend_assets_served():
    assert c.get("/static/resample.js").status_code == 200 and c.get("/static/worklet.js").status_code == 200
    assert c.get("/frontend/resample.js").status_code == 200 and c.get("/frontend/worklet.js").status_code == 200
    assert "StreamResampler" in c.get("/static/resample.js").text


def test_user_dataset_split_is_by_file_and_evaluate_runs(tmp_path, monkeypatch):
    """Regression: windows from one recording must never be on both sides of the train/test split."""
    from scipy.io import wavfile
    from app import model as M
    for cls, fake in (("real", False), ("fake", True)):
        d = tmp_path / "data" / cls; d.mkdir(parents=True)
        for i in range(8):
            x = np.concatenate([synth(np.random.default_rng(i + (50 if fake else 0)), fake) for _ in range(8)])
            wavfile.write(d / f"{cls}{i}.wav", 16000, (x * 32767).astype(np.int16))
    monkeypatch.setattr(M, "ROOT", tmp_path); monkeypatch.setattr(M, "MODEL_PATH", tmp_path / "m.joblib"); monkeypatch.setattr(M, "META_PATH", tmp_path / "m.json")
    M.train()
    tr, te = set(M.LAST_SPLIT["train_files"]), set(M.LAST_SPLIT["test_files"])
    assert tr and te and not (tr & te)
    import json; assert json.loads((tmp_path / "m.json").read_text())["split"] == "by_file"


def test_one_log_entry_per_high_episode_not_per_window(tmp_path, monkeypatch):
    """Regression: a continuous HIGH episode used to write a row for every 0.5 s decision."""
    from app import main as Mn
    monkeypatch.setattr(Mn, "LOG", tmp_path / "logs" / "events.jsonl")
    s = Session(log=True); s.control({"sensitivity": 100}); x = np.concatenate([synth(np.random.default_rng(7), True, base=130) for _ in range(10)])
    highs = sum(s.step(x[i:i + SR])["state"] == "HIGH" for i in range(0, len(x) - SR, SR // 2))
    lines = (tmp_path / "logs" / "events.jsonl").read_text().splitlines() if (tmp_path / "logs" / "events.jsonl").exists() else []
    assert highs >= 6 and 1 <= len(lines) <= max(2, highs // 3)
