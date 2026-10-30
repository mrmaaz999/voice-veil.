import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from app import main
from app.main import app, Session, SR, LOG
from app.model import synth

c = TestClient(app)
rng = np.random.default_rng(3)
def speech(base=None, fake=False, secs=1): return np.concatenate([synth(rng, fake, base=base) for _ in range(secs)])

def test_health_reports_prototype_model():
    h = c.get("/api/health").json()
    assert h["ok"] and h["audio_stored"] is False and h["model"]["validated_on_real_cloned_voices"] is False

def test_silence_is_monitoring_not_safe():
    r = Session().step(np.zeros(SR, np.float32)); assert r["state"] == "MONITORING" and r["trust"] is None

def test_nan_and_clipping_are_inconclusive():
    assert Session().step(np.full(SR, np.nan, np.float32))["state"] == "INCONCLUSIVE"
    assert Session().step(np.sign(np.sin(np.linspace(0, 2000, SR))).astype(np.float32))["state"] == "INCONCLUSIVE"

def test_model_failure_is_inconclusive(monkeypatch):
    class Boom:
        def predict_proba(self, X): raise RuntimeError("x")
    monkeypatch.setattr(main, "MODEL", Boom())
    r = Session().step(speech()); assert r["state"] == "INCONCLUSIVE" and r["trust"] is None

def test_sensitivity_changes_threshold_and_state():
    s = Session(); s.control({"sensitivity": 0}); lo = s.threshold; s.control({"sensitivity": 100}); hi = s.threshold
    assert abs(lo - 0.3) < 1e-6 and abs(hi - 0.7) < 1e-6
    s.control({"sensitivity": 999}); assert s.sens == 100

def test_high_needs_three_consecutive_windows_and_clears():
    s = Session(); s.control({"sensitivity": 100}); x = speech(fake=True, secs=6); st = []
    for i in range(0, len(x) - SR, SR // 2): st.append(s.step(x[i:i + SR])["state"])
    assert "HIGH" in st and st.index("HIGH") >= 2
    good = speech(secs=10); s2 = Session(); s2.control({"sensitivity": 0})
    assert all(s2.step(good[i:i + SR])["state"] != "HIGH" for i in range(0, len(good) - SR, SR // 2))

def test_demo_labelled_and_separates():
    a, b = c.get("/api/demo/real?seed=2").json(), c.get("/api/demo/fake?seed=2").json()
    assert a["demo"] and b["demo"] and a["mean_trust"] > b["mean_trust"]

def test_voice_id_match():
    s = Session(); s.control({"mode": "enroll"}); x = speech(120, secs=12)
    for i in range(0, len(x) - SR, SR // 2): s.step(x[i:i + SR])
    prof = s.control({"mode": "finish_enroll"})["profile"]
    def last(b):
        t = Session(); t.control({"profile": prof}); y = speech(b, secs=6); m = []
        for i in range(0, len(y) - SR, SR // 2): m.append(t.step(y[i:i + SR])["signals"]["voice_id"])
        return float(np.mean(m))          # mean over windows: a single window is noisy
    assert last(120) > 0.7 and last(190) < 0.5

def test_websocket_survives_malformed_input_and_logs_metadata_only():
    LOG.unlink(missing_ok=True)
    with c.websocket_connect("/ws/stream") as ws:
        ws.send_text("not json"); assert ws.receive_json()["type"] == "error"
        ws.send_text('{"profile":{"mean":[1],"sd":[1]}}'); assert ws.receive_json()["type"] == "error"
        ws.send_bytes(b"abc")                                   # odd length: must not crash
        ws.send_text('{"sensitivity": 100}')
        x = speech(fake=True, secs=4).astype(np.float32)
        for i in range(0, len(x), 4096): ws.send_bytes(x[i:i + 4096].tobytes())
        assert ws.receive_json()["state"] in ("MONITORING", "INCONCLUSIVE", "LOW_RISK", "ELEVATED", "HIGH")
    for l in (LOG.read_text().splitlines() if LOG.exists() else []):
        assert set(json.loads(l)) == {"time", "score"}

def test_events_clear_and_bad_upload():
    assert c.delete("/api/events").json()["cleared"] and c.get("/api/events").json() == []
    assert c.post("/api/analyze", files={"file": ("x.wav", b"nope")}).status_code == 400


def test_fastforest_matches_sklearn_exactly():
    """The low-overhead inference path must give the same probabilities as sklearn's predict_proba."""
    from app import main as Mn
    from app.features import extract
    rng = np.random.default_rng(99)
    X = np.array([extract(synth(rng, i % 2 == 0))[0] for i in range(40)])
    assert Mn.MODEL.fast is True
    assert np.allclose(Mn.MODEL.predict_proba(X), Mn.MODEL.clf.predict_proba(X), atol=1e-9, rtol=0)


def test_fastforest_falls_back_to_sklearn_when_it_disagrees(monkeypatch):
    """If a scikit-learn upgrade changed the tree layout, the self-check must disable the shortcut, not return wrong scores."""
    from app import main as Mn
    from app.model import FastForest, _probe
    monkeypatch.setattr(FastForest, "_fast", lambda self, X: np.full((len(X), 2), 0.5))
    ff = FastForest(Mn.MODEL.clf, _probe(6))
    assert ff.fast is False
    X = _probe(4)
    assert np.allclose(ff.predict_proba(X), Mn.MODEL.clf.predict_proba(X), atol=1e-9, rtol=0)


def test_corrupt_or_foreign_model_file_is_retrained_not_crashed(tmp_path, monkeypatch):
    """A model file copied from another machine / scikit-learn version must trigger a retrain, not a crash."""
    from app import model as M
    monkeypatch.setattr(M, "MODEL_PATH", tmp_path / "m.joblib"); monkeypatch.setattr(M, "META_PATH", tmp_path / "m.json")
    monkeypatch.setattr(M, "ROOT", tmp_path)                     # no data/ folder -> synthetic training
    real_train = M.train; monkeypatch.setattr(M, "train", lambda n=500: real_train(60))
    (tmp_path / "m.joblib").write_bytes(b"not a joblib file")
    m = M.load_or_train()
    assert m.fast is True and m.predict_proba(M._probe(2)).shape == (2, 2)
    import json; meta = json.loads((tmp_path / "m.json").read_text())
    assert meta["split"] == "synthetic_random" and "NOT generalisation" in meta["split_note"] and meta["train_windows"] > 0
