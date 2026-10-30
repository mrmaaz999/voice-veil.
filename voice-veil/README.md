# Voice Veil – Live Deepfake Call Shield (Team AVENGERS, Dexathon 2026) — v3.1

> **Read this first.** This is a working *pipeline prototype*. Its model is trained only on generated synthetic signals and has **never seen real speech or real cloned voices**.
> In our checks it flagged every ordinary recorded human voice we tried as HIGH risk (see "Natural speech vs synthetic audio"). It is **not** a validated deepfake detector.

## Run on Windows (PowerShell)
Run from the folder that contains `backend`, `frontend` and `run.bat` (if `dir` does not show them, `cd` into the inner `voice-veil` folder):

    cd C:\path\to\voice-veil
    .\run.bat

Then open http://localhost:8000 in Chrome or Edge. First start installs packages and trains the prototype model (~20–60 s). No API keys, cloud services or paid APIs are used.

Manual start (Python 3.10–3.13; if pip wants a C++ compiler you are on a Python too new for some wheels – install 3.12):

    cd C:\path\to\voice-veil\backend
    py -3.12 -m venv ..\.venv
    Set-ExecutionPolicy -Scope Process Bypass        # only if activation is blocked
    ..\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

## Deploy to Render

The repository includes a root-level `render.yaml` Blueprint for deploying the full FastAPI app, including its WebSocket audio stream:

1. Sign in to [Render](https://render.com/) and choose **New + → Blueprint**.
2. Connect this GitHub repository and select the `main` branch. Render will read `render.yaml` and create the service.
3. Wait for the first build to finish, then open the service's `onrender.com` URL.

The bundled model is used at startup. The free instance may sleep when idle, so the first request after inactivity can take longer. Its filesystem is ephemeral; saved event logs may be lost when the service restarts. Audio is sent to the hosted server for processing in memory and is not stored. Only the time and score of high-risk events are logged. This prototype is not a validated deepfake detector; do not rely on its output for real calls.

## Tests (PowerShell; 21 backend tests, 8 frontend tests, browser E2E)

    cd C:\path\to\voice-veil\backend
    python -m pytest -q tests                  # 21 backend tests
    python bench.py 300                        # compute latency per decision (features vs model)
    python make_test_audio.py                  # writes e2e\audio\natural_48000.wav + synthetic_48000.wav (GENERATED test audio)
    python check_wav.py your_recording.wav     # run any WAV through the exact streaming pipeline, both sensitivities

    # server for the next two steps, in a SECOND PowerShell window:
    cd C:\path\to\voice-veil\backend; ..\.venv\Scripts\Activate.ps1; python -m uvicorn app.main:app --port 8765
    python latency_e2e.py ws://127.0.0.1:8765/ws/stream 3     # real WebSocket: chunk->verdict and onset->warning

    cd C:\path\to\voice-veil\e2e
    npm install
    node --test unit.test.mjs                  # 8 resampler + worklet unit tests (Node 18+)
    $env:CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"   # or ...\Edge\Application\msedge.exe
    node run.mjs .\audio\natural_48000.wav                              # 37 checks; the WAV is played as the fake microphone
    $env:EXPECT = "high"; node run.mjs .\audio\synthetic_48000.wav      # 40 checks: additionally requires the HIGH warning + event log entry

`run.mjs` drives the real dashboard in Chrome with `--use-fake-device-for-media-stream --use-file-for-fake-audio-capture`, so the genuine
`getUserMedia -> AudioWorklet -> resampler -> WebSocket` path runs. It does **not** use real microphone hardware. Defaults to http://127.0.0.1:8765.

## How it works
Browser (mic or shared-tab audio) -> AudioWorklet averages all channels to mono -> streaming resampler to 16 kHz -> Float32 chunks (~130 ms)
over a local WebSocket -> 1 s window, one decision per 0.5 s -> features (32 ms frames, 10 ms hop: MFCC, spectral, pitch/jitter/shimmer, pauses)
-> **RandomForest (scikit-learn, CPU)** -> fused with a pitch cue and optional Voice ID match -> rolling median -> HIGH after 3 consecutive low decisions.
States: Monitoring · Inconclusive (clipping, invalid audio, analysis failure) · Low risk indicators · Elevated · High. Never "safe".
The dashboard shows what the backend really receives (rate, format, level, clipping, browser capture rate/channels) and warns on a rate mismatch.

## What is VERIFIED and what is NOT
Verified by running (Linux sandbox, 1 vCPU, headless Chrome 131 with Chrome's *fake audio device* playing a WAV file as the microphone):
* Backend 21/21 tests, frontend 8/8 unit tests. Browser E2E, four separate audio files, all passing: generated "natural-class" 37/37, generated "synthetic-class" 40/40,
  real recorded human speech 37/37, espeak-ng text-to-speech 37/37.
* Covered by those E2E checks: Start/Stop, restart, rapid start/stop x4, device list, sensitivity at 0 and 100 (UI threshold *and* backend threshold),
  waveform + timeline drawing, High-risk warning shown iff the backend sent HIGH, event log row = time + score only, demo mode labelling,
  clear-events (confirm dialog), mobile width (390 px) without horizontal scroll, no console/page errors, no failed requests.
* Audio path: browser captured 44.1 kHz stereo -> backend measured 16.0 kHz, float32 mono, 0 dropped frames, non-silent, not clipped.
* Cleanup on Stop: mic tracks 0 live, AudioContext and WebSocket references released, WebSocket closed on the wire, no frames sent afterwards, single send loop after restart.
* Server log over the final runs: 228 requests, all HTTP 200, 0 tracebacks.

NOT verified: a real microphone or sound card, Edge, Firefox/Safari, **Windows** (everything above ran on Linux), phones, real tab-audio capture (`getDisplayMedia`),
real phone calls, detection of real cloned voices (ElevenLabs etc.), and any accuracy on real speech. The fake-audio-device test proves the software path, not hardware behaviour
(device gain, echo cancellation, USB/Bluetooth latency).

## Natural speech vs synthetic audio (kept separate on purpose)
Pipeline outcome of `check_wav.py` on audio the model was **not** trained on (decisions every 0.5 s; HIGH = 3 consecutive low decisions):

| Audio | Kind | Decisions at default sensitivity 50 | First HIGH |
|---|---|---|---|
| Generated "natural-class" (make_test_audio.py) | synthetic generator, the model's own training distribution | 25 LOW_RISK, 2 ELEVATED, 0 HIGH (sens 100: 4 HIGH) | never (sens 100: 2.0 s) |
| Generated "synthetic-class" | same generator | 25 HIGH, 2 ELEVATED | 2.0 s |
| JFK speech, 11 s (public domain, real human, old recording) | **real human speech** | 19 HIGH, 2 ELEVATED | 2.0 s |
| TIMIT sentence, 2.9 s / read-speech clip, 3.6 s | **real human speech** | HIGH in 2/4 and 4/6 decisions | 2.0 s |
| 30 s two-speaker conversation (pyannote sample) | **real human speech** | 45 HIGH, 12 MONITORING (pauses), 2 ELEVATED | 8.0 s |
| espeak-ng TTS, 3 sentences/voices | ordinary text-to-speech (not a modern neural clone) | HIGH in 5–10 of 15–16 decisions (more at sens 100) | 2.0–3.0 s |

What this shows, and what it does **not**:
* On generated audio the model separates its two classes (that is what it was trained on). That is plumbing and self-consistency, **not** detection accuracy.
* On every real human recording we tried (4 distinct recordings: JFK, a TIMIT sentence, a short read-speech clip and a 30 s two-speaker conversation) the prototype said HIGH. It therefore does **not** distinguish real from fake on real speech; it has a false-alarm
  problem on natural voices. The TTS result is HIGH too, but for a model that flags real speech as HIGH that is not evidence it detects TTS.
* Four recordings is far too few for any rate or accuracy figure. We report **no** real-world accuracy, and none should be claimed. The third-party clips are not bundled (licences); test your own files with `check_wav.py`.

## Latency (measured; sandbox, loopback, 1 vCPU shared with headless Chrome — expect different numbers on your laptop)
| What | Result |
|---|---|
| Feature extraction | ~3.5 ms median |
| Model inference | ~1.0 ms median (was ~11–16 ms before the fix below, same machine) |
| Full compute per decision | 4.6 / 4.8 ms median, 6.1 / 7.6 ms p95 (bench.py, 300 windows, two runs). Before: 15.9 ms median, 24.6 ms p95 (A/B, same machine, identical decisions) |
| Chunk sent -> verdict received (real WebSocket, loopback) | 8 ms median, 11–12 ms p95 (latency_e2e.py, n=69 per setting); 15–19 ms in the browser |
| Audio onset -> first Elevated | 0.66–1.7 s |
| Audio onset -> HIGH warning | sensitivity 50: 2.71 s (3 of 3 runs); sensitivity 100: 2.71 s, 1.69 s, 2.72 s |
| HIGH before the switch (false alarms on the generated natural audio) | sensitivity 50: 1 in 3 runs; sensitivity 100: 4, 0 and 1 |

Not measured: microphone/OS audio-input latency. **A sub-200 ms end-to-end claim is NOT supported**; a warning needs a 1 s window plus 3 consecutive decisions,
so onset->warning is dominated by that design, not by compute. The sensitivity slider trades speed for false alarms.

### Why latency differs between environments (investigated)
A reported 49 ms median compute on another machine vs ~14 ms here for the same code pointed at fixed per-call overhead, not the model's maths. Found and fixed:
1. **scikit-learn `predict_proba` overhead.** For a single row almost all of the ~11–16 ms was input validation and joblib dispatch. `FastForest` (app/model.py) averages the trees' own predictions directly:
   identical probabilities (max difference 0.0; identical state and score on 300 windows), ~1 ms. It checks itself against sklearn at start-up and falls back to sklearn automatically if they ever disagree (tested).
2. **Thread pool per call.** The saved model had `n_jobs=-1`; on a multi-core machine every prediction can spin up a thread pool for a 1-row input. It is now `n_jobs=1`.
3. **BLAS/OpenMP threads.** The matrices here are tiny, so worker threads only add wake-up cost; `OMP/OPENBLAS/MKL_NUM_THREADS` default to 1 (override by setting them yourself).
4. Remaining variation is genuine CPU speed, power plan (use "Best performance" on laptops), and contention: while the browser and server shared this single core, compute rose temporarily (38 ms before the fix, ~10 ms after).

Limits: this sandbox has **1 vCPU**, so the multi-core effects in items 2–3 could not be reproduced or measured here; they are the expected cause, not a measured one. Re-run `python bench.py 300` on your machine and compare.

## Honest model status
* The bundled model learns from **generated synthetic signals** (800 windows per class). Its hold-out accuracy (**0.892**, EER 0.117) is on the same generator and says nothing about real clones or real speech.
  It is a labeled PROTOTYPE (banner in the UI). It is not PyTorch and not ONNX.
* **Dataset split — what the code actually does** (`models/voice_veil.meta.json` and `/api/health` state it):
  * *No files in `data/real` + `data/fake`* (the default): all training windows are generated; every window is an independent sample, so the split is a random per-window hold-out
    (`"split": "synthetic_random"`). There are no recordings to leak, but the score only measures how separable the generator's own classes are.
  * *With your own WAVs (>= 50 windows per class):* the split is **by file** (`"split": "by_file"`, enforced by an assertion): no recording contributes windows to both train and test. A leakage bug in an earlier random-window split was fixed.
  * Earlier docs quoted "0.89"; the first-start auto-train used 500 windows (0.864) while `train.py` used 800 (0.892). Both now use 800, so one number applies.
* A pretrained model was investigated: AASIST (MIT, github.com/clovaai/aasist) has weights, but no labeled real-vs-clone speech was reachable
  for evaluation, it needs PyTorch, and it was trained on ASVspoof 2019 and is known to generalise poorly to modern clones/phone audio. Not integrated.
* To train/evaluate on real data: put WAVs in `data/real/` and `data/fake/` (e.g. ASVspoof, asvspoof.org), then `cd backend; python train.py`.
  Then run `python evaluate.py <real_dir> <fake_dir>` on speakers/recordings NOT used for training; it reports accuracy and EER and warns if there are too few files.
* A saved model that is corrupt or was written by a different scikit-learn version is detected at start-up and retrained automatically (tested).
* Voice ID is stable for the same voice but unreliable for similar-pitch voices (synthetic tests only).
* "Timing stability" = regularity of chunk arrival between browser and local server, not carrier network jitter.
* Browsers cannot capture normal phone calls; use speaker mode or tab audio for web calls.
* Privacy: audio stays in memory on this machine; only `{time, score}` of HIGH events is written to `logs/events.jsonl` (Clear button deletes it).
  The backend stores no device names. No raw audio is persisted.

## Suggested PowerPoint wording (no .pptx was included in the upload, so these are text replacements to apply by hand)
* Slide 2/3 "scored in short frames (20 ms)": "Audio is analysed in 1-second windows (32 ms feature frames), with one decision every 0.5 s."
* "under 200 ms per decision": "About 5 ms of compute per decision on our 1-core test machine (measure yours with bench.py); a warning appears roughly 2–3 s after a voice change because it needs several consecutive windows (measured)."
* "PyTorch classifier / ONNX Runtime": "A lightweight scikit-learn RandomForest runs on the CPU; ONNX export is planned."
* "Trained on public real vs cloned speech": "The current prototype is trained only on generated synthetic signals. It has not been validated on real speech and, in our checks, flags real human speech as high risk. Training and evaluation on ASVspoof with held-out speakers is the next step."
* Any accuracy figure: "0.89 hold-out accuracy on synthetic data from the same generator — not a real-world detection rate."
* "Network-jitter cues": "Timing-stability cue (audio chunk regularity)."
* "Runs on existing phones": "Runs in a laptop/phone browser; native phone-call capture needs a mobile app (future work)."
* Keep: on-device processing, no raw audio stored, only time + score logged, fused cues, rolling smoothing, adjustable sensitivity.
