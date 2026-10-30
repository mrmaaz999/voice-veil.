// Frontend unit tests (Node >= 18):  node --test unit.test.mjs
import test from 'node:test'; import assert from 'node:assert/strict'; import { createRequire } from 'node:module'; import fs from 'node:fs';
const require = createRequire(import.meta.url); const { StreamResampler } = require('../frontend/resample.js');
const sine = (f, sr, n, a = 0.5) => Float32Array.from({ length: n }, (_, i) => a * Math.sin(2 * Math.PI * f * i / sr));
const zc = x => { let c = 0; for (let i = 1; i < x.length; i++) if (x[i - 1] < 0 && x[i] >= 0) c++; return c; };
const rms = x => Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length);
function run(inRate, chunk, f = 1000, secs = 3) {
  const x = sine(f, inRate, inRate * secs), rs = new StreamResampler(inRate), outs = [];
  for (let i = 0; i < x.length; i += chunk) outs.push(rs.push(x.subarray(i, i + chunk)));
  const o = new Float32Array(outs.reduce((a, b) => a + b.length, 0)); let p = 0; for (const b of outs) { o.set(b, p); p += b.length; } return o;
}
for (const sr of [22050, 32000, 44100, 48000, 96000]) test(`resampler ${sr} Hz -> 16 kHz keeps frequency, length and level (128-sample pushes)`, () => {
  const o = run(sr, 128), secs = 3;
  assert.ok(Math.abs(o.length - 16000 * secs) <= 3, `length ${o.length}`);
  const f = zc(o) / secs; assert.ok(Math.abs(f - 1000) / 1000 < 0.005, `freq ${f}`);
  assert.ok(Math.abs(rms(o) / (0.5 / Math.SQRT2) - 1) < 0.05, `rms ratio ${rms(o)}`);
});
test('streaming output is independent of chunk size', () => {
  const a = run(44100, 128), b = run(44100, 1000), n = Math.min(a.length, b.length) - 2; let d = 0; for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[i] - b[i])); assert.ok(d < 1e-4, 'max diff ' + d);
});
test('empty / tiny pushes are safe', () => { const rs = new StreamResampler(48000); assert.equal(rs.push(new Float32Array(0)).length, 0); assert.equal(rs.push(new Float32Array(1)).length, 0); });
// ---- worklet downmix (mocked AudioWorkletProcessor)
test('worklet averages all channels to mono and survives empty input', async () => {
  const msgs = []; let Cls; globalThis.AudioWorkletProcessor = class { constructor() { this.port = { postMessage: (m) => msgs.push(m) }; } }; globalThis.registerProcessor = (_, c) => (Cls = c);
  (0, eval)(fs.readFileSync(new URL('../frontend/worklet.js', import.meta.url), 'utf8') + '\n;globalThis.__C = Cap;'); const p = new globalThis.__C();
  const L = Float32Array.from([1, 1, 1, 1]), R = Float32Array.from([0, 0, 0, 0]);
  assert.equal(p.process([[L, R]]), true); assert.deepEqual([...msgs[0].d], [0.5, 0.5, 0.5, 0.5]); assert.equal(msgs[0].c, 2);
  p.process([[Float32Array.from([0, 0]), Float32Array.from([0.8, -0.8])]]); assert.ok(Math.abs(msgs[1].d[0] - 0.4) < 1e-6, 'right-only audio must not be lost');
  const n = msgs.length; assert.equal(p.process([[]]), true); assert.equal(p.process([]), true); assert.equal(msgs.length, n);
});
