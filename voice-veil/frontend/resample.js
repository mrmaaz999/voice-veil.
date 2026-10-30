// Streaming area-average resampler (any input rate -> 16 kHz). Keeps fractional phase between pushes,
// so non-integer ratios (e.g. 44100 -> 16000) do not drift or alias with floor() boundaries.
class StreamResampler {
  constructor(inRate, outRate = 16000) { this.r = inRate / outRate; this.buf = new Float32Array(0); this.phase = 0; }
  push(x) {
    const b = new Float32Array(this.buf.length + x.length); b.set(this.buf); b.set(x, this.buf.length);
    const r = this.r, n = Math.floor((b.length - this.phase) / r + 1e-9);
    if (n <= 0) { this.buf = b; return new Float32Array(0); }
    const cs = new Float64Array(b.length + 1); for (let i = 0; i < b.length; i++) cs[i + 1] = cs[i] + b[i];
    const C = t => { const k = Math.floor(t); return cs[k] + (k < b.length ? (t - k) * b[k] : 0); };
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) { const t0 = this.phase + i * r; out[i] = (C(t0 + r) - C(t0)) / r; }
    const used = this.phase + n * r, k = Math.floor(used);
    this.buf = b.slice(k); this.phase = used - k; return out;
  }
}
if (typeof module !== 'undefined') module.exports = { StreamResampler };
