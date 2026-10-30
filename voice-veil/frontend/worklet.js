// Capture processor: averages ALL input channels to mono (tab/system audio is often stereo) and posts {d, c}.
class Cap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0];
    if (!ch || !ch.length || !ch[0] || !ch[0].length) return true;
    const n = ch[0].length, m = new Float32Array(n);
    for (let c = 0; c < ch.length; c++) { const a = ch[c]; for (let i = 0; i < n; i++) m[i] += a[i] / ch.length; }
    this.port.postMessage({ d: m, c: ch.length }, [m.buffer]);
    return true;
  }
}
registerProcessor('cap', Cap);
