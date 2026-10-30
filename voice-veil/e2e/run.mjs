// Browser E2E for Voice Veil. Usage: node run.mjs <wav-as-fake-microphone> [url]
// Uses Chrome's fake-audio-capture (a WAV file stands in for the microphone). This exercises the real
// getUserMedia -> AudioWorklet -> resampler -> WebSocket path, but NOT real microphone hardware.
import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';
const wav = process.argv[2] ? path.resolve(process.argv[2]) : null, url = process.argv[3] || 'http://127.0.0.1:8765';
if (!wav || !fs.existsSync(wav)) { console.error('Usage: node run.mjs <wav file> [url]   (file not found: ' + process.argv[2] + ')'); process.exit(2); }
let exe = process.env.CHROME_PATH;
if (!exe) for (const p of ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', 'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe']) if (fs.existsSync(p)) { exe = p; break; }
if (!exe) { try { const c = (await import('@sparticuz/chromium')).default; exe = await c.executablePath(); } catch (e) { console.error('Set CHROME_PATH to chrome.exe or msedge.exe'); process.exit(2); } }
const results = []; const ok = (name, pass, info = '') => { results.push({ name, pass, info }); console.log((pass ? 'PASS ' : 'FAIL ') + name + (info ? ' :: ' + info : '')); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const browser = await puppeteer.launch({ executablePath: exe, headless: 'new', args: ['--no-sandbox', '--disable-gpu', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  `--use-file-for-fake-audio-capture=${wav}`, '--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage(); await page.setViewport({ width: 1280, height: 1500 });
const consoleErrors = [], failed = []; page.on('console', m => { if (['error', 'warning'].includes(m.type())) consoleErrors.push(m.type() + ': ' + m.text()); });
page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message)); page.on('requestfailed', r => failed.push(r.url()));
const cdp = await page.createCDPSession(); await cdp.send('Network.enable'); const sent = [], recv = [];
cdp.on('Network.webSocketFrameSent', e => { if (e.response.opcode === 2) sent.push({ t: Date.now(), len: Buffer.from(e.response.payloadData, 'base64').length }); });
cdp.on('Network.webSocketFrameReceived', e => { try { recv.push({ t: Date.now(), ...JSON.parse(e.response.payloadData) }); } catch (_) {} });
const wsClosed = []; cdp.on('Network.webSocketClosed', e => wsClosed.push(Date.now()));
await fetch(url + '/api/events', { method: 'DELETE' });   // start from an empty event log so the log checks below are deterministic
let dialogs = 0; page.on('dialog', async d => { dialogs++; await d.accept(); });
await page.goto(url, { waitUntil: 'networkidle0' });
ok('page loads, title', (await page.title()).includes('Voice Veil'));
ok('prototype banner shows model + synthetic warning', /SYNTHETIC|synthetic/.test(await page.$eval('#minfo', e => e.textContent)), await page.$eval('#minfo', e => e.textContent));
ok('Stop disabled before start', await page.$eval('#stop', e => e.disabled));
// ---- start monitoring
await page.click('#go'); await sleep(1500);
ok('no error banner after start', (await page.$eval('#err', e => e.style.display)) !== 'block', await page.$eval('#err', e => e.textContent));
ok('Start disabled / Stop enabled while running', (await page.$eval('#go', e => e.disabled)) && !(await page.$eval('#stop', e => e.disabled)));
const devs = await page.$$eval('#dev option', o => o.map(x => x.textContent)); ok('device list populated', devs.length > 0 && !/No microphone/.test(devs[0]), JSON.stringify(devs));
await sleep(9000);
ok('binary audio frames reach WebSocket', sent.length > 20, `${sent.length} frames`);
const sizes = [...new Set(sent.map(s => s.len))]; ok('frames are float32 (size % 4 == 0)', sent.every(s => s.len % 4 === 0), 'sizes ' + sizes.slice(0, 6).join(','));
const span = (sent.at(-1).t - sent[0].t) / 1000, samples = sent.reduce((a, s) => a + s.len / 4, 0), rate = samples / span;
ok('effective sample rate ~16 kHz (±6%)', Math.abs(rate - 16000) < 960, rate.toFixed(0) + ' Hz measured over ' + span.toFixed(1) + ' s');
const dec = recv.filter(r => r.state); ok('backend returns decisions', dec.length >= 6, `${dec.length} decisions`);
const diag = recv.filter(r => r.type === 'diag').at(-1); ok('backend reports audio diagnostics', !!diag, JSON.stringify(diag || {}));
ok('backend-measured sample rate within 8% of 16 kHz', diag && diag.rate_ok === true, diag && String(diag.rate_hz));
ok('backend sees non-silent audio (rms>0.005, not clipped)', diag && diag.rms > 0.005 && diag.clipped_pct < 1, diag && `rms ${diag.rms} peak ${diag.peak} clip ${diag.clipped_pct}%`);
ok('capture info: ctx rate + channel count reported by browser', diag && diag.client.ctx_rate > 0 && diag.client.input_channels >= 1, JSON.stringify(diag && diag.client));
ok('no malformed frames', diag && diag.dropped_frames === 0);
ok('UI shows what the backend receives', /kHz/.test(await page.$eval('#diag', e => e.textContent)), await page.$eval('#diag', e => e.textContent));
const withT = dec.filter(d => d.latency_ms != null); const q = a => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
console.log('LATENCY(browser run, ms): model-compute median', q(withT.map(d => d.latency_ms)), '| features', q(withT.map(d => d.feature_ms)), '| infer', q(withT.map(d => d.infer_ms)), '| server receipt->reply', q(withT.map(d => d.server_ms)));
const lastDec = dec.filter(d => d.trust != null).at(-1); ok('score is derived (0..1) from audio', lastDec && lastDec.trust >= 0 && lastDec.trust <= 1, JSON.stringify(lastDec && { state: lastDec.state, trust: lastDec.trust }));
const wavePx = await page.$eval('#wave', c => { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; }); ok('waveform canvas is drawing', wavePx > 500, wavePx + ' px');
ok('timeline has data', await page.$eval('#tl', c => { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n > 300; }));
ok('state text shown', (await page.$eval('#state', e => e.textContent)).length > 0, await page.$eval('#state', e => e.textContent));
const e2eTxt = await page.$eval('#e2e', e => e.textContent); ok('UI shows browser-measured chunk->verdict latency', /^\d+$/.test(e2eTxt), e2eTxt + ' ms'); console.log('UI latency line:', await page.$eval('#lat2', e => e.textContent));
// ---- sensitivity
await page.$eval('#sens', e => { e.value = 100; e.dispatchEvent(new Event('input')); }); await sleep(800);
ok('sensitivity 100 -> threshold 0.70 in UI', (await page.$eval('#thr', e => e.textContent)) === '0.70');
await sleep(1500); const thrSrv = recv.filter(r => r.threshold != null).at(-1)?.threshold; ok('backend adopted new threshold', thrSrv === 0.7, String(thrSrv));
// ---- warning display: the UI must mirror what the backend decided (EXPECT=high -> the HIGH warning must really appear)
const seen = new Set(); let highUI = false;
for (let i = 0; i < 16; i++) { const st = await page.$eval('#state', e => e.textContent); seen.add(st);
  if (st === 'High risk' && await page.$eval('.hero', e => e.classList.contains('HIGH'))) highUI = true; await sleep(500); }
const backendHigh = recv.filter(r => r.state === 'HIGH').length;
console.log('UI states seen:', [...seen].join(' | '), '| backend HIGH decisions:', backendHigh);
ok('UI shows High-risk warning iff backend sent HIGH', highUI === (backendHigh > 0), `ui=${highUI} backendHigh=${backendHigh}`);
if (process.env.EXPECT === 'high') {
  ok('EXPECT=high: warning appeared (hero class HIGH, advice text)', highUI && /Hang up and call back/.test(await page.$eval('#advice', e => e.textContent)));
  const rows = await page.$$eval('#log tr', trs => trs.map(tr => [...tr.children].map(c => c.textContent.trim())));
  const last = rows.at(-1) || [];
  ok('EXPECT=high: HIGH event row = time + numeric score only', last.length === 2 && /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(last[0]) && /^0\.\d+$/.test(last[1]), JSON.stringify(rows.slice(-2)));
  const evs = await (await fetch(url + '/api/events')).json(); ok('EXPECT=high: server log holds {time,score} only, one entry per episode', evs.length >= 1 && evs.every(e => Object.keys(e).sort().join() === 'score,time') && evs.length <= 3, JSON.stringify(evs));
}
await page.$eval('#sens', e => { e.value = 0; e.dispatchEvent(new Event('input')); }); await sleep(2500);
ok('sensitivity 0 -> threshold 0.30 in UI and backend', (await page.$eval('#thr', e => e.textContent)) === '0.30' && recv.filter(r => r.threshold != null).at(-1)?.threshold === 0.3);
await page.$eval('#sens', e => { e.value = 100; e.dispatchEvent(new Event('input')); }); await sleep(1500);
await page.screenshot({ path: process.env.SHOT || 'shot_running.png', fullPage: true });
const hi = recv.filter(r => r.state === 'HIGH').length; console.log('HIGH decisions:', hi, 'states:', JSON.stringify(Object.fromEntries([...new Set(dec.map(d => d.state))].map(s => [s, dec.filter(d => d.state === s).length]))));
// ---- stop
await page.click('#stop'); await sleep(1200);
ok('Stop re-enables Start', !(await page.$eval('#go', e => e.disabled)) && await page.$eval('#stop', e => e.disabled));
const live = await page.evaluate(() => (typeof stream === 'undefined' || stream === null) ? 0 : stream.getTracks().filter(t => t.readyState === 'live').length); ok('mic tracks released after Stop', live === 0, live + ' live tracks');
const cleaned = await page.evaluate(() => ({ ws: typeof ws === 'undefined' ? 'n/a' : ws, ctx: typeof ctx === 'undefined' ? 'n/a' : ctx }));
ok('AudioContext + WebSocket references released after Stop', cleaned.ws === null && cleaned.ctx === null, JSON.stringify(cleaned));
ok('WebSocket closed on the wire after Stop (server session ended)', wsClosed.length >= 1, wsClosed.length + ' close event(s)');
const n0 = sent.length; await sleep(1500); ok('no frames sent after Stop', sent.length === n0);
// ---- restart (resource cleanup / duplicate loops)
await page.click('#go'); await sleep(2500); const a = sent.length; await sleep(3000); const perSec = (sent.length - a) / 3; ok('restart works; single send loop (frames/s sane)', perSec > 3 && perSec < 14, perSec.toFixed(1) + ' frames/s');
await page.click('#stop'); await sleep(800);
// ---- rapid start/stop x4 (regression: worklet messages arriving after stop threw 'null.sampleRate')
for (let i = 0; i < 4; i++) { await page.click('#go'); await sleep(700); await page.click('#stop'); await sleep(250); }
await sleep(1000);
// ---- demo mode
await page.click('#df'); await sleep(2500); ok('demo shows DEMO tag', await page.$eval('#demoTag', e => e.classList.contains('show')));
ok('demo result text labelled DEMO', /DEMO/.test(await page.$eval('#res', e => e.textContent)), await page.$eval('#res', e => e.textContent));
// ---- events clear
const before = await page.$$eval('#log tr', r => r.length); await page.click('#clrlog'); await sleep(800); const txt = await page.$eval('#log', e => e.textContent); ok('clear all events (confirm dialog accepted)', dialogs >= 1 && /No high-risk events/.test(txt), `rows before ${before}`);
// ---- responsive
await page.setViewport({ width: 390, height: 900 }); await sleep(300); ok('mobile width: no horizontal scroll', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), await page.evaluate(() => document.documentElement.scrollWidth + ' vs ' + window.innerWidth));
await page.screenshot({ path: (process.env.SHOT || 'shot_running.png').replace('.png', '_mobile.png') });
ok('no console errors / page errors', consoleErrors.length === 0, consoleErrors.slice(0, 4).join(' | '));
ok('no failed network requests', failed.length === 0, failed.join(','));
await browser.close();
const f = results.filter(r => !r.pass).length; console.log(`\n${results.length - f}/${results.length} checks passed`); process.exit(f ? 1 : 0);
