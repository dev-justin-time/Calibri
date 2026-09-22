// Proof-of-life tests for usecases/marketplace1.html and usecases/service.html
// using headless Chrome + CDP. Generalized from proof_marketplace.mjs.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PORT = 9225;
const DIR = 'C:/Users/dividicus/calibri/Calibri/usecases';

const results = [];
const check = (name, pass, detail = '') =>
  results.push({ name, pass: !!pass, detail: String(detail) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'cbx-chrome-'));
const chrome = spawn(CHROME, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore' });

try {
  for (let i = 0; i < 60; i++) {
    try { const r = await fetch(`http://127.0.0.1:${PORT}/json/version`); if (r.ok) break; } catch {}
    await sleep(250);
  }

  const wsOf = async (file) => {
    const nt = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent('file:///' + DIR + '/' + file)}`, { method: 'PUT' });
    const target = await nt.json();
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    const page = list.find((t) => t.id === target.id) ?? list[0];
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    return ws;
  };

  const makeCtx = () => {
    let mid = 0;
    const pending = new Map();
    const consoleErrors = [];
    const failedRequests = [];
    let send, evalJs;
    const attach = (ws) => {
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
        if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
          consoleErrors.push('[console.error] ' + msg.params.args.map(a => a.value ?? a.description ?? '').join(' ').slice(0, 200));
        }
        if (msg.method === 'Runtime.exceptionThrown') {
          consoleErrors.push('[exception] ' + (msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text).slice(0, 200));
        }
        if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
          consoleErrors.push(`[log.${msg.params.entry.source}] ` + msg.params.entry.text.slice(0, 200));
        }
        if (msg.method === 'Network.loadingFailed') {
          failedRequests.push(`${msg.params.errorText} (${msg.params.type})`);
        }
      };
      send = (method, params = {}) => new Promise((res) => {
        const id = ++mid;
        pending.set(id, res);
        ws.send(JSON.stringify({ id, method, params }));
      });
      evalJs = async (expression) => {
        const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
        if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 300));
        return r.result?.result?.value;
      };
    };
    return { attach, send: (...args) => send(...args), get evalJs() { return evalJs; }, consoleErrors, failedRequests };
  };

  const runPage = async (file, shot, fn) => {
    const ctx = makeCtx();
    const ws = await wsOf(file);
    ctx.attach(ws);
    await ctx.send('Runtime.enable');
    await ctx.send('Log.enable');
    await ctx.send('Network.enable');
    await ctx.send('Page.enable');
    await sleep(2500);
    console.log(`\n=== ${file} ===`);
    const failed = await fn(ctx.evalJs, ctx.send);
    const shotRes = await ctx.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    fs.writeFileSync(shot, Buffer.from(shotRes.result.data, 'base64'));
    check(`${file}: screenshot captured`, fs.existsSync(shot), shot);
    const errs = ctx.consoleErrors.filter(e => e.startsWith('[console.error]') || e.startsWith('[exception]') || e.startsWith('[log.'));
    check(`${file}: zero JS runtime errors`, errs.length === 0, errs.join(' | ').slice(0, 200));
    check(`${file}: zero failed network requests`, ctx.failedRequests.length === 0, ctx.failedRequests.join(' | ').slice(0, 200));
    ws.close();
  };

  // ------------------------------------------------------------------ PAGE 1
  await runPage('marketplace1.html', DIR + '/marketplace1_proof.png', async (evalJs) => {
    // title missing? (audit probe)
    const title = await evalJs('document.title');
    check('AUDIT: page has no <title> element', title === '', JSON.stringify(title));

    const bg = await evalJs(`getComputedStyle(document.body).backgroundColor`);
    check('Tailwind CDN + light theme applied (body bg #ffffff)', bg === 'rgb(255, 255, 255)', bg);

    // seat calculator initial: 25 seats x $114
    const init = await evalJs(`({
      seats: document.getElementById('seatCounterDisplay').innerText,
      po: document.getElementById('poPriceDisplay').innerText,
    })`);
    check('initial: 25 Distributed Seats', init.seats === '25 Distributed Seats', init.seats);
    check('initial: PO $2,850.00 (25 x $114)', init.po === '$2,850.00', init.po);

    // + button: 25 -> 30 seats, $3,420
    await evalJs(`document.querySelectorAll('button[onclick="adjustSeats(5)"]')[0].click()`);
    const plus = await evalJs(`({
      seats: document.getElementById('seatCounterDisplay').innerText,
      po: document.getElementById('poPriceDisplay').innerText,
    })`);
    check('plus: 30 seats', plus.seats === '30 Distributed Seats', plus.seats);
    check('plus: PO $3,420.00 (30 x $114)', plus.po === '$3,420.00', plus.po);

    // slider to max 500, then minus clamp back to 495? minus: 500-5=495 x 114 = 56,430
    await evalJs(`(() => { const s = document.getElementById('seatRange'); s.value = 500; s.dispatchEvent(new Event('input', {bubbles:true})); })()`);
    const maxS = await evalJs(`document.getElementById('seatCounterDisplay').innerText`);
    check('slider max: 500 seats', maxS === '500 Distributed Seats', maxS);
    const maxPo = await evalJs(`document.getElementById('poPriceDisplay').innerText`);
    check('slider max: PO $57,000.00 (500 x $114)', maxPo === '$57,000.00', maxPo);

    // minus clamp probe: push above max via setter then minus should clamp
    await evalJs(`adjustSeats(-5)`);
    const clamp = await evalJs(`document.getElementById('seatCounterDisplay').innerText`);
    check('minus after max: 495 seats', clamp === '495 Distributed Seats', clamp);

    // clamp floor probe: direct call below min
    const floor = await evalJs(`(function(){ for (let i=0;i<120;i++) adjustSeats(-5); return document.getElementById('seatCounterDisplay').innerText; })()`);
    check('floor clamp: never below 5 seats', floor === '5 Distributed Seats', floor);

    // Add-to-PO animation
    const btn0 = await evalJs(`document.getElementById('addPoBtn').innerText`);
    await evalJs(`document.getElementById('addPoBtn').click()`);
    await sleep(400);
    const btnMid = await evalJs(`document.getElementById('addPoBtn').innerText`);
    check('PO button animates to "Added to PO-8902!"', btnMid.includes('Added to PO-8902'), btnMid.trim());
    check('AUDIT: initial PO button label', btn0.includes('Add to Master PO'), btn0.trim());

    // dead-control probes (audit evidence)
    const searchVal = await evalJs(`document.getElementById('enterpriseSearchInput').value`);
    check('AUDIT: search input has prefilled value, no filter wiring', searchVal === 'FLUX.1 Extreme NFE', searchVal);
    const selects = await evalJs(`document.querySelectorAll('select').length`);
    check('AUDIT: 4 facet selects present (wired to nothing)', selects === 4, String(selects));
  });

  // ------------------------------------------------------------------ PAGE 2
  await runPage('service.html', DIR + '/service_proof.png', async (evalJs) => {
    const title = await evalJs('document.title');
    check('page has expected <title>', title === 'Calibrix — Real Logic Reference & Architecture Ledger', title);

    // claim checks against live DOM
    const claim70 = await evalJs(`document.body.innerText.includes('70 / 70') && document.body.innerText.includes('70/70 PASS')`);
    check('AUDIT: page claims 70/70 tests in DOM', claim70, '');

    // structure: 8 domain sections + services + limitations anchors resolve
    const anchors = await evalJs(`['warranty','domain-1','domain-2','domain-3','domain-4','domain-5','domain-6','domain-7','domain-8','services-s1-s8','limitations'].map(id => !!document.getElementById(id))`);
    check('all 11 section anchors resolve', anchors.every(Boolean), anchors.join(','));

    // sub-nav links point at real ids
    const badNav = await evalJs(`[...document.querySelectorAll('a[href^="#"]')].map(a => a.getAttribute('href')).filter(h => h.length > 1 && !document.getElementById(h.slice(1)))`);
    check('sub-nav: no dead anchor links', badNav.length === 0, badNav.join(','));

    // per-domain test badges vs header claim (audit: do they sum to 70?)
    const badgeSum = await evalJs(`[...document.querySelectorAll('section span')].filter(s => /\\d+ TESTS PASS/.test(s.innerText)).map(s => parseInt(s.innerText))`);
    const sum = badgeSum.reduce((a, b) => a + b, 0);
    check('AUDIT: per-domain TESTS badges sum != header claim 70 (inconsistency)', sum !== 70, `sum=${sum} badges=${badgeSum.join('+')}`);

    // feature-count audit: D1 says 14 features but lists F011-F014 on ONE row
    const d1rows = await evalJs(`document.querySelectorAll('#domain-1 tbody tr').length`);
    check('AUDIT: D1 table lists 11 rows for 14 claimed features', d1rows === 11, String(d1rows));

    // fonts
    const fonts = await evalJs(`Promise.all([
      document.fonts.load('400 12px "IBM Plex Sans"'),
      document.fonts.load('400 12px "IBM Plex Mono"'),
      document.fonts.load('400 16px "Material Symbols Outlined"'),
    ]).then(() => ({
      plex: document.fonts.check('400 12px "IBM Plex Sans"'),
      mono: document.fonts.check('400 12px "IBM Plex Mono"'),
      icons: document.fonts.check('400 16px "Material Symbols Outlined"'),
    }))`);
    check('fonts: IBM Plex Sans loaded', fonts.plex, '');
    check('fonts: IBM Plex Mono loaded', fonts.mono, '');
    check('fonts: Material Symbols loaded', fonts.icons, '');

    // static page: no interactive JS at all (audit)
    const inlineScripts = await evalJs(`[...document.querySelectorAll('script')].filter(s => !s.src && s.id !== 'tailwind-config').length`);
    check('AUDIT: zero interactive inline scripts (pure static doc)', inlineScripts === 0, String(inlineScripts));
  });

  // summary
  console.log('\n=== PROOF RESULTS ===');
  for (const r of results) console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}${r.detail ? '  ->  ' + r.detail : ''}`);
  const failed = results.filter(r => !r.pass).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exitCode = failed ? 1 : 0;
} finally {
  chrome.kill();
  await sleep(300);
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
}
