# OpenWatch Deployment — One Binary, Everywhere, Without Telling Anyone Who You Are

This deploys the daily audit (``platform_watch``) as a locale-aware static site + optional stealth capture sidecar. You publish; platforms can't trivially tag the probe.

## 10 Locales — Coverage & Cultural Names

| # | Code | Name | Meaning | Language / Native | Dir | Region | Speakers (M) | Why this name |
|---|------|------|---------|-------------------|-----|--------|--------------|----------------|
| 1 | `en` | **Beacon** | beacon / signal fire | English — English | ltr | Global | 1450 | *beacn* — watchtower light. Root of French/German equivalents. |
| 2 | `zh-Hans` | **Mingjing** | bright mirror | Mandarin — 中文 | ltr | East Asia | 1120 | 明镜 — reflects without distortion (Tang poetry, legal tradition). |
| 3 | `es-419` | **Vela** | vigil / candle watch | Spanish — Español | ltr | LatAm + Spain | 560 | *hacer vela* — to keep vigil. |
| 4 | `hi` | **Drishti** | sight / clear vision | Hindi — हिन्दी | ltr | South Asia | 610 | दृष्टि — discerning, not just seeing. |
| 5 | `ar` | **Noor** | light | Arabic — العربية | **rtl** | MENA | 400 | نور — shared Urdu/Persian/Swahili, one syllable. |
| 6 | `pt-BR` | **Claro** | clear / bright | Portuguese — Português | ltr | Brazil+Lusophone | 265 | *claro* — a clean record. |
| 7 | `fr` | **Éclaire** | it illuminates | French — Français | ltr | W. Africa+Europe | 310 | *éclairer* — to clarify, active verb. |
| 8 | `ru` | **Dozor** | watch / patrol | Russian — Русский | ltr | E. Europe+Central Asia | 255 | Дозор — night watch. |
| 9 | `ja` | **Akashi** | revealing light / proof | Japanese — 日本語 | ltr | Japan | 125 | 明かし — to disclose. |
| 10 | `bn` | **Alo** | light | Bengali — বাংলা | ltr | Bengal | 275 | আলো — warm daily word. |

~4.7B L1 + ~1.5B L2 at launch. Phase 2: `id-ID`, `tr-TR`, `ur-PK`, `vi-VN`, `it-IT`.

Rendering: per-locale HTML under ``openwatch/<code>/<date>.html`` + ``index.html`` mirror, ``lang`` + ``dir`` set, language switcher in header. JSON stays single-source (``openwatch/<date>.json``) with ``locale`` field.

## Leverage: Obscura (h4ckf0r0day/obscura)

Apache-2.0, Rust, V8, no Chromium, CDP-compatible. Why it matters here:

- **Evidence, not the judge.** Deterministic grading stays offline; Obscura only fetches/renders so humans can verify the grade (PNG/PDF via native engine).
- **Lightweight**: 30 MB / 70 MiB / 85 ms / instant start — 10-locale capture doesn't need 10 Chromes.
- **Anti-detect built-in**: per-session fingerprint randomization, `isTrusted=true`, `webdriver=undefined`, `Function.toString()→[native code]`, tracker blocking 3,520 domains. Enable with ``--stealth``.
- **Drops into your stack**: ``obscura serve --port 9222`` speaks CDP; `puppeteer-core` / `playwright-core` / MCP `obscura mcp` all work.
- **Anonymity-friendly**: ``--proxy socks5://...`` per locale, `--stealth`, `OBSCURA_CDP_TOKEN` auth, `127.0.0.1:9222` binding. Pair with residential pools (NodeMaven/ProxyEmpire/Masklabs use their own IPs — see Obscura README codes).

This repo wraps it in ``calibrix/doberwatch/platform_watch/obscura_adapter.py`` — builders return inspectable argv, degrade cleanly when the binary is absent (tests/CI never hard-depend on it), and ``evidence_capture_plan()`` gives PNG+PDF capture for each locale bundle.

## Competitors — Who We Watch, Per Region

OpenWatch is **not** a model. It audits the six platforms people actually rent:

| Platform | Owner | Why it competes (on the behaviours C1–C6 score) | Loudest where |
|----------|-------|--------------------------------------------------|---------------|
| **ChatGPT / GPT-4o / mini (“Luna”-class wrappers)** | OpenAI | System-prompt > user-prompt; non-deterministic; history rent; subscription vs cure. | en, es-419, pt-BR, fr |
| **Claude** | Anthropic | Preachy refusals; moralizing over neutral help (C3 delta). | en, hi, fr |
| **Gemini** | Google | Hallucination with confidence; policy preemption stance. | hi, ja, bn, pt-BR |
| **Meta AI / Llama** | Meta | Open-weight but platform lock (hosting rent). | pt-BR, fr (Africa), es-419 |
| **Grok** | xAI | Claims anti-bias; measures well on C3, uneven on C4/C5. | en, ru |
| **Mistral / Le Chat** | Mistral | EU-hosted alternative; strong on ownership (C5) pitch. | fr, ru, en |

Regionally, the strongest *deployment* competition is C5 (ownership): providers who let you run a model locally win on “can I keep the data and rerun tomorrow without a bill?” — which is precisely what OpenWatch caches as its proof.

## Anonymity — Accessing Platforms as an Unknown Person (What This Does and Doesn't Do)

**What it does:**
- **Residential-IP per locale**: set ``OBSCURA_PROXY_EN``, ``OBSCURA_PROXY_ZH``, … (or global ``OBSCURA_PROXY``) to socks5/http proxies you control. The runner picks per-locale proxy automatically and adds ``--stealth --proxy`` to every ``obscura fetch/scrape``.
- **Untagged probe**: ``--stealth`` randomizes GPU/screen/canvas/audio, spoofs `userAgentData` (Chrome 145), hides webdriver — platforms see a normal user, not “benchmark”.
- **CDP auth**: ``OBSCURA_CDP_TOKEN`` (``openssl rand -hex 32``) — CDP WebSocket is token-gated and bound to ``127.0.0.1:9222`` only.
- **Traffic shaping**: ``obscura scrape`` with low ``--concurrency``, jitter, and ``--wait-until networkidle0`` looks bursty-human, not scanner.
- **Your data stays yours**: reports write to your ``ResponseCache`` at ``openwatch/cache.json`` (or any disk/cloud path) under ``local_only``; no telemetry, no call-home.

**What it doesn't do (and why):**
- It does **not** bypass login/auth, steal private prompts, or scrape logged-in content. Login forms are driven via CDP only on your own test accounts if you provide them.
- It does **not** claim to make you legally anonymous. Your proxy provider sees you. Use Tor (``socks5://127.0.0.1:9050``) only if you understand exit-node risks.
- It does **not** hide you from your own hosting provider. Self-host on hardware you control, or accept that your VPS host knows it hosts you.

Checklist ``ANONYMITY_CHECKLIST`` in ``obscura_adapter.py`` is the runbook; tests pin it.

## Deployment

### 1) Static site (no obscura required)

```bash
# from repo root
python -c "
import sys; sys.path.insert(0, 'calibrix')
from calibrix.doberwatch.platform_watch.prompts import PACK_HASH
from calibrix.doberwatch.platform_watch.blog import build_openwatch_report, write_openwatch_reports_all_locales
report = build_openwatch_report('2026-09-24', PACK_HASH, platforms, ledger_digest='...')
paths = write_openwatch_reports_all_locales(report, 'openwatch')
print(paths)
"
# -> openwatch/2026-09-24.json, openwatch/2026-09-24.html, openwatch/<locale>/2026-09-24.html
# Host on: GitHub Pages / Netlify free / Cloudflare Pages — just static files.
```

### 2) With Obscura evidence sidecar (evidence screenshots/PDFs)

```yaml
# docker-compose.yml — add to this repo or deploy separately
services:
  obscura:
    image: h4ckf0r0day/obscura
    restart: unless-stopped
    ports: ["127.0.0.1:9222:9222"]
    environment:
      OBSCURA_CDP_TOKEN: ${OBSCURA_CDP_TOKEN:?set with openssl rand -hex 32}
      OBSCURA_ALLOW_PRIVATE_NETWORK: "0"   # allow only for file:// capture
      OBSCURA_SCRIPT_DEADLINE_MS: "30000"
    # distroless cc:nonroot, uid 65532 — keep storage writable by 65532
    # volumes:
    #   - ./openwatch:/storage:ro

  openwatch:
    build: .  # or python:3.12-slim + pip install -e ./calibrix
    volumes: ["./openwatch:/app/openwatch"]
    environment:
      OBSCURA_CDP_URL: ws://obscura:9222/devtools/browser
      OBSCURA_PROXY_EN: ${OBSCURA_PROXY_EN:-}
      OBSCURA_PROXY_AR: ${OBSCURA_PROXY_AR:-}
      # ... per-locale proxies
    command: python scripts/run_openwatch_daily.py  # caller supplies prompt→platform→grade loop
    depends_on: [obscura]
```

Host Obscura install (non-Docker):
```bash
curl -LO https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-x86_64-linux.tar.gz
tar xzf obscura-x86_64-linux.tar.gz
./obscura --help && ./obscura fetch https://example.com --dump text --stealth | head
./obscura serve --port 9222 --stealth  # CDP at ws://127.0.0.1:9222
```

Capture evidence (after nightly grade writes HTML):
```python
from calibrix.doberwatch.platform_watch.obscura_adapter import evidence_capture_plan
plan = evidence_capture_plan("openwatch/2026-09-24.html", "openwatch/evidence", locale="ar")
# plan["fetch_png_cmd"] -> ["obscura", "--stealth", "fetch", "file:///...", "--screenshot", ".../2026-09-24.ar.png"]
```

### 3) Nightly runner (GitHub Actions sketch)

```yaml
# .github/workflows/openwatch.yml
on: { schedule: [{ cron: "0 2 * * *" }], workflow_dispatch: {} }
jobs:
  audit:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -e ./calibrix
      - run: python -m unittest discover -s calibrix/tests  # gates CLEANROOM
      - run: python scripts/run_openwatch_daily.py  # -> openwatch/<date>.{json,html}
      - uses: actions/upload-artifact@v4
        with: { name: openwatch, path: openwatch/ }
```

## Verify

```bash
cd calibrix
python -m unittest tests.test_platform_watch -v
python -m unittest tests.test_locales_obscura -v
python -m unittest discover -s tests  # 360+ tests
python -m calibrix.studio.ledger scan
python -m calibrix.studio.ledger build
```

## Honest bounds

- Lexical verification needs shared terms of art to clear 0.45 — a perfect synonym with no shared words scores low (stated, not silent).
- Obscura's rendering is independent — long-tail CSS/compositor effects may differ from Chromium (see obscura README).
- ``evidence_status`` stays ``simulation/reference_only`` until a real holdout passes; do not market grades as certification.
