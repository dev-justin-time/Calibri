# weregoodboss.md — `usecases/marketplace.html` Proof & Audit

**Date:** 2026-09-21
**Subject:** `usecases/marketplace.html` — "Calibrix Yield & Compute Arbitrage Exchange (CALX-MKT)" (766 lines, static mock dashboard)
**Method:** Full file read + headless Chrome (CDP) execution probe. Repro script: `usecases/proof_marketplace.mjs` · Visual evidence: `usecases/marketplace_proof.png`

---

## 1. Proof that it works

`usecases/proof_marketplace.mjs` launches real headless Chrome (`--headless=new`, CDP over WebSocket), loads the file via `file://`, and drives it like a user. **23/24 checks passed, 0 JS runtime errors, 0 console errors, 0 failed network requests.**

### What was proven, with live-verified evidence

| # | Check | Evidence captured |
|---|-------|-------------------|
| 1 | Page loads, correct title | `Calibrix Yield & Compute Arbitrage Exchange (CALX-MKT)` |
| 2 | Tailwind CDN + dark theme applied | body bg computed = `rgb(8, 9, 11)` (`surface-dim`) |
| 3 | Inline script executes on load (2.5M vol, 64% yield) | gross `$38,500.00`, post `$13,860.00`, net `+$24,640.00 /mo`, buyer `+$18,480.00`, seller `+$6,160.00` — all exact |
| 4 | ARB LATCH row 2 wires table → calculator | label `SD 3.5 Large (vae_post_quant)`, post-cost recalcs `$21,367.50`, savings `+$17,132.50 /mo` |
| 5 | Slider drives recalculation (5M) | gross `$77,000.00`, post `$42,735.00`, savings `+$34,265.00 /mo` |
| 6 | Execute button full animation cycle | `LATCHING SMART CONTRACT...` → `ARBITRAGE ACTIVE (BLOCK #984211)` → restores |
| 7 | Fonts load | IBM Plex Mono + Material Symbols confirmed loaded; no font network failures |
| 8 | Full-page screenshot | `usecases/marketplace_proof.png` (217 KB) |

Math independently verified by hand: `2,500,000 × $0.0154 = $38,500.00`; `38,500 × (1−0.64) = $13,860.00`; savings `$24,640.00`; split 75/25 → `$18,480.00 / $6,160.00`. **Every rendered number is arithmetically correct.**

One initial check "failed" (volume label expected `requests`, got `REQUESTS`): that is CSS `text-transform: uppercase`, not a code bug — test expectation corrected.

---

## 2. Audit — missing or bad logic

### 🔴 Confirmed bugs (reproduced live) — B1 & B2 FIXED 2026-09-21, see §2a

**B1. `selectArbPair()` ignores the `price` argument — calculator can never reflect row pricing.** ✅ FIXED
The handler received the Ask price (`12.0`, `8.0`, `15.0`, `18.0`) but never assigned it; `currentCostPerReq` was a hardcoded constant `0.0154`. Probe had shown: called with price `100.0` then `1.0` → gross stayed identical.
Fix applied: `currentCostPerReq = price / 1000.0` inside `selectArbPair` (ask is quoted per seat/1k-req scale), guarded by `isFinite(price) && price > 0`; default baseline aligned to the default FLUX row (`12.0 / 1000.0`).

**B2. Yield badge loses its `+` prefix on latch.** ✅ FIXED
Initial HTML showed `+64.0% Yield`; after latch it degraded to `44.5% Yield`.
Fix applied: badge text is now `'+' + yieldPct + ' Yield'` — matches the order-book column format.

**B3. Execute button hardcodes `BLOCK #984211`.**
`triggerExecution()` fires no request and fakes a Merkle block number that collides with the static "Merkle Block Audit Stream" (which tops out at `#984210`). It's theater posing as state — acceptable for a mock, but the number should at least be `984211 = 984210 + 1` by construction, or clearly labeled SIMULATION.

### 🟡 Missing logic (declared but absent)

**M1. Zero state sync with the real marketplace backend.** ✅ FIXED 2026-09-21 — see §2b
The order book was fully hardcoded while the real UC-4 server (`seller.py serve`, port 8700) held live data in `store.json` that never appeared here. **Fix:** new `GET /api/listings` JSON endpoint on `MarketplaceServer` (CORS-enabled, spec-free) + dashboard hydration with graceful offline fallback to the demo rows.

**M2. Filter tabs are dead buttons.** "ALL ARB CONTRACTS / DIFFUSION / LLM PREFILL" and the fleet `<select>` have no handlers; the select changes nothing (no cost-model delta per cluster).

**M3. Nav links (`href="#"`), "Post Order / Kernel", "Explore Full Ledger", header icon buttons** — all no-ops. No routing, no modals.

**M4. No persistence / reset.** Latch state dies on reload; the slider's default isn't restored; there's no way to clear the "ARBITRAGE ACTIVE" state other than waiting out the 3 s timer.

**M5. Accessibility gaps:** icon-only buttons lack `aria-label`; the status dot / "LIVE SYNC" ping conveys state by color alone; table has no `<caption>`/`scope` semantics; slider lacks an `aria-valuetext` with the formatted number.

**M6. Tailwind via CDN warning** (console): fine for a mock, will block production use — needs a build step before this ships anywhere real.

### 🟢 Non-issues (checked and clean)

- No JS exceptions, no console errors, no 404s, no CORS problems (single inline script, two font/CDN origins).
- No `onclick` handler throws even with edge inputs (slider min/max, repeated latches, double-click execute).
- Slider bounds respected: `min=500000 step=250000 max=10000000` — no negative or NaN outputs possible via the UI.
- The 75/25 split bar widths are hardcoded `75% / 25%` and match the JS split constants — they can't drift apart today, but see B1-adjacent risk: if someone makes the split configurable in JS, the bar must be updated too.

---

## 2a. Fix log (2026-09-21, post-audit)

**Changes to `usecases/marketplace.html`:**
- `currentCostPerReq` default: `0.0154` → `12.0 / 1000.0` (baseline = default FLUX.1-dev row, so the preselected panel and the numbers agree).
- `selectArbPair()` now adopts the row's ask price as the unit cost (`price / 1000.0`, guarded against non-positive/non-finite).
- Badge: `'+' + yieldPct + ' Yield'` (B2).
- Static HTML initial values re-aligned to the corrected economics (2.5M × $12/1k @ 64% yield): gross `$30,000.00`, post `$10,800.00`, net `+$19,200.00 /mo`, buyer `+$14,400.00`, seller `+$4,800.00`.

**Re-verification (same headless-Chrome harness): 25/25 checks passed, 0 JS errors, 0 failed network requests, exit code 0.** Key new checks:
- `FIX B1: latch adopts ask $8.00/1k, gross = $20,000.00` → PASS
- `FIX B1: different asks yield different gross` (probe: $100 vs $1 ask → `$250,000.00` vs `$2,500.00`) → PASS
- `FIX B2: latch badge KEEPS the + prefix` (`+44.5% Yield`) → PASS
- Hand-verified: 2.5M × $8/1k = $20,000 gross; ×(1−0.445) = $11,100 post; savings $8,900; 75/25 = $6,675/$2,225.

The screenshot at `usecases/marketplace_proof.png` was regenerated from the fixed page.

## 2b. M1 fix — live data wiring (2026-09-21)

**Server (`calibrix/calibrix/marketplace/server.py`):**
- New `GET /api/listings` route → `MarketplaceServer.handle_api_listings()` returning `{listings: [...], stats: {...}}` from the same `JsonStore` the seller console uses.
- Read-only + privacy-preserving: **no `kernel_spec`, no license material** — only a module-family hint (`vector: "attn"`), price, description, scoring, activity, and per-listing order/revenue aggregation from the order ledger. Verified by test `test_api_listings_shape_and_privacy` (asserts the raw spec bytes never appear in the response).
- Per-listing `orders`/`revenue_cents` and global `stats.gross_usd` aggregate straight from `store.json` — dashboards can't drift from the ledger.
- CORS (`Access-Control-Allow-Origin: *`) + `Cache-Control: no-store` so the file:// dashboard can read it.
- `_send()` extended with a `headers` dict (backward compatible).
- Tests: 19/19 pass in `calibrix/tests/test_marketplace.py` (2 new: shape/privacy + revenue aggregation through a full mock webhook sale).

**Dashboard (`usecases/marketplace.html`):**
- On load, fetches `/api/listings` (3 s abort timeout) and re-renders the order-book `<tbody>` from live listings: title, pool id (`LIVE · <listing_id>`), vector chip, derived yield from scoring, ask price, units sold + realized revenue, ARB LATCH wired through the same `selectArbPair` (B1/B2 fixes hold with live data).
- All dynamic content HTML-escaped (`esc()`); footer reports the live source via `#bookStatus`.
- Graceful degradation: if the UC-4 server is offline, the original demo rows remain (fallback verified in the harness).
- Live end-to-end proof: real `store.json` (`product-kernel-v1`, $12.00, 1 order) renders as a row; latching it yields gross `$30,000.00`, post `$0.00`, savings `+$30,000.00 /mo` — derived, not hardcoded.

**Final verification: 28/28 checks, 0 JS errors, 0 failed network requests, exit 0.**

---

## 3. Verdict

**Works: yes.** It renders pixel-perfect, its JavaScript runs clean, its arithmetic is exact, and every interactive element it claims to wire is wired and verified. Screenshot proof is on disk next to the file.

**Update (2026-09-21):** B1 and B2 are fixed and re-verified — the Ask Price column now genuinely drives the calculator. **M1 is closed too:** the order book hydrates from the live UC-4 backend via `GET /api/listings`, verified end-to-end against the real `store.json` (28/28 checks). Remaining gaps are cosmetic/dead-control items (M2–M6) and B3; the headline stat cards and ticker are still static.

**Recommended next step:** wire `selectArbPair` to `fetch('http://127.0.0.1:8700/...')`-style JSON (extend `MarketplaceServer.render_storefront` with a small `/api/listings` endpoint) so the order book reflects `store.json` — that single change converts this from brochure to dashboard.

---

*Repro: `node usecases/proof_marketplace.mjs` (requires local Chrome; exit code 0 = all checks pass).*

---

# §4 — `usecases/marketplace1.html` Proof & Audit

**Subject:** "Calibrix Enterprise Registry" — B2B kernel catalog + seat/PO calculator (789 lines, light theme, static mock).
**Method:** headless Chrome (CDP) via `usecases/proof_more.mjs` · Evidence: `usecases/marketplace1_proof.png`

## Proof that it works

**17/17 page checks passed, 0 JS runtime errors, 0 failed network requests.**

| Check | Evidence |
|---|---|
| Tailwind + light theme applied | body bg `rgb(255, 255, 255)` |
| Seat calculator initial state | `25 Distributed Seats`, PO **$2,850.00** = 25 × $114 ✓ |
| `+` button (adjustSeats(+5)) | 30 seats → **$3,420.00** ✓ |
| Slider to max (500) | 500 seats → **$57,000.00** ✓ |
| Minus after max | 495 seats ✓ |
| **Floor clamp probe** (120 × adjustSeats(-5)) | never below **5 seats** — clamp logic correct ✓ |
| Add-to-PO button animation | → "Added to PO-8902!" → restores ✓ |
| Fonts (Plex Sans/Mono, Material Symbols) | all loaded ✓ |

## Audit findings

**🔴 M1b. No `<title>` element at all.** `document.title === ""` — browser tab shows the file path. One-line fix.

**🟡 M2b. Dead controls:** search input (prefilled `FLUX.1 Extreme NFE`, no filter wiring), all 4 taxonomy `<select>` facets, "Query" button, every "Add to Master PO" per-item button (only the inspector one animates), Manifest/Audit Dossier/Lab Report/Test Suite/Spec Sheet buttons, "Execute Master Order PO-8902", DPA download, all nav links.

**🟡 M3b. Static catalog:** 5 kernel cards are hardcoded; zero data source (not even the UC-4 API wired into marketplace.html).

**🟢 Non-issues:** seat math exact at every probed point incl. clamps; no JS errors; no broken anchors (only `href="#"` stubs); button animation state restores cleanly.

**Verdict:** works; polished mock. One trivial fix (title), otherwise same class of gaps as marketplace.html pre-wiring.

---

# §5 — `usecases/service.html` Proof & Audit

**Subject:** "Real Logic Reference & Architecture Ledger" — F001–F100 feature taxonomy + S1–S8 commercial services (1274 lines, pure static document, zero interactive JS).
**Method:** headless Chrome (CDP) via `usecases/proof_more.mjs` + repo ground-truth checks · Evidence: `usecases/service_proof.png`

## Proof that it works

**13/13 page checks passed, 0 JS errors, 0 failed network requests.** All 11 section anchors resolve; every sub-nav `#anchor` link points at an existing id (no dead links); fonts load; title correct.

## Audit findings — the page makes *verifiable claims*, so they were verified

**🔴 C1. "70/70 Unit Tests Passing" contradicts the repo: 75 tests actually pass.** `python -m unittest tests.test_studio` → **Ran 75 tests, OK**. The page under-claims. Worse: internally inconsistent — the per-domain "N TESTS PASS" badges (12+11+9+10+9+9+11+6) **sum to 77**, not 70. Three different numbers floating in one document (70 claimed in header + footer, 77 in badges, 75 real).

**🔴 C2. Claimed F001–F100 taxonomy has holes — 8+ feature IDs are never documented.** Spot-probes with grep: **F016, F043, F047, F051, F064, F095, F097, F098, F099** appear nowhere in the page body. D1 claims "14 FEATURES [F001–F014]" but its table has 11 rows (F011–F014 collapsed into one combined row); D2 claims 14 but F015/F016 are never listed; D5 omits F064; D8 claims F091–F100 (10) but documents only 6, omitting F095/F097–F099. The taxonomy header "Visual Matrix Taxonomy (F001–F100)" overstates coverage.

**🟡 M4b. Footer stamp is stale:** `STAMP: 2025-02-18T16:10:04.092Z` (18+ months old vs. repo timeline); "Next Recertification: Q3 2025" also in the past.

**🟡 M5b. Buttons are decorative:** Export Spec, Run Test Suite, affidavits button — no handlers (the page has zero inline scripts, by design, but then the buttons shouldn't promise actions).

**🟢 Non-issues:** anchor navigation is genuinely complete and correct (rare!); honest "Non-Goals & Bounds" section is a genuinely good engineering-culture touch; no runtime errors trivially since there's no JS.

**Verdict:** renders perfectly and its navigation is solid — but this is a *claims document* with three mutually inconsistent test-count claims and an overstated feature taxonomy. Since it presents itself as an "authoritative ledger", the number mismatches matter more here than on a marketing page.

---

## §5a. Recommended fixes (service.html)

1. Update "70/70" → the real count (75/75 today; better, stamp it at build time so it can't drift), and make the per-domain badges sum to the same number.
2. Either document the missing F-IDs or narrow the ranges in headings (e.g. "D8: F091–F096, F100").
3. Add a `<title>`… wait, service.html has one; that fix belongs to marketplace1.html.
4. Refresh/remove the stale 2025-02-18 stamp and Q3 2025 recertification line.

*Repro: `node usecases/proof_more.mjs` (drives both pages; exit 0 = all checks pass).*
