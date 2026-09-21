# Calibri × Heretic × Calibrix — Use Cases & Monetization

Grounded in the code in this repository, the actual market landscape (Heretic
5000+ community models on Hugging Face, Calibri's published DiT calibration
results), and the license constraints that decide what can legally be
commercialized.

---

## 1. What the assets actually are

| Asset                            | What it does                                                                                                                                                                                                                                                                                | Key code                                                                                                   | License      |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|--------------|
| **Calibri** (`src/`, `configs/`) | CMA-ES calibration of frozen DiT text-to-image models — FLUX.1-dev, SD3.5 medium/large, Qwen-Image. ~100 gate parameters, CMA-ES, reward-driven (HPSv3, Q-Align, PickScore, ImageReward).                                                                                                   | `src/models/*_sg.py` (per-arch gate hooks), `src/optim/cmaes.py`, reward servers in `src/metrics/`         | **MIT**      |
| **Heretic** (`heritic/heretic/`) | Fully automatic abliteration of LLMs: difference-of-means residual directions, weighted directional ablation, TPE co-optimization of refusal-rate vs KL-drift, benchmark/eval harness, ~30 min per model on one GPU.                                                                        | `src/heretic/` (`analyzer`, `model`, `scorer`, `plugin`, `reproduce`, scorers package)                     | **AGPL-3.0** |
| **Calibrix** (`calibrix/`)       | Generalization of both: 6-param modulation kernel per component, scorer-panel co-optimization, train/holdout split with overfitting alarm, Pareto trial selection, cost metering + `plan_budget`, JSONL event log, self-contained HTML report, offline mode, ComfyUI adapter + kernel node. | `calibrix/calibrix/{kernel,scorers,adapters,engine,metering,report,optimizers,judge}.py`, `comfyui_nodes/` | **MIT**      |

**The one-sentence pitch:** Calibri proved ~100-parameter calibration moves
reward on frozen image models; Heretic proved auto-optimized parametric
interventions move behavior on frozen language models; Calibrix already
unifies the pattern (any frozen transformer + scorer panel + kernel search)
and adds the missing business layer: auditable cost metering, holdout
validation, and portable exports. The product is the **"tune a model's
behavior for a reward you care about, cheaply, provably, without retraining"
service** on top of that stack.

**License constraint (decides everything):**
- Calibri + Calibrix are **MIT** → embed, SaaS, resale all fine.
- Heretic is **AGPL-3.0** → you may *use* it, but any product that links or
  derives from it must be AGPL and you must offer source to network users.
  **Viable strategies:** (a) keep Heretic strictly as a separate OSS/CLI
  surface, (b) reimplement the needed ideas (the *method* is not copyrighted
  — directional ablation + TPE co-optimization, citing Arditi et al. 2024)
  inside MIT-licensed Calibrix, or (c) negotiate dual-licensing with the
  author. Do not build a closed-source SaaS on `heritic/` code.

---

## 2. High-value use cases (from strongest product-market fit down)

### UC-1. Reward-calibrated hosted image models (highest revenue ceiling)
**Who pays:** app builders, marketing-tech, game-studio content pipelines,
stock-image competitors — anyone who ships text-to-image to end users and
loses to a benchmark or loses users to bad outputs.
**Flow:** customer names a hosted model + a reward (their metric, your
scorer library, or their custom scorer via the plugin API) → your stack
calibrates gates on their prompt distribution → they get a ~100-parameter
kernel JSON that rides on the hosted API's diffusion model — **no weights
access needed** if the host applies per-block gains, or fully portable via
the ComfyUI node otherwise.
**Why it wins:** Calibri's published results show large reward gains at a
fraction of NFE (15 steps FLUX, 30 SD3.5/Qwen). Selling "30% fewer steps at
equal or better quality" is a direct GPU-bill cut — the buyer's CFO signs.
**Code hook:** `SearchEngine` + `CalibriFluxAdapter` already run this flow;
`plan_budget` prices it before you quote it.

### UC-2. Inference-cost reduction as a service (clear ROI, easy sell)
**Who pays:** any team whose text-to-image cost line is visible.
**Pitch:** "We cut your NFE 30–50% with a ~100-parameter calibration, we
bill a percentage of the savings." The holdout/overfitting alarm in
`calibrix/engine.py` is the trust mechanism that makes this sellable: you
can *prove* the improvement isn't overfit to the calibration prompts.
**Code hook:** `NFETrap` scorer + `Meter` ledger + holdout validation.

### UC-3. Domain-tuned image models without training (fastest time-to-revenue)
**Who pays:** niche verticals — product photography, anime/sticker
generators, architectural visualization, medical-adjacent stock, print-on-
demand. They currently pay LoRA trainers $2–5k per run and wait days.
**Pitch:** same turnaround in hours, a fraction of the price, zero risk of
catastrophic forgetting because the base weights never change.
**Code hook:** Calibri adapters + `OfflineColorAlignment`,
`OfflineImageQuality` scorers; deliverable is a portable kernel spec
(`attn:1.15@0.55:0.82:0.4|mlp:...`) consumable by the ComfyUI node.

### UC-4. ComfyUI marketplace of calibrated kernels ("app store" model)
**Who pays:** the ComfyUI user base (largest prosumer image-gen community).
You already ship `comfyui_nodes/calibrix_node.py` — a drop-in node that
applies an exported kernel in any workflow.
**Flow:** run a cheap local or cloud search for a (model, style, reward)
combo → publish the kernel spec to a marketplace → buyers drop it into their
existing workflows for a few dollars. Marginal cost ≈ $0, every sale is
margin.
**Why it wins:** kernels are tiny JSON, work offline, and are shareable
"like any other workflow" (your own README's framing) — the distribution
channel is built in.

### UC-5. Managed abliteration marketplace (Heretic surface, AGPL-safe)
**Who pays:** r/LocalLLaMA hobbyists (proven willingness — 5000+ HF models)
and small labs that want a specific uncensored model without running the
pipeline themselves.
**Model:** B2C one-off payments for GPU minutes on your cluster, "bring your
own model" uploads, charge per successful run. Delivered as a separate
product whose user-facing value is Heretic-the-tool; keep it a distinct OSS
CLI + hosted runner so AGPL obligations are met by publishing your runner
fork.
**Caution:** explicitly AGPL boundary — the hosted runner fork is published;
no proprietary code touches it.

### UC-6. Safety/compliance *restoration* (the inverse product — underrated)
**Who pays:** enterprises who must deploy open models (air-gapped, legal
requirements) but need *increased* guardrail compliance, with evidence.
**Flow:** same kernel machinery, opposite direction — search kernels that
raise refusal-rate-on-policy-violations while minimizing KL drift from the
base model, ship the audit trail (event log + HTML report) as the compliance
artifact.
**Why it wins:** a kernel JSON + a generated report is a governance story
procurement can sign; nobody else in the abliteration landscape packages
the *inverse*.

### UC-7. Interpretability-as-a-service (research → revenue)
**Who pays:** labs, universities, AI-safety orgs with grant money.
**Code hook:** Heretic's `--plot-residuals` (PaCMAP layer-by-layer
animations) and `--print-residual-geometry` (per-layer geometry tables);
Calibrix's scorer-panel reports. Sell as: managed runs on their model,
delivered as static HTML/GIF reports.

### UC-8. Calibrix as an open-core library (the platform play)
**Who pays:** ML platforms and inference providers who want
"calibration-as-a-feature."
**Model:** MIT core stays free forever (that's the funnel — it's what makes
Calibrix the reference implementation of this pattern); paid tiers are
hosted orchestration, extra scorers (brand-safety, custom reward APIs),
SLAs, and the enterprise compliance features from UC-6.

---

## 3. Monetizable features ranked by (value × feasibility ÷ license-risk)

| # | Feature                                                                                                         | Price anchor                                     | License risk                                        |
|---|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------|-----------------------------------------------------|
| 1 | **Hosted calibration jobs** (image models, UC-1/UC-3): upload prompts + pick reward → kernel JSON + HTML report | $200–2,000 per run (vs $2–5k LoRA, days faster)  | None (MIT)                                          |
| 2 | **NFE-reduction retainer** (UC-2): we cut your gen costs X%, you keep Y%                                        | 20–30% of documented savings                     | None (MIT)                                          |
| 3 | **Kernel marketplace** (UC-4): per-kernel sales via ComfyUI node                                                | $3–20 per kernel, ~100% margin                   | None (MIT)                                          |
| 4 | **Calibration API** — same engine behind REST for platforms                                                     | Usage-based, `plan_budget` = native billing unit | None (MIT)                                          |
| 5 | **Compliance/restoration reports** (UC-6): audited guardrail-tuning runs                                        | $500–5,000 per report (enterprise pricing)       | None (MIT, using Calibrix reimplementation)         |
| 6 | **Managed decensoring** (UC-5)                                                                                  | $5–25 per run, GPU-minutes margin                | **AGPL** — publish runner fork, keep it separate    |
| 7 | **Interp report service** (UC-7)                                                                                | $100–1,000 per model report                      | AGPL for Heretic parts — re-export via own pipeline |
| 8 | **Enterprise open-core** (UC-8)                                                                                 | $1–10k/mo platform deals                         | None (MIT core)                                     |

---

## 4. Competitive landscape (why the window is open)

- **Abliteration space:** Heretic is the category leader (trendshift #1,
  5000+ community models) but is a *free CLI* with **no hosted offering, no
  marketplace, no cost accounting, no compliance packaging**. Manual
  competitors (mlabonne's notebooks, abliterator.py, ErisForge, etc.) are
  notebooks/expert tools — none productized.
- **Image-calibration space:** Calibri is the only published parameter-
  efficient DiT calibration result; T-LoRA/ProLoRA/DreamBooth competitors
  all *train weights* (slower, riskier, pricier). Nobody sells "calibration
  kernels" as a product or marketplace.
- **The gap Calibrix uniquely fills:** scorer-panel co-optimization with
  holdout validation + cost metering + portable kernel export. Neither
  Heretic nor Calibri has cost accounting or overfitting alarms; no
  competitor in either space has the marketplace distribution (ComfyUI
  node) already built.

---

## 5. Recommended execution order

1. **Week 1–2 — Prove UC-2 on one vertical.** Pick a public image-gen
   community (ComfyUI/FLUX subreddits), calibrate 3 popular checkpoints
   against PickScore/OfflineImageQuality, publish kernels + HTML reports
   free. This is content marketing *and* the marketplace seed inventory.
2. **Week 3–6 — Open the marketplace** (UC-4) using the existing ComfyUI
   node as the client. Stripe, per-kernel pricing. Zero GPU cost to serve.
3. **Month 2 — Launch hosted calibration** (UC-1/UC-3) on the same engine;
   `plan_budget` gives quotes, `Meter` gives margin visibility per job.
4. **Month 3+ — Enterprise compliance track** (UC-6) once 2–3 design-partner
   logos exist; the report artifacts are already generated by the engine.
5. **Parallel, lowest priority:** managed decensoring (UC-5) — proven demand
   but smallest margin and AGPL overhead. Ship it as a standalone AGPL
   product or don't ship it.

## 6. Risks

- **AGPL contamination** (Heretic): the single biggest legal risk. Keep the
  boundary architectural (separate process/package), or reimplement in
  Calibrix (methods are public — Arditi et al. 2024, Lai 2025).
- **Hosted-API kernel applicability** (UC-1): per-block gains need host
  support. Mitigate by leading with ComfyUI/local checkpoints where you
  control the graph, and offering a hosted-API variant only where the host
  exposes modulation hooks.
- **Kernel commoditization** (UC-4): once published, a kernel is copyable.
  Accept it — the moat is the *search service* and fresh kernels for new
  models/rewards, not any single artifact.
- **Reward-model licensing** (HPSv3, Q-Align, ImageReward, PickScore) and
  **base-model licenses** (FLUX-dev is non-commercial!) must be checked per
  deployment — calibrating a FLUX-dev-derived product for resale likely
  violates FLUX's license even though your code is MIT.
