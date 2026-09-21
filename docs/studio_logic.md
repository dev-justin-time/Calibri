# Calibrix Logic Studio — Real Logic Reference

Every feature from the Visual Matrix Synthesizer taxonomy (F001–F100) mapped
to its real implementation in `calibrix/calibrix/studio/`, plus the new
high-value services (S1–S8) that the taxonomy didn't cover.

**License posture (why this is safe):** every module is a clean-room
implementation written from *published equations and public specifications
only* — academic papers (cited inline per function), ISO/CIE standards, and
open file-format specs. No code from Heretic (AGPL), no proprietary runtime
code. Methods and formats are not copyrightable; our expression of them is
original, MIT-licensed, and documented.

**Verification:** 70 unit tests in `calibrix/tests/test_studio.py` cover every
function below; all run offline in ~1.3s with numpy only (torch optional).

---

## Domain 1 — Model Steering & Gate Hooks (`steering.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F001 Q/K Matrix Gain | `qk_gain`, `qk_gain_per_head` | `logits' = g·(QKᵀ/√d)` (Vaswani 2017 Eq. 1) | Highest-leverage steering knob: all token mixing flows through attention logits |
| F002 MLP Residual Ratio | `mlp_residual_ratio` | `x' = x + r·MLP(x)` | Depth-specific fact/skill tuning without touching attention |
| F003 LayerNorm ε Clamp | `layernorm`, `rmsnorm` | Ba 2016 / Zhang & Sennrich 2019 with ε ≥ 1e-6 clamp | High-gain steering collapses variance; clamping prevents silent zero-output |
| F004 Cross-Attn Bias Gate | `context_bias` | `c' = c + b` | One d_model vector steers what conditioning reads — cheap, powerful |
| F005 Skip-Connection α | `skip_alpha` | `out = deep + α·shallow` | Controls detail survival under NFE cuts (UC-2 critical) |
| F006 RoPE Perturb | `rope_angles`, `apply_rope` | Su et al. 2021 θᵢ = base^(−2i/d), freq-scale knob | Long-prompt fidelity tunable with one parameter |
| F007 SwiGLU Trim | `swiglu` | Shazeer 2020 silu(g)·u with threshold zeroing | Cheap sparsification; scorers guard quality |
| F008 Zero-Mod Bypass Tap | `ZeroModTap` | SHA-256 digests at attach, verify on demand | The *enforcement* behind every "non-destructive" claim and D7/F081 |
| F009 GQA Router | `gqa_route` | q_heads→kv_heads grouping map | Group-selective KV steering instead of global |
| F010 DiT Stream Sync | `dit_cross_gain` | Independent cross-stream gains for MM-DiT | FLUX/SD3/Qwen-specific: text↔image influence balance |
| F011 Pooled Injection | `pooled_injection` | `h + s·pooled` | Global style shift with one parameter |
| F012 Softmax Temp Annealer | `softmax_temperature`, `anneal_temperature` | p = softmax(z/T), linear schedule | Emulates depth sharpening without weights |
| F013 Output γ Damper | `output_gamma`, `compounded_gain` | out·γ; tracks Πgains through depth | Prevents compounding blowup when many gains are active |
| F014 Multi-LoRA Arbiter | `lora_blend`, `lora_blend_normalized` | Σwᵢδᵢ (raw or simplex) | Blend search across adapter libraries; normalization bounds magnitude |

## Domain 2 — Scorer Panels & Rewards (`scoring.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F019 CIELAB ΔE | `srgb_to_lab`, `delta_e_cie76`, `color_drift_score` | CIE 1976 + IEC sRGB→XYZ matrices | *Perceptual* color drift — RGB distance lies; ΔE matches human JND |
| F022 SSIM | `ssim` (+`_gaussian_kernel`, `_filter2`) | Wang 2004 with 11×11 Gaussian window, no scipy | The NFE-cut guard: structure must survive (F017's pair) |
| F025 Patch-LPIPS proxy | `patch_lpips` | LPIPS *concept*: per-patch normalized gradient statistics | Style-drift detection without model weights — deterministic, offline |
| F023 OCR Sharpness | `text_sharpness` | Gradient energy (|∇x|+|∇y| mean) | Glyph legibility proxy fast enough for inside the loop |
| F018 Refusal Projection | `refusal_projection` | Projection on refusal direction in σ units | Runtime compliance probe reusing the D-ablation machinery |
| F021 Geometry L2 | `geometry_anomaly` | Landmark deviation / canonical scale | Anatomy guard; scoring math is model-free, detector pluggable |
| F024 VRAM Scorer | `VramMeter`, `vram_score` | Peak-by-construction meter (CUDA or RSS); quadratic over-budget decay | Memory is a real constraint for edge deployment; billing consumes it |
| F017 NFE Penalty | `nfe_penalty` | α·max(0, steps − ref) | The reward-hacking trap: quality gains that are just compute |
| F027 Weight Auto-Sum | `normalize_weights` | Clip ≥ 0, renormalize (convex combo) | Without it the optimizer games weight magnitude, not tradeoffs |
| F026 Safety Gate | `safety_gate` | all(harm ≤ 0.05) | Panel abort: never optimize through unsafe content |
| F028 Custom Scorer Hook | `load_scorer_class` | Restricted-globals exec + documented leakiness | Customer rewards plug in; honesty about sandbox limits |
| F015/F016/F020 | *(already real)* `calibrix.scorers`, `src/metrics/*` | PickScore/HPSv3/KL live in the Calibri stack | No duplication — the studio composes them |

## Domain 3 — Optimizers & Pareto (`optim.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F031 NSGA-II | `pareto_fronts`, `crowding_distance`, `nsga2_select`, `pareto_mask` | Deb 2002: fast non-dominated sort + crowding | True multi-objective search — the compliance-vs-quality-vs-cost front |
| F039 Knee Selector | `knee_point` | Achievement-scalarizing min-max on normalized front | Picks THE kernel to ship without client judgment calls |
| F032 Differential Evolution | `DifferentialEvolution` | Storn & Price 1997 DE/rand/1/bin | Multimodal landscapes where CMA-ES covariance estimation struggles |
| F033 Hyperband Pruner | `SuccessiveHalving` | Jamieson & Talwalkar 2016 η-elimination | Kills bad candidates at 1/η budget — direct compute savings |
| F034 GP-UCB | `GPUCB` (+internal RBF GP) | Srinivas 2010; μ + βσ on a compact GP | Sample-efficient when evaluations are expensive (they are) |
| F035 Subspace Projector | `random_subspace` | Random active-axis subset (JL-style) | 114-dim → 16-dim warm phase; expand if front still moves |
| F036 σ Decayer | `adapt_sigma` | Classic 1/5th success rule | Self-tuning ES without CMA's O(d²) cost |
| F037 Islands | `Islands` | Ring migration, top-m replace-worst (Cantú-Paz 1998) | Multi-GPU shards with periodic exchange |
| F038 Entropy Booster | `entropy_inject` | Gaussian restart noise | Escape valve for premature convergence |
| F040 Warm-Start | `warm_start_from_prior` | N(μ_prior, σ²) population seed | Re-runs and cross-model transfer start half-priced |
| F029/F030 | *(already real)* `calibrix/optimizers.py` | CMA-ES + Optuna TPE | The workhorses; studio adds what was missing |

## Domain 4 — Overfit & Proof (`proof.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F041 Holdout Splitter | `holdout_split` | Deterministic RNG, stratified option | Refusal probes are rare — stratification keeps them in holdout |
| F042 Gap Alarm | `gap_alarm`, `pvalue_from_deltas` | Relative train/holdout gap + exact sign test | The rule *and* the evidence: alarm + attached p-value |
| F043 K-Fold CV | `kfold_scores` | Caller-supplied fold evaluator, mean/std | Blind cross-validation for the dossier |
| F044 Checkpoint Seal | `seal_bytes/array/state` | Order-stable SHA-256 over named tensors | Every "weights untouched" claim binds to a digest |
| F045 OOD Stressor | `ood_stress`, `robustness_delta` | Deterministic casing/spacing/unicode corruption | Cheap OOD probes; >20% drop ⇒ kernel memorized prompts |
| F046 Auto-Rollback | `RollbackLedger`, `Snapshot` | Last-known-robust by gap criterion, sealed | Alarms without recovery are useless; this recovers |
| F047 Semantic Divergence | `lexical_divergence` | Jaccard distance over token sets | Weights-free degeneration monitor; swap embeddings in later |
| F048 Zero-Grad Proof | `zero_grad_proof` | All-grads-absent-or-zero attestation | Cryptographic-ish honesty: no backprop reached base weights |
| F049 Generalization Bound | `rademacher_bound` | Bartlett-Mendelson-style √(d·log/δ) bound | d≈100 is why kernels generalize — this *quantifies* the sales claim |
| F050 Memorization Canary | `memorization_canary` | Shared 8-gram detection vs training prompts | The standard memorization-audit technique, reduced to its core |
| F051 Patience Watchdog | `PatienceWatchdog` | min-δ patience counter | Stops the tail where 80% of compute buys 20% of quality |
| F052 Dossier Generator | `build_dossier` | JSON bundle: seals + stats + ledger, traceable | The artifact procurement signs (UC-6's deliverable) |

## Domain 5 — Billing & Arbitrage (`billing.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F053 Budget Guard | `BudgetGuard` | Step→USD accrual, affordability, remaining headroom | Ex-ante ceiling enforcement on top of ex-post Meter |
| F059 Cost Breaker | `CostBreaker` | Quote×(1+tol) limit, fail-closed latch | Runaway spot churn or trial storms kill margin — this can't be un-tripped by accident |
| F054 Spot Bidder | `SpotQuote`, `best_spot`, `job_cost_usd` | Provider-agnostic quotes, min-discount floor | 40–70% compute arbitrage is the margin engine for UC-1 |
| F055 Retainer Margin | `retainer_margin` | retainer − compute, % surface | Negative margin surfaces mid-month, not at invoice |
| F056 Micro Meter | `MicroMeter` | Per-trial FLOP→µUSD ledger | Feeds cost-aware Pareto: a 10× costlier trial must win by enough |
| F057 Eviction Failover | `EvictionPlan`, `should_checkpoint` | Signal-or-cadence checkpointing | Spot eviction resilience — the price of the arbitrage |
| F058 BYOC Relay | `ByocRelay` | In-memory keys, zeroize on release | Customer cloud keys: minimal hygiene, KMS-swappable interface |
| F060 Quota Guard | `QuotaGuard` | Concurrent + monthly-step caps per org | Multi-tenant SaaS needs admission control *before* billing |
| F061 Idle Reclaim | `IdleReclaimer` | Idle-second tracking → unload signal | Expensive models parked in VRAM while idle burn the margin |
| F062 CO₂ Ticker | `co2_grams`, `kwh_from_gpu_hours` | Published grid intensities, PUE formula | Enterprise ESG reporting asks for exactly this number |
| F063 Gainshare Math | `gainshare` | verified savings × share | The UC-2 billing primitive (F091 wraps it in a contract) |
| F064 Invoice Gen | `build_invoice` | Itemized qty×unit lines, Net-30 terms | Metering → money handoff |
| F065 Credit Wallet | `CreditWallet` | reserve-before-dispatch, settle-actuals | Pre-funded enterprise flow; no negative balances |

## Domain 6 — Runtime & Artifacts (`runtime.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F066 ComfyUI Export | `export_comfyui_workflow` | Published workflow JSON schema (nodes/links/extra) | UC-4's distribution channel — kernels as drag-and-drop workflows |
| F078 Safetensors Header | `safetensors_header` | u64_le length + JSON header spec (dtype/shape/offsets + `__metadata__`) | License + kernel hash travel *inside* the artifact |
| F069 FP8 Preserver | `quantize_gains_fp8_e4m3`, `quantization_penalty` | E4M3 simulation (3-bit mantissa, max 448) | Scorers validate the *dequantized* gains — no deployment surprises |
| F076 Edge TPU Clamp | `quantize_gains_int8` | Symmetric int8 with scale, clamp ±127 | Edge deployment path |
| F074 Hot-Swap Registry | `GateBundle`, `GateRegistry` | Atomic dict-pointer swap + history | Swap kernels without reloading 24GB of weights |
| F075 ONNX Props | `onnx_metadata_props` | ModelProto.metadata_props contract | Provenance in the universal export format |
| F071 Metal Tile | `tile_gains_uniform`, `threadgroup_size` | Contiguous f32 buffer, multiple-of-32 threadgroups | Apple Silicon runtime layout |
| F072 Triton | *(via tile_gains_uniform)* | Same contiguous layout feeds fused kernels | One layout, many backends |
| F070 REST Plan | `build_api_plan` | Endpoint contract: middleware, concurrency, SLO | Deployment without a live server in the loop |
| F077 K8s Scaler | `desired_replicas` | ⌈queue/target⌉ with HPA's 10% anti-flap band | Real autoscaler math, testable |
| F073 Diffusers Hook | `diffusers_hook_spec` | Forward-hook attachment spec | Monkeypatch-free pipeline integration |
| F067 vLLM Table | `vllm_gain_table` | Per-layer f32 table contract | Worker-plugin gain delivery |
| F068 TRT Plan | `trt_plan_spec` | Constant-weight plugin plan metadata | Gains baked as engine constants |

## Domain 7 — Governance (`governance.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F079 Cleanroom Verify | `cleanroom_scan` | Regex scan for AGPL/GPL markers + SPDX headers | Machine-checkable version of the license claim — run it in CI |
| F081 Zero-Leak Proof | `zero_leak_proof` | Pre/post digest bit-equality attestation | The formal version of F008's runtime tap |
| F080 EU AI Act | `eu_ai_act_annex_iv` | Annex IV section structure populated from telemetry | The documentation *draft* compliance teams review (not legal advice) |
| F082 Provenance | `provenance_digest` | SHA-256 over dataset-id + ordered prompts | Binds the exact prompt distribution to a digest |
| F083 Audit Chain | `AuditChain` | Hash-linked append-only log, verify() walk | SOC2-style tamper-evident logging, stdlib only |
| F084 IP Indemnity | `indemnity_policy` | Contract↔kernel-seal binding record | Commercial safe-harbor needs artifact-level traceability |
| F085 Watermarking | `watermark_latents`, `detect_watermark` | Keyed pattern, scale-relative amplitude, cosine detection | Output provenance at perception thresholds |
| F086 Sanitizer | `sanitize_prompt`, `sanitize_batch` | Injection-pattern regex filter | Prompt-injection hygiene before search |
| F087 RBAC | `authorize` | Role→action set membership | The minimal gate D5/F060 quotas assume |
| F088 PII Masking | `mask_pii` | Documented pattern set (email/phone/SSN/IP/IBAN) | Enterprise prompt banks can't enter logs with PII |
| F089 Encrypt Vault | `envelope_encrypt/decrypt` | DEK + KEK envelope (interface matches AES KMS) | Encrypted exports; swap `cryptography` in for production |
| F090 Keystore | `Keystore` | get/set/delete + digest contract | HSM/KMS-swappable secret resolution |

## Domain 8 — Settlement (`settlement.py`)

| Feature | Implementation | Real logic | Why included |
|---|---|---|---|
| F091 Gainshare Engine | `GainshareContract` | Baseline vs actual steps → savings × share, floor | The UC-2 retainer as a *contract object*, not a formula in a script |
| F092 Escrow | `Escrow` | hold → capture(actual) → release(rest) | Stripe manual-capture semantics, testable |
| F093 Royalty Splitter | `RoyaltyParty`, `split_receipts` | Normalized shares + platform remainder | Marketplace payouts that reconcile to the cent |
| F094 Tokenized License | `mint/verify_license_token` | HMAC token (CBXT1), same scheme as CBX1 keys | One offline verification story: marketplace + enterprise |
| F095 Micro-Royalty | `MicroRoyaltyLedger` | µUSD accumulation, batch settlement above floor | Per-inference royalties only work batched — the ledger encodes that |
| F096 Burn Tracker | `BurnTracker` | spend/cap %, alert at 85%, exhaustion flag | Retainer overruns surfaced before they happen |
| F098 FX Settlement | `convert` | Reference-rate table + fx fee | Enterprise billing in EUR/GBP/USD |
| F099 SLA Credits | `SLA_TIERS`, `sla_credit` | Tiered uptime floors → credit % | The enterprise contract clause, as code |
| F100 ZK-style Attest | `license_attestation/verify` | Commitment + challenge-response | The commit-challenge-respond API a real zk system would keep |
| F097 Publisher | *(already real)* `marketplace/store.py` | Listings + orders + fulfillment | UC-4 is the implementation |

---

## New high-value services (`services.py`) — not in the taxonomy, added because nothing above covers them

| Service | What it does | The gap it fills | Revenue tie-in |
|---|---|---|---|
| **S1 Challenger Watch** | Compares live baseline scorer values against the *sealed* baseline; flags when upstream model updates invalidate a deployed kernel | Nothing in D1–D8 detects kernel rot after a provider ships a new checkpoint | **Recurring re-calibration jobs** — the trigger was missing |
| **S2 Compat Scanner** | Static block-structure analysis → family, kernel-param count, min-trial estimate | Every prospect asks "does my model work?" — answers needed an actual run | **Instant quoting** in the sales flow (UC-1) |
| **S3 Drift Sentinel** | Watches production outputs vs sealed baseline scores; alerts on threshold breaches | One-time jobs without monitoring = no recurring revenue, no trust | **SaaS monitoring subscription** on top of UC-1/UC-2 |
| **S4 KernelOps** | Versioned kernels, staged % rollout, automatic rollback on live regression vs holdout estimate | Deploying a kernel is a production change; without rollback it's an ops liability | **Enterprise gate** for UC-1/UC-2 deals |
| **S5 Benchmark Harmonizer** | Percentile-normalizes PickScore/HPSv3/ImageReward onto one 0–1 scale for listing scores | Marketplace claims like "+0.2" are meaningless across reward scales | **Marketplace trust layer** (UC-4) |
| **S6 Prompt Pack Curator** | Farthest-point (k-center greedy) selection of the most informative prompts for a compute budget | Cost scales with prompts-per-eval; most banks are 80% redundant | **Same compute explores more** → direct margin on every job |
| **S7 Kernel Transfer** | Resamples a gain profile across model depth (linear interp over normalized depth) | Same-family models share gain *shape* at different depths | **Warm-started searches ≈ half the trials** → cheaper jobs |
| **S8 SLO Governor** | Stops the search when marginal fitness per marginal dollar drops below the bar | "Run until convergence" wastes the last 20% of quality on 80% of compute | **Enforces the profit SLO** on every hosted run |

---

## How it composes with the existing stack

```
SearchEngine (engine.py) ── uses ──> optimizers.py (CMA-ES/TPE) + studio.optim (NSGA-II/DE/GP-UCB/SH)
        │  scorers.py + studio.scoring (ΔE/SSIM/sharpness panel)
        │  metering.py (ledger) ← guarded by → studio.billing (BudgetGuard/Breaker/Quotas)
        ▼
TrialResult ── validated by ──> studio.proof (k-fold, seals, canary, bounds)
        │
        ▼
runtime.py (ComfyUI/safetensors/ONNX artifacts) ── attested by ──> governance.py (cleanroom/zero-leak/audit)
        │
        ▼
settlement.py (contracts, escrow, splits) ── paid via ──> marketplace/stripe_provider.py
```

- **Everything is numpy-only and offline** — the entire studio test suite (70 tests) runs in ~1.3s with no GPU and no network.
- **Torch is optional** and only needed where physics demands it (real model forward passes).
- Every "proof" claim in the docs now has a machine-checkable function behind it; every money-flow has deterministic, unit-tested math that Stripe calls merely enforce at the edges.

**Explicit non-goals / honest limits** (documented in-code): the F028 sandbox is hygiene, not security; F089's cipher is a stdlib stand-in for AES; F100 is commitment-based, not a zk-SNARK; F080 produces a documentation *draft*, not legal advice. Production swaps for these are interface-compatible and noted at each site.
