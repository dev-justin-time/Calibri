# Calibrix Logic Studio — Architecture Ledger

Build digest: `1afb44e1eedfbedd…` · generated 2026-09-21T14:19:46Z

## §0 Provenance & Cleanroom Warranty

- SPDX grants: MIT
- Cleanroom verdict: **CONTAMINATION-REVIEW-REQUIRED** (32 files scanned, 1 reviewed exemptions)
- Taxonomy coverage: 93/108 (86.1%)
- Tests passing at build time: **0** (counts derived from test-file ASTs)
- Missing from taxonomy: F015, F016, F020, F030, F032, F072, F097, S1, S2, S3, S4, S5, S6, S7, S8

## D1: Model Steering & Gate Hooks (`calibrix/studio/steering.py`)

F001-F014 · 14 features · 377 LOC · 24 functions · 0 tests · sha256:75fdd34d41b3

| ID | Name |
|----|------|
| F001 | Q/K Matrix Gain Modulation |
| F002 | MLP Residual Ratio Shifter |
| F003 | LayerNorm Epsilon Clamping |
| F004 | Cross-Attention Bias Gate |
| F005 | Skip-Connection Alpha Damping |
| F006 | Rotary Position (RoPE) Angle Perturbation |
| F007 | SwiGLU Gate Projection Trim |
| F008 | Zero Weight-Mod Bypass Tap |
| F009 | GQA Head Replication Router |
| F010 | DiT Double-Stream Synchronizer |
| F011 | Dynamic Text-Pooling Injection |
| F012 | Softmax Temperature Annealer |
| F013 | Output Projection Gamma Damper |
| F014 | Multi-LoRA Dynamic Arbiter |

## D2: Scorer Panels & Reward Functions (`calibrix/studio/scoring.py`)

F015-F028 · 11 features · 267 LOC · 13 functions · 0 tests · sha256:997ee5a4f792

| ID | Name |
|----|------|
| F019 | CIELAB Delta-E Gamut Scorer |
| F022 | SSIM Structural Preserver |
| F025 | Style-Consistency Patch Distance (LPIPS proxy) |
| F023 | OCR Typographic Sharpness (gradient energy) |
| F018 | Refusal-direction classifier hook |
| F021 | Face/Hand Geometry L2 (deterministic ROI proxy) |
| F024 | VRAM Peak Footprint Scorer (host-side metering) |
| F017 | NFETrap (studio-side, composable with calibrix.scorers.NFETrap) |
| F027 | Normalized Weight Auto-Sum |
| F026 | Zero-Shot Safety Boundary Gate |
| F028 | Custom Python Scorer Hook (sandboxed) |

## D3: Optimizers & Pareto Frontiers (`calibrix/studio/optim.py`)

F029-F040 · 10 features · 354 LOC · 10 functions · 0 tests · sha256:1ea774f667e8

| ID | Name |
|----|------|
| F031 | NSGA-II non-dominated sorting |
| F039 | Pareto Knee-Point Selector |
| F029 | CMA-ES is in calibrix/optimizers.py; here: the missing solvers |
| F033 | Successive Halving / Hyperband pruner |
| F034 | Tiny GP-UCB |
| F035 | Random Subspace Projector |
| F036 | Step-size adaptive decayer (1/5th rule, classic ES) |
| F038 | Exploration Entropy Booster |
| F037 | Population Island Migrator |
| F040 | Warm-Start Run Clone Importer |

## D4: Overfit Detection & Proof Engines (`calibrix/studio/proof.py`)

F041-F052 · 12 features · 284 LOC · 14 functions · 0 tests · sha256:99fbb33fe1b8

| ID | Name |
|----|------|
| F041 | 80/20 Holdout Splitter (deterministic, stratified option) |
| F042 | Holdout Delta Gap Alarm |
| F043 | Blind Cross-Validation Fold |
| F044 | SHA-256 Checkpoint Seal |
| F048 | Gradient Bleed Leak Probe |
| F045 | Out-of-Distribution Stressor |
| F046 | Auto-Rollback Snapshotter |
| F047 | Semantic Divergence Monitor (embedding-free) |
| F049 | Generalization Bounds Bounder |
| F050 | Prompt Memorization Canary |
| F051 | Early Stop Patience Watchdog |
| F052 | Auditable Dossier Generator |

## D5: Billing, Metering & Compute Arb (`calibrix/studio/billing.py`)

F053-F065 · 13 features · 298 LOC · 8 functions · 0 tests · sha256:f694484f777e

| ID | Name |
|----|------|
| F053 | plan_budget Verification Core |
| F059 | Hard Cost Limit Circuit Breaker |
| F054 | RunPod Spot Instance Bidder (provider-agnostic pricing core) |
| F055 | Retainer Margin Arbitrage |
| F063 | Dynamic Gainshare Calculator |
| F056 | Per-Trial Token Cost Ticker |
| F057 | Spot Eviction Failover Gate |
| F058 | Lambda Labs BYOC Key Relay (credential hygiene core) |
| F060 | SaaS Multi-Tenant Quota Guard |
| F061 | VRAM Idle Reclaim Automator |
| F062 | CO2 Emissions Carbon Ticker |
| F064 | Net-30 Invoice Generator |
| F065 | Credit Balance Auto-Debit |

## D6: Runtime Engines & Artifact Packing (`calibrix/studio/runtime.py`)

F066-F078 · 12 features · 255 LOC · 13 functions · 0 tests · sha256:7a8e6b850172

| ID | Name |
|----|------|
| F066 | ComfyUI Native Node Exporter |
| F078 | Safetensors Metadata Injector |
| F069 | FP8 Quantization Gate Preserver |
| F074 | Hot-Swappable Gate Server |
| F075 | ONNX Runtime Serialization (metadata side; graph export needs torch) |
| F071 | Apple Metal / F072 — Triton: gain precompute helpers |
| F076 | Edge TPU Quantization Clamp |
| F070 | REST API Endpoint Dispatcher (routing plan; HTTP layer is server.py) |
| F077 | Kubernetes Cluster Auto-Scaler (HPA math) |
| F073 | Diffusers Pipeline Adapter (monkeypatch-free hook spec) |
| F067 | vLLM PagedAttention Hook (gain table contract) |
| F068 | TensorRT-LLM Engine Builder (plan spec; compile happens in TRT) |

## D7: Governance, IP Warranty & Compliance (`calibrix/studio/governance.py`)

F079-F090 · 12 features · 314 LOC · 13 functions · 0 tests · sha256:43117c3bfb5c

| ID | Name |
|----|------|
| F079 | MIT Cleanroom Verification |
| F081 | Weight Modification Zero-Leak |
| F080 | EU AI Act Annex IV technical documentation |
| F082 | Data Lineage Provenance Stamp |
| F083 | SOC2-style Audit Log Chain |
| F086 | Adversarial Prompt Sanitizer |
| F088 | PII Masking & Anonymizer |
| F085 | Toxic Output Watermarking (deterministic latent-bit watermark) |
| F087 | Role-Based Permission Gate |
| F089 | Export Encryption Vault (envelope encryption, stdlib AES-free) |
| F090 | Keystore (interface for HSM/KMS backends) |
| F084 | IP Indemnity Policy Seal (contract metadata) |

## D8: Commercial Settlement & Royalties (`calibrix/studio/settlement.py`)

F091-F100 · 9 features · 237 LOC · 7 functions · 0 tests · sha256:0b66c84fd6f6

| ID | Name |
|----|------|
| F091 | 25% Savings Gainshare Engine (contract-bound) |
| F092 | Escrow Hold (authorize → capture → release) |
| F093 | Automated Royalty Splitter |
| F094 | Tokenized Calibration License (signable, portable) |
| F095 | Pay-Per-Inference Micro-Royalty |
| F096 | Monthly Retainer Burn Tracker |
| F098 | Multi-Currency Settlement (published reference-rate conversion) |
| F099 | Enterprise SLA Uptime Credit |
| F100 | Zero-Knowledge-style License Attestation |

## Services S1-S8 (`calibrix/studio/services.py`)

8 services · 0 tests · sha256:a6f483ca857b

| ID | Name |
|----|------|
| S1 | Challenger Watch (kernel invalidation monitoring) |
| S2 | Compat Scanner (instant quoting for arbitrary models) |
| S3 | Drift Sentinel (production monitoring subscription) |
| S4 | KernelOps (versioning, staged rollout, rollback) |
| S5 | Benchmark Harmonizer (cross-reward normalization) |
| S6 | Prompt Pack Curator (active-selection for calibration budgets) |
| S7 | Kernel Transfer (cross-model warm start) |
| S8 | SLO Governor (marginal-utility early stop) |

## Non-Goals & Honest Bounds

- F028: user scorers run in a restricted AST sandbox, never as arbitrary production bytecode.
- F089: envelope encryption uses a documented stdlib construction; swap to AES-256-GCM/HSM for production key custody.
- F100: hash-commitment + challenge-response attestation, not a zk-SNARK circuit.
- F080: Annex IV output is audit documentation, not legal advice or a notified-body certification.
