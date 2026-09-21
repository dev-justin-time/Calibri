# Usecases — independently runnable vertical slices

One folder per use case from `docs/product_opportunities.md`. Every folder
is a standalone product surface:

- it has its own entry script (`python <script>.py`) and README
- it owns ALL use-case-specific logic, config, and outputs inside the folder
- it depends ONLY on the `calibrix` core package (engine, scorers, adapters,
  metering, marketplace, ablation) — never on another use-case folder
- each runs 100% offline by default (scripted adapter / mock payments); the
  README inside each folder shows the one-line swap to real models/providers

| # | Folder | Use case | Revenue model |
|---|--------|----------|---------------|
| 1 | `1_reward_calibration` | Hosted reward-calibrated image-model kernels | $200–2k per job |
| 2 | `2_nfe_cost_reduction` | Inference-cost reduction retainers | % of documented savings |
| 3 | `3_domain_kernels` | Domain-tuned kernels without LoRA training | per-kernel sales |
| 4 | `4_kernel_marketplace` | Stripe-ready kernel marketplace (ComfyUI client) | $3–20 per kernel |
| 5 | `5_abliteration_service` | Managed decensoring (MIT clean-room method) | per-run GPU minutes |
| 6 | `6_compliance_restoration` | Guardrail *restoration* + audit artifacts | enterprise per-report |
| 7 | `7_interp_reports` | Interpretability report service | per-model reports |
| 8 | `8_open_core` | MIT open-core install validation | funnel → hosted tiers |

Run any of them from inside its folder:

```bash
cd usecases/1_reward_calibration && python run_job.py
```
