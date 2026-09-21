# UC-5 — Managed abliteration service (MIT clean-room)

Managed refusal-removal runs using the **MIT-licensed** Calibrix directional
ablation (`calibrix/ablation.py` — clean-room implementation of Arditi et
al. 2024 / Lai 2025 methods; no AGPL Heretic code). Sell GPU-minutes per
successful run: direction extraction → TPE co-optimization of compliance vs
drift → holdout-validated artifact.

**Self-contained**: owns the run pipeline, quality gates, and run ledger.
Depends only on the `calibrix` core.

## Run offline demo

```bash
python abliterate.py --trials 6 --popsize 5
python abliterate.py --trials 6 --popsize 5 --json   # run ticket
```

## Go live

Point `--url` at an OpenAI-compatible server exposing a modulatable model
(vLLM with the calibrix in-process adapter, or a local checkpoint via
torch). The offline sim here exercises the identical scoring/optimization/
gating path so the service logic is verifiable without GPUs.

## Quality gate

A run only "ships" when: holdout refusal rate <= threshold AND drift
scorers within budget AND no overfit flag. Failed runs are logged but not
delivered (no charge). Ledger: `runs/<ts>/run_ticket.json`.
