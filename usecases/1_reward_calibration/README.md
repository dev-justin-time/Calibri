# UC-1 — Hosted reward-calibrated image-model kernels

Sell calibration as a service: customer names a model + reward, you run a
Calibrix search, deliver a portable kernel spec + auditable report.

**This folder is self-contained.** It owns the service flow: pricing ->
calibration run -> artifact export -> ledger reconciliation. It depends only
on the `calibrix` core package.

## Run offline demo

```bash
python run_job.py                    # flux-dev-gates preset, scripted adapter
python run_job.py --preset qwen-image-gates --trials 20
python run_job.py --json             # machine-readable job ticket + quote
```

## Go live

- swap `ScriptedAdapter` for `CalibriFluxAdapter` (needs the `torch` extra)
  in `run_job.py` to calibrate real FLUX checkpoints
- real reward models: pass `reward_fn` entries (hpsv3_remote, pickscore, ...)
  through `config` as in the Calibri `configs/calibri.py` recipes

## Artifacts

- `jobs/<ts>_<preset>/quote.json`    — customer-facing quote (from `plan_budget`)
- `jobs/<ts>_<preset>/report.json`   — full trial ledger + holdout validation
- `jobs/<ts>_<preset>/kernel_spec.txt` — paste into the ComfyUI node
- `jobs/<ts>_<preset>/ledger.json`   — actual compute actually used
