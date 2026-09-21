# UC-2 — Inference-cost reduction retainers

Sell documented savings: calibrate a model, measure quality-equal NFE
reduction, invoice a share of the savings. The holdout/overfitting alarm is
the trust mechanism that makes the number defensible.

**Self-contained**: owns baseline measurement, calibration, savings math,
and the customer-facing savings report. Depends only on the `calibrix` core.

## Run offline demo

```bash
python reduce_costs.py                 # baseline vs 15/10/5-NFE quality sweep
python reduce_costs.py --share 0.25 --monthly-steps 50000000
```

## The pitch math

`savings_report.json` contains: baseline reward, calibrated reward at each
reduced NFE, the largest NFE cut with non-inferior quality, monthly step
volume, current spend, projected savings, and your share. Hand it over with
the HTML report — the holdout column proves it's not overfit.
