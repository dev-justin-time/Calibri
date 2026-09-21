# UC-7 — Interpretability report service

Sell managed interpretability runs: per-layer residual-geometry analysis of
any transformer, delivered as self-contained HTML/GIF artifacts. Customers:
labs, universities, AI-safety orgs (grant money).

**Self-contained**: owns the residual extraction protocol, the per-layer
metrics table, and the report packaging. Depends only on the `calibrix`
core (plus its offline sim model — swap in a torch model for real runs).

## Run offline demo

```bash
python interp_report.py --model-sim            # offline simulated residuals
python interp_report.py --model-sim --json     # pricing-ready summary
```

## Metrics

Per layer: cluster separation S(g,r), silhouette coefficient, residual norms
(|g|, |b|, |r|), and a confidence-style signal for where behavior is
mediated. Rendered as a self-contained HTML report (file://-safe, like all
Calibrix reports) plus a machine-readable JSON for downstream pricing.
