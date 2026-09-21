# UC-8 — Open-core install validation (the funnel)

The 30-second "does Calibrix work on my machine?" check that converts
GitHub visitors into users. This is the top of the open-core funnel: anyone
can run it, offline, free — and the output tells them exactly which
paid-tier features their install could unlock.

**Self-contained**: owns the environment probe, the feature matrix, and the
upsell report. Depends only on the `calibrix` core.

## Run

```bash
python check_install.py
python check_install.py --json
```

## What it reports

- which optional stacks are importable (numpy / torch / cma / optuna / stripe)
- which use-case verticals this install can run TODAY (1–8)
- which paid tiers each vertical maps to (hosted jobs, retainers,
  marketplace, compliance, reports)
