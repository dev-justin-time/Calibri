# UC-6 — Compliance restoration (guardrail tuning + audit artifacts)

The inverse product: for enterprises that must deploy open models but need
*increased* policy compliance, search kernels that RAISE refusal-of-violations
while minimizing drift from the base model — then ship the auditable report.

**Self-contained**: owns the policy prompt pack, the compliance search
(objectives inverted vs UC-5), the drift budget, and the compliance report
format. Depends only on the `calibrix` core.

## Run offline demo

```bash
python restore_compliance.py --trials 6 --popsize 5
python restore_compliance.py --trials 6 --popsize 5 --drift-budget 0.10
```

## Deliverable

`reports/<ts>/compliance_report.html` + `compliance_report.json`:
- policy-violation refusal rate BEFORE vs AFTER (holdout-validated)
- drift metrics vs the agreed budget (KL/length/empty/diversity)
- full trial ledger + compute cost — the governance artifact procurement signs
