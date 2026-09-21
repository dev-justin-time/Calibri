# UC-4 — Kernel marketplace seller (Stripe-ready)

Operate the kernel marketplace: stock it with calibrated kernels, run the
storefront, fulfill orders, issue licenses. The ComfyUI node
(`calibrix/comfyui_nodes/`) is the buyer-side client that verifies the
CBX1 license keys offline.

**Self-contained**: owns inventory (listings), the payment provider switch,
order fulfillment, and revenue reporting. Depends only on the `calibrix`
core marketplace package.

## Run offline demo (mock payments, zero keys)

```bash
python seller.py stock --from-dir ../3_domain_kernels/kernels/*/listing.json
python seller.py serve --port 8700          # storefront on http://127.0.0.1:8700
python seller.py buy --listing product-kernel-v1 --email test@buyer.io
python seller.py revenue
```

`buy` prints the signed mock webhook curl — POST it (or run `seller.py
webhook`) to complete payment and issue the license.

## Go live with Stripe

```bash
pip install 'calibrix[stripe]'
export STRIPE_SECRET_KEY=sk_live_...
export STRIPE_WEBHOOK_SECRET=whsec_...
python seller.py serve
```

## Revenue model

Per-kernel sales ($3–20). `revenue` reconciles the order ledger: units,
gross, refunds, and per-listing breakdown.
