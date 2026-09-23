# UC-4 — Product-photo kernel marketplace (Stripe-ready)

This is a focused catalog for **ComfyUI product-photography kernels**—not a
fake financial exchange. The first vertical is e-commerce imagery: clean
backgrounds, accurate color, sharp subjects, and repeatable catalog style.

The storefront is backed by the same JSON ledger that handles checkout,
webhooks, licenses, and revenue. It provides:

- product-photo as the default catalog category
- URL-backed search, category, and evidence filters
- cards with compatibility, tags, scores, and evidence status
- product detail pages with install requirements and buyer onboarding
- fulfillment page that shows the delivered license after the webhook
- honest evidence labels: `simulation`, `measured`, `reference_only`, `verified`

A `verified` card is only possible after the real FLUX/ComfyUI validation path
passes and the matching listing checksum is promoted. No card claims sales
lift or customer ROI.

## Run locally

```bash
python seller.py serve --port 8700
# open http://127.0.0.1:8700/
```

Useful catalog URLs:

```text
/                                      # product-photo catalog
/?q=background                       # search title/description/tags
/?evidence=verified                   # show only verified evidence
/?category=all                       # inspect non-product-photo inventory
/listing/product-kernel-v1            # detail + onboarding
```

## Buyer journey

1. Confirm the exact compatible checkpoint shown on the card.
2. Install `calibrix/comfyui_nodes/calibrix_node.py` into ComfyUI.
3. Open the product detail page and submit an email to checkout.
4. Complete the configured payment provider flow.
5. The payment webhook fulfills the order and binds a `CBX1` license to the
   exact kernel checksum.
6. Copy the key from the fulfillment page into the Calibrix Kernel Scale node.

The base checkpoint and generated images are **not** included. Buyers should
run their own before/after acceptance test on their catalog before scaling.

## Offline demo (mock payments, zero keys)

```bash
python seller.py demo
python seller.py stock --from-dir ../3_domain_kernels/kernels/*/listing.json
python seller.py buy --listing product-kernel-v1 --email test@buyer.io
python seller.py revenue
```

The mock provider is for local testing only. It does not prove demand or
revenue. Go live only after configuring Stripe:

```bash
pip install 'calibrix[stripe]'
export STRIPE_SECRET_KEY=sk_live_...
export STRIPE_WEBHOOK_SECRET=whsec_...
python seller.py serve
```

## Revenue model

Start with $7–20 one-time kernel licenses, then use verified product kernels
to sell higher-value custom calibration/validation work. The marketplace is
the distribution channel; the defensible paid service is the measured
customer-specific kernel and acceptance report.
