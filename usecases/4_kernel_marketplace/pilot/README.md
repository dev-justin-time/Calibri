# Product Photo Kernel Pilot Package

This folder is the customer-facing offer for the first sellable wedge:
product-photography kernels for an existing FLUX-compatible ComfyUI workflow.

## Package contents

- `landing.html` — static sales page; deploy it as-is or copy the content into
  a real sales site.
- `baseline_prompts.jsonl` — 24 fixed prompts with 16 train and 8 holdout
  cases.
- `acceptance_criteria.json` — technical, machine-quality, human-review, and
  business boundaries.
- `pricing.json` — $295 fixed pilot offer and explicit exclusions.
- `buyer_questionnaire.md` — intake form for checkpoint, workflow, catalog,
  reviewer, and baseline details.
- `package.json` — machine-readable package manifest.

## Internal delivery flow

1. Send `landing.html` and the questionnaire to a prospective buyer.
2. Confirm the checkpoint license, ComfyUI environment, prompt pack, and
   acceptance reviewer before taking payment.
3. Run `usecases/3_domain_kernels/validate_real.py` against the customer's
   ComfyUI instance/checkpoint using this pack's declared split:

   ```bash
   python ../../3_domain_kernels/validate_real.py \
     --store ../store.json \
     --listing-id product-kernel-v1 \
     --checkpoint FLUX.1-dev.safetensors \
     --prompts baseline_prompts.jsonl \
     --comfy-url http://127.0.0.1:8188
   ```
4. Review `validation.html` with the buyer and complete the human comparison
   sheet.
5. If every gate passes, use `--promote` to attach the matching validation
   artifact to the marketplace listing.
6. Deliver the kernel/license and the complete evidence bundle.

## Pricing boundary

The $295 price pays for the defined pilot work. It is not a promise that the
kernel will improve revenue, conversion, or production cost. If the declared
quality gate fails, report `REVIEW`, perform the included prompt-pack retest,
and do not relabel the result as verified.
