# UC-3 — Domain-tuned kernels without training

Vertical kernels (product photography, anime/posters, pastel art, ...)
calibrated against domain prompt packs + offline scorers — the LoRA-
replacement pitch: hours instead of days, no catastrophic forgetting.

**Self-contained**: owns the domain prompt packs, calibration, and kernel
exports. Depends only on the `calibrix` core.

## Run offline demo

```bash
python tune_domain.py --domain product    # e-commerce product shots
python tune_domain.py --domain anime      # poster/illustration look
python tune_domain.py --domain pastel     # soft palette art
python tune_domain.py --list              # available domain packs
```

## Go live

The domain scorer weights live in `domain_packs.json` (editable). For real
image scoring, replace `OfflineImageQuality`-style scorers with the CLIP-
backed ones from the Calibri stack (`src/metrics/`) and run against a local
checkpoint via the ComfyUI/Calibri adapters.

## Output

`kernels/<domain>_<ts>/` — kernel_spec.txt (drop into the ComfyUI node),
report.json/html (holdout-validated), and a listing-ready `listing.json`
(price, description, scoring metrics) that UC-4 can publish as-is.
