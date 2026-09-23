# UC-3 real FLUX / ComfyUI validator.
#
# This command is intentionally not part of the offline tuner. It requires a
# running ComfyUI instance with the CalibrixKernelScale custom node installed,
# and compares the exact same prompts/seeds with and without the kernel.

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "calibrix"))

from calibrix.image_adapters import ComfyUIAdapter
from calibrix.image_scorers import OfflineColorAlignment, OfflineImageQuality
from calibrix.image_validation import ValidationConfig, validate_real_kernel
from calibrix.kernel import ModulationSpec, parse_kernel_spec
from calibrix.marketplace.store import JsonStore
from calibrix.scorers import seed_prompts

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_inputs(args):
    listing = None
    if args.listing_id:
        if not args.store:
            raise ValueError("--store is required with --listing-id")
        listing = JsonStore(args.store).get_listing(args.listing_id)
        if listing is None:
            raise ValueError(f"unknown listing {args.listing_id!r}")
        kernel_spec = listing.kernel_spec
    else:
        kernel_spec = args.kernel_spec
    if not kernel_spec:
        raise ValueError("provide --listing-id or --kernel-spec")

    declared_split = None
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as handle:
            if args.prompts.lower().endswith(".jsonl"):
                records = [json.loads(line) for line in handle if line.strip()]
                prompt_values = [record["prompt"] for record in records]
                train = [i for i, record in enumerate(records) if record.get("split") == "train"]
                holdout = [i for i, record in enumerate(records) if record.get("split") == "holdout"]
                if train and holdout:
                    declared_split = (train, holdout)
            else:
                prompt_values = [line.strip() for line in handle if line.strip()]
    else:
        pack_path = os.path.join(HERE, "domain_packs.json")
        with open(pack_path, encoding="utf-8") as handle:
            pack = json.load(handle)[args.domain]
        prompt_values = pack["prompts"]
    return listing, kernel_spec, seed_prompts(prompt_values), declared_split


def _kernel_specs(kernel_spec: str, n_blocks: int):
    channels = parse_kernel_spec(kernel_spec)
    by_name = {name: params for name, params in channels}
    if not {"attn", "mlp"}.issubset(by_name):
        raise ValueError("real FLUX validation requires attn and mlp channels")
    return [ModulationSpec(component=name, n_sites=n_blocks, kernel=params)
            for name, params in (("attn", by_name["attn"]), ("mlp", by_name["mlp"]))]


def _probe(adapter: ComfyUIAdapter, kernel_spec: str) -> dict:
    adapter.system_stats()
    info = adapter.object_info()
    node_info = info.get("CalibrixKernelScale")
    workflow = adapter._workflow("validation probe", 0, adapter._last_gains)
    return {
        "backend": "comfyui",
        "base_url": adapter.base_url,
        "checkpoint": adapter.checkpoint,
        "node_available": node_info is not None,
        "node_name": "CalibrixKernelScale",
        "workflow_has_node": any(
            node.get("class_type") == "CalibrixKernelScale"
            for node in workflow.values()
        ),
        "kernel_spec_in_workflow": kernel_spec,
        "mechanism": "flux_norm1_gate_tuple",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a real FLUX kernel through ComfyUI")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--checkpoint", required=True,
                        help="exact checkpoint filename installed in ComfyUI")
    parser.add_argument("--domain", choices=("product", "anime", "pastel"), default="product")
    parser.add_argument("--prompts", help="optional text file, one prompt per line")
    parser.add_argument("--kernel-spec")
    parser.add_argument("--store", help="marketplace store JSON")
    parser.add_argument("--listing-id")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-blocks", type=int, default=19)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--promote", action="store_true",
                        help="promote the matching listing after a passing gate")
    args = parser.parse_args()

    listing, kernel_spec, prompts, declared_split = _load_inputs(args)
    out_dir = args.out or os.path.join(
        HERE, "validation_runs", time.strftime("%Y%m%d_%H%M%S")
    )
    baseline = ComfyUIAdapter(
        base_url=args.comfy_url, checkpoint=args.checkpoint,
        width=args.width, height=args.height, steps=args.steps,
        modulation_mode="none", n_blocks=args.n_blocks,
    )
    kernel = ComfyUIAdapter(
        base_url=args.comfy_url, checkpoint=args.checkpoint,
        width=args.width, height=args.height, steps=args.steps,
        modulation_mode="calibrix_node", n_blocks=args.n_blocks,
    )
    kernel.apply_modulation(_kernel_specs(kernel_spec, args.n_blocks))
    probe = _probe(kernel, kernel_spec)
    scorers = [OfflineImageQuality(prompts), OfflineColorAlignment(prompts)]
    report = validate_real_kernel(
        baseline, kernel, kernel_spec, prompts, scorers, out_dir,
        config=ValidationConfig(
            generation_seed=args.seed,
            primary_metric="OfflineImageQuality",
            train_indices=declared_split[0] if declared_split else None,
            holdout_indices=declared_split[1] if declared_split else None,
        ),
        backend_probe=probe,
    )

    promoted = False
    if args.promote and report["gate"]["passed"]:
        if not args.store or not args.listing_id:
            raise ValueError("--promote requires --store and --listing-id")
        store = JsonStore(args.store)
        primary = report["gate"]["primary_metric"]
        store.promote_verified_listing(args.listing_id, {
            "status": "verified",
            "quality_gate": "PASS",
            "kernel_checksum": report["kernel_checksum"],
            "report_path": os.path.abspath(os.path.join(out_dir, "validation.json")),
            "holdout_score": report["scores"]["holdout"]["kernel"][primary],
            "baseline_score": report["scores"]["holdout"]["baseline"][primary],
            "scoring": report["scores"]["holdout"]["kernel"],
            "note": "Real ComfyUI paired holdout passed; customer ROI is not established.",
        })
        promoted = True

    print(json.dumps({
        "status": report["evidence"]["status"],
        "quality_gate": report["gate"]["quality_gate"],
        "promoted": promoted,
        "out": os.path.abspath(out_dir),
        "reasons": report["gate"]["reasons"],
    }, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
