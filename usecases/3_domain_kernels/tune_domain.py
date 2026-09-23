# UC-3 domain kernel tuner (self-contained).
#
# Calibrates a kernel against a domain prompt pack and exports a
# marketplace-ready listing (consumable by UC-4's storefront as-is) plus a
# ComfyUI-node-consumable kernel spec. Offline scorers keep the demo free;
# swap in CLIP-backed scorers for production runs.

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "calibrix"))

from calibrix.adapters import ScriptedAdapter
from calibrix.engine import SearchConfig, SearchEngine
from calibrix.kernel import unpack_spec_vector
from calibrix.report import write_report
from calibrix.scorers import (
    DiversityDrop,
    EmptyRate,
    KeywordRate,
    LengthDrift,
    Scorer,
    Score,
    seed_prompts,
)

HERE = os.path.dirname(os.path.abspath(__file__))


class DomainQuality(Scorer):
    """Stands in for a CLIP/aesthetic domain scorer.

    Uses deterministic prompt-conditioned length+content heuristics of the
    scripted adapter; in production replace get_score with a CLIP-score or
    aesthetic-model call over generated images.
    """

    optimization = "maximize"

    def __init__(self, prompts_spec) -> None:
        self.prompts_spec = prompts_spec

    def init(self, ctx) -> None:
        self.prompts = seed_prompts(self.prompts_spec)

    def get_score(self, ctx) -> Score:
        responses = ctx.get_responses(self.prompts)
        # reward non-empty, non-refusing, detailed answers
        good = sum(
            1 for r in responses
            if r.strip() and not r.lower().startswith("i'm sorry")
            and len(r.split()) >= 6
        )
        return Score(value=good / max(len(responses), 1),
                     display=f"{good}/{len(responses)}")


def main() -> int:
    p = argparse.ArgumentParser(description="UC-3 domain kernel tuner")
    p.add_argument("--domain", default="product")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    with open(os.path.join(HERE, "domain_packs.json"), encoding="utf-8") as f:
        packs = json.load(f)
    if args.list:
        for name, pack in packs.items():
            print(f"{name:10s} ${pack['price_cents'] / 100:>5.2f}  {pack['description']}")
        return 0
    if args.domain not in packs:
        print(f"unknown domain {args.domain!r}; use --list", file=sys.stderr)
        return 1
    pack = packs[args.domain]

    out_dir = os.path.join("kernels", f"{args.domain}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    prompts = seed_prompts(pack["prompts"])
    adapter = ScriptedAdapter(n_layers=19)
    scorers = [
        DomainQuality(prompts),
        KeywordRate(prompts, score_name="Refusals"),
        LengthDrift(prompts),
        EmptyRate(prompts),
        DiversityDrop(prompts),
    ]
    cfg = SearchConfig(
        n_trials=args.trials, popsize=args.popsize, optimizer="tpe", seed=0,
        holdout_fraction=0.25,
        log_path=os.path.join(out_dir, "events.jsonl"),
        meter_path=os.path.join(out_dir, "ledger.json"),
        objective_weights=pack["objective_weights"],
    )
    engine = SearchEngine(adapter, scorers, prompts, cfg)
    result = engine.run()
    paths = write_report(result, adapter_name=f"uc3({args.domain})",
                         scorer_names=[s.score_name for s in scorers],
                         out_dir=out_dir)

    best = result["best"]
    sites = adapter.spec_sites()
    unpack_spec_vector(sites, best.vector)
    spec_str = "|".join(
        f"{s.component}:{s.kernel.weight:.3f}@{s.kernel.position:.3f}:"
        f"{s.kernel.floor:.3f}:{s.kernel.focus:.3f}"
        for s in sites
    )
    with open(os.path.join(out_dir, "kernel_spec.txt"), "w", encoding="utf-8") as f:
        f.write(spec_str + "\n")

    holdout = best.holdout or best.objectives
    listing = {
        "listing_id": f"{args.domain}-kernel-v1",
        "title": f"{args.domain.capitalize()} Look Kernel",
        "model": "FLUX.1-dev",
        "kernel_spec": spec_str,
        "price_cents": pack["price_cents"],
        "description": pack["description"],
        "category": "product-photo" if args.domain == "product" else args.domain,
        "tags": ["ecommerce", "catalog", "studio"] if args.domain == "product" else [args.domain],
        "compatible_models": ["FLUX.1-dev"],
        "scoring": {k: round(v, 3) for k, v in holdout.items()},
        # This vertical currently uses ScriptedAdapter. Marking it as a
        # simulation prevents the marketplace from implying real image-model
        # or customer ROI evidence.
        "evidence_status": "simulation",
        "evidence_note": "Generated with ScriptedAdapter; replace with a real image adapter and reviewed holdout before claiming quality gains.",
        "report_path": paths["json"],
        "holdout_score": round(holdout.get("DomainQuality", 0.0), 3),
        "baseline_score": round(result.get("baseline_holdout", {}).get("DomainQuality", 0.0), 3),
        "quality_gate": "REVIEW",
        "overfit_flagged": result["overfit"]["flagged"],
    }
    with open(os.path.join(out_dir, "listing.json"), "w", encoding="utf-8") as f:
        json.dump(listing, f, indent=2)

    print(f"domain      : {args.domain}")
    print(f"best fitness: {best.fitness:.4f}")
    print(f"overfit     : {'FLAGGED' if result['overfit']['flagged'] else 'clear'}")
    print(f"kernel spec : {spec_str}")
    print(f"listing     : {os.path.join(out_dir, 'listing.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
