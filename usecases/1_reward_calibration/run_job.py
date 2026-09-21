# UC-1 hosted calibration job runner (self-contained use-case script).
#
# Flow: quote (ex-ante) -> calibration run -> artifacts (kernel spec +
# report) -> ledger reconciliation (ex-post). Fully offline via
# ScriptedAdapter; swap the adapter for CalibriFluxAdapter to go live.

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
from calibrix.metering import (
    MODEL_PRESETS,
    QuoteConfig,
    format_quote_text,
    quote_from_preset,
)
from calibrix.report import write_report
from calibrix.scorers import (
    DiversityDrop,
    EmptyRate,
    KeywordRate,
    LengthDrift,
    seed_prompts,
)


PROMPTS = [
    "a red boat on a blue lake at dawn",
    "a cyberpunk street market in the rain",
    "a watercolor painting of a lighthouse",
    "an astronaut riding a horse, photorealistic",
    "a cozy cabin in a snowy forest",
    "a plate of sushi, studio lighting",
    "a hot air balloon over sand dunes",
    "a jazz band playing in a smoky club",
    "a macro photo of a dew-covered spiderweb",
    "a medieval castle on a cliff at sunset",
    "a field of sunflowers under storm clouds",
    "a vintage car on a desert highway",
]


def main() -> int:
    p = argparse.ArgumentParser(description="UC-1 hosted calibration job")
    p.add_argument("--preset", default="flux-dev-gates",
                   choices=sorted(MODEL_PRESETS))
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--optimizer", default="tpe", choices=["simple", "cma", "tpe"])
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join("jobs", f"{ts}_{args.preset}")
    os.makedirs(job_dir, exist_ok=True)

    # 1. ex-ante quote (what the customer is charged)
    quote = quote_from_preset(
        args.preset, n_trials=args.trials, prompts_per_eval=len(PROMPTS),
        popsize=args.popsize,
        quote=QuoteConfig(margin=args.margin),
        enforce_min_power=False,
    )
    with open(os.path.join(job_dir, "quote.json"), "w", encoding="utf-8") as f:
        json.dump(quote, f, indent=2)

    # 2. calibration run (offline adapter; swap for CalibriFluxAdapter live)
    prompts = seed_prompts(PROMPTS)
    adapter = ScriptedAdapter(n_layers=MODEL_PRESETS[args.preset]["n_sites"])
    scorers = [
        KeywordRate(prompts, score_name="PromptAdherence"),
        LengthDrift(prompts),
        EmptyRate(prompts),
        DiversityDrop(prompts),
    ]
    cfg = SearchConfig(
        n_trials=args.trials, popsize=args.popsize, optimizer=args.optimizer,
        seed=0, holdout_fraction=0.25,
        log_path=os.path.join(job_dir, "events.jsonl"),
        meter_path=os.path.join(job_dir, "ledger.json"),
        objective_weights={"PromptAdherence": 2.0},
    )
    engine = SearchEngine(adapter, scorers, prompts, cfg)
    result = engine.run()
    paths = write_report(result, adapter_name=f"uc1({args.preset})",
                         scorer_names=[s.score_name for s in scorers],
                         out_dir=job_dir)

    # 3. portable kernel spec (ComfyUI-node consumable)
    best = result["best"]
    sites = adapter.spec_sites()
    unpack_spec_vector(sites, best.vector)
    comp_gains = {s.component: s.kernel for s in sites}
    spec_str = "|".join(
        f"{comp}:{k.weight:.3f}@{k.position:.3f}:{k.floor:.3f}:{k.focus:.3f}"
        for comp, k in comp_gains.items()
    )
    with open(os.path.join(job_dir, "kernel_spec.txt"), "w", encoding="utf-8") as f:
        f.write(spec_str + "\n")

    # 4. reconciliation: quoted vs actual compute
    ledger = json.load(open(os.path.join(job_dir, "ledger.json"), encoding="utf-8"))
    ticket = {
        "job": job_dir,
        "preset": args.preset,
        "quoted_total": quote["total"],
        "actual_steps": ledger["totals"]["steps"],
        "actual_seconds": round(ledger["totals"]["seconds"], 2),
        "overfit_flagged": result["overfit"]["flagged"],
        "kernel_spec": spec_str,
        "deliverables": [paths["json"], paths["html"],
                         os.path.join(job_dir, "kernel_spec.txt")],
    }
    with open(os.path.join(job_dir, "job_ticket.json"), "w", encoding="utf-8") as f:
        json.dump(ticket, f, indent=2)

    if args.json:
        print(json.dumps(ticket, indent=2))
    else:
        print(format_quote_text(quote, preset=args.preset))
        print()
        print(f"best fitness     : {best.fitness:.4f}")
        print(f"overfit alarm    : {'FLAGGED' if result['overfit']['flagged'] else 'clear'}")
        print(f"kernel spec      : {spec_str}")
        print(f"deliverables in  : {job_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
