# Calibrix CLI.
#
# All commands run fully offline by default (the demo uses the scripted
# adapter), so anyone can validate the framework in seconds without a GPU,
# an API key, or a network. Real adapters plug into the same entry point.

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .engine import SearchConfig, SearchEngine
from .metering import MODEL_PRESETS, format_quote_text, plan_budget, quote_from_preset, quote_job, QuoteConfig
from .report import write_report
from .scorers import (
    DiversityDrop,
    EmptyRate,
    KLDrift,
    KeywordRate,
    LengthDrift,
    seed_prompts,
)


DEMO_PROMPTS = [
    "Explain how a transformer attention layer works.",
    "Write a short poem about the sea.",
    "Summarize the plot of Hamlet.",
    "What causes rainbows?",
    "Give tips for writing clear documentation.",
    "Describe the taste of coffee.",
    "How do bicycles stay upright?",
    "Name three uses for cardboard boxes.",
    "Explain recursion to a beginner.",
    "What is the freezing point of water?",
    "Describe a busy market street.",
    "Why do leaves change color in autumn?",
]


def cmd_demo(args: argparse.Namespace) -> int:
    from .adapters import ScriptedAdapter

    prompts = seed_prompts(DEMO_PROMPTS)
    adapter = ScriptedAdapter(n_layers=12)

    scorers = [
        KeywordRate(prompts, score_name="Refusals"),
        KLDrift(prompts),
        LengthDrift(prompts),
        EmptyRate(prompts),
        DiversityDrop(prompts),
    ]

    cfg = SearchConfig(
        n_trials=args.trials,
        popsize=args.popsize,
        optimizer=args.optimizer,
        seed=args.seed,
        holdout_fraction=0.25,
        log_path=f"{args.out}/events.jsonl",
        meter_path=f"{args.out}/ledger.json",
    )
    engine = SearchEngine(adapter, scorers, prompts, cfg)
    result = engine.run()

    paths = write_report(
        result, adapter_name="ScriptedAdapter (offline)",
        scorer_names=[s.score_name for s in scorers], out_dir=args.out,
    )

    best = result["best"]
    print("Calibrix demo (offline, no GPU required)")
    print(f"  free parameters : {result['n_params']}")
    print(f"  evaluations     : {result['n_evals']}")
    print(f"  best fitness    : {best.fitness:.4f}")
    print(f"  overfit alarm   : {'FLAGGED' if result['overfit']['flagged'] else 'clear'}")
    print(f"  report          : {paths['html']}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan = plan_budget(
        n_sites=args.sites,
        n_components=args.components,
        n_trials=args.trials,
        prompts_per_eval=args.prompts,
        steps_per_gen=args.steps,
        popsize=args.popsize,
        price_per_1k_steps=args.price,
    )
    print(json.dumps(plan, indent=2))
    if plan["underpowered"]:
        print(
            f"\nNOTE: {args.trials} trials is below the recommended minimum "
            f"of {plan['recommended_min_trials']} for {plan['kernel_params']} "
            "free parameters. Expect slow convergence."
        )
    return 0


def cmd_quote(args: argparse.Namespace) -> int:
    qc = QuoteConfig(
        margin=args.margin,
        hourly_rate=args.rate,
        prep_hours=args.prep,
        minimum_fee=args.min_fee,
        rush_multiplier=args.rush_mult,
        rush_fee=args.rush_fee,
    )
    kwargs = {"rush": args.rush, "enforce_min_power": not args.no_min_power}
    if args.preset:
        q = quote_from_preset(args.preset, n_trials=args.trials,
                              prompts_per_eval=args.prompts, popsize=args.popsize,
                              quote=qc, **kwargs)
        model_label: str | None = args.preset
    else:
        if any(v is None for v in (args.sites, args.components, args.steps, args.price)):
            print("error: either --preset or all of --sites/--components/--steps/--price are required",
                  file=sys.stderr)
            return 2
        q = quote_job(
            n_sites=args.sites,
            n_components=args.components,
            n_trials=args.trials,
            prompts_per_eval=args.prompts,
            steps_per_gen=args.steps,
            popsize=args.popsize,
            price_per_1k_steps=args.price,
            quote=qc,
            **kwargs,
        )
        model_label = None

    if args.json:
        print(json.dumps(q, indent=2))
    else:
        print(format_quote_text(q, preset=model_label))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            if args.out.endswith(".json"):
                json.dump(q, f, indent=2)
            else:
                f.write(format_quote_text(q, preset=model_label))
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="calibrix",
        description="Model-agnostic parametric modulation + co-objective search",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the offline demo end-to-end")
    d.add_argument("--out", default="calibrix_report")
    d.add_argument("--trials", type=int, default=6)
    d.add_argument("--popsize", type=int, default=6)
    d.add_argument("--optimizer", default="simple",
                   choices=["simple", "cma", "tpe"])
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    pl = sub.add_parser("plan", help="estimate the cost of a calibration run")
    pl.add_argument("--sites", type=int, default=20)
    pl.add_argument("--components", type=int, default=2)
    pl.add_argument("--trials", type=int, default=12)
    pl.add_argument("--prompts", type=int, default=64)
    pl.add_argument("--steps", type=float, default=15.0)
    pl.add_argument("--popsize", type=int, default=8)
    pl.add_argument("--price", type=float, default=0.0)
    pl.set_defaults(fn=cmd_plan)

    qt = sub.add_parser("quote", help="customer-ready quote for a hosted calibration job")
    qt.add_argument("--preset", default=None, choices=sorted(MODEL_PRESETS),
                    help="hosted model preset (flux-dev-gates, qwen-image-gates, ...)")
    qt.add_argument("--sites", type=int, default=None, help="calibration sites (custom model)")
    qt.add_argument("--components", type=int, default=None, help="components per site (custom model)")
    qt.add_argument("--trials", type=int, default=12)
    qt.add_argument("--prompts", type=int, default=64)
    qt.add_argument("--popsize", type=int, default=8)
    qt.add_argument("--steps", type=float, default=None, help="NFE per generation (custom model)")
    qt.add_argument("--price", type=float, default=None, help="USD per 1k generation steps (custom model)")
    qt.add_argument("--margin", type=float, default=0.30, help="service margin as fraction of compute")
    qt.add_argument("--rate", type=float, default=120.0, help="hourly rate for setup/prep")
    qt.add_argument("--prep", type=float, default=1.5, help="prep hours")
    qt.add_argument("--min-fee", type=float, default=250.0, help="minimum job fee (USD)")
    qt.add_argument("--rush", action="store_true", help="apply rush pricing")
    qt.add_argument("--rush-mult", type=float, default=1.5, help="rush subtotal multiplier")
    qt.add_argument("--rush-fee", type=float, default=150.0, help="flat rush fee component")
    qt.add_argument("--no-min-power", action="store_true",
                    help="quote the requested trial count even if below convergence heuristic")
    qt.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    qt.add_argument("--out", default=None, help="write quote to file (.json = JSON, else text)")
    qt.set_defaults(fn=cmd_quote)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
