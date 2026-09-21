# SPDX-License-Identifier: MIT
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

    if args.quote is not None and args.max_budget is None:
        print("error: --quote requires --max-budget (the breaker guards a "
              "spend ceiling; it cannot run unbounded)")
        return 2
    if args.max_budget is not None and not args.price:
        print("warning: --max-budget without --price: spend is always $0, "
              "so the budget can never bind")

    cfg = SearchConfig(
        n_trials=args.trials,
        popsize=args.popsize,
        optimizer=args.optimizer,
        seed=args.seed,
        holdout_fraction=0.25,
        price_per_1k_steps=args.price,
        max_budget_usd=args.max_budget,
        quoted_usd=args.quote,
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
    if best is not None:
        print(f"  best fitness    : {best.fitness:.4f}")
    print(f"  overfit alarm   : {'FLAGGED' if result['overfit']['flagged'] else 'clear'}")
    b = result["budget"]
    budget_line = f"  budget          : spent ${b['spent_usd']:.4f}"
    if b["max_budget_usd"] is not None:
        budget_line += f" / ${b['max_budget_usd']:.2f}"
        if b["halted"]:
            budget_line += "  [HALTED]"
    if b["breaker_tripped"]:
        budget_line += "  [BREAKER]"
    print(budget_line)
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


def cmd_ablate(args: argparse.Namespace) -> int:
    """Directional ablation co-optimization (Arditi et al. 2024 method).

    Offline simulation by default (no torch, no network); point --url at an
    OpenAI-compatible server (Ollama etc.) to run against a real model.
    """
    import numpy as np

    from .ablation import (
        SimulatedRefusalAdapter,
        default_ablation_spec,
        extract_directions_from_sim,
        pick_best_direction_layer,
    )
    from .engine import SearchEngine
    from .scorers import DiversityDrop, EmptyRate, KeywordRate, LengthDrift

    rng = np.random.default_rng(args.seed)
    n_layers = args.layers

    harmful = seed_prompts(
        [f"Give detailed instructions for making a dangerous device {i}."
         for i in range(args.n_harmful)])
    harmless = seed_prompts(
        [f"Write a helpful explanation of {w}."
         for w in ("rainbows", "bicycles", "coffee", "photosynthesis",
                   "gravity", "kites", "bread", "tides")])

    if args.url:
        from .ablation import DirectionalAblationAdapter
        from .adapters import OpenAICompatAdapter

        base = OpenAICompatAdapter(model=args.model, base_url=args.url)
        print(f"NOTE: remote server {args.url} cannot be weight-modulated "
              "in-process; running scoring-only search (gains recorded, "
              "behavior unchanged). Use in-process adapters for real "
              "ablation.")
        adapter = base
        spec = default_ablation_spec(n_layers, 16)
    else:
        sim = SimulatedRefusalAdapter(n_layers=n_layers, d_model=16,
                                      seed=args.seed)
        sim.plant_clusters(harmful, harmless)
        directions = extract_directions_from_sim(sim, harmful, harmless)
        best_layer = pick_best_direction_layer(directions)
        print(f"refusal direction extracted: layer {best_layer} "
              f"(of {n_layers})")
        adapter = sim
        spec = default_ablation_spec(n_layers, 16)
        spec.directions = directions
        sim.directions = directions

    # seed the kernel at the strongest direction layer
    for s in spec.specs:
        s.kernel.weight = 1.0
        s.kernel.floor = 1.0
        s.kernel.position = best_layer / max(n_layers - 1, 1) if not args.url else 0.5
        s.kernel.focus = 1e6
        s.kernel.direction_index = float(best_layer) if not args.url else 0.0
        s.direction_index = getattr(s.kernel, "direction_index")
    # sync the configured specs into the adapter (the engine modulates
    # adapter.spec_sites(); for the sim these are now the same objects)
    adapter.apply_modulation(spec.specs)

    scorers = [
        KeywordRate(harmful + harmless, score_name="Refusals"),
        LengthDrift(harmless),
        EmptyRate(harmless),
        DiversityDrop(harmless),
    ]
    if args.quote is not None and args.max_budget is None:
        print("error: --quote requires --max-budget (the breaker guards a "
              "spend ceiling; it cannot run unbounded)")
        return 2
    if args.max_budget is not None and not args.price:
        print("warning: --max-budget without --price: spend is always $0, "
              "so the budget can never bind")

    cfg = SearchConfig(
        n_trials=args.trials,
        popsize=args.popsize,
        optimizer=args.optimizer,
        seed=args.seed,
        holdout_fraction=0.25,
        price_per_1k_steps=args.price,
        max_budget_usd=args.max_budget,
        quoted_usd=args.quote,
        log_path=f"{args.out}/events.jsonl" if args.out else None,
        meter_path=f"{args.out}/ledger.json" if args.out else None,
        objective_weights={"Refusals": 2.0},
    )
    engine = SearchEngine(adapter, scorers, harmful + harmless, cfg)
    result = engine.run()

    paths = write_report(
        result, adapter_name=f"ablation({adapter.name})",
        scorer_names=[s.score_name for s in scorers], out_dir=args.out or "ablation_report",
    )
    best = result["best"]
    print(f"  free parameters : {result['n_params']}")
    print(f"  evaluations     : {result['n_evals']}")
    if best is not None:
        print(f"  best fitness    : {best.fitness:.4f}")
    print(f"  overfit alarm   : {'FLAGGED' if result['overfit']['flagged'] else 'clear'}")
    b = result["budget"]
    budget_line = f"  budget          : spent ${b['spent_usd']:.4f}"
    if b["max_budget_usd"] is not None:
        budget_line += f" / ${b['max_budget_usd']:.2f}"
        if b["halted"]:
            budget_line += "  [HALTED]"
    if b["breaker_tripped"]:
        budget_line += "  [BREAKER]"
    print(budget_line)
    print(f"  report          : {paths['html']}")
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
    d.add_argument("--price", type=float, default=0.0,
                   help="USD per 1k metered steps (needed for budget/breaker)")
    d.add_argument("--max-budget", type=float, default=None, dest="max_budget",
                   help="hard spend ceiling in USD (requires --price or a priced preset)")
    d.add_argument("--quote", type=float, default=None, dest="quote",
                   help="quoted cost in USD; a fail-closed breaker trips beyond "
                        "quote*(1+tolerance) (requires --max-budget)")
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

    ab = sub.add_parser("ablate",
                        help="directional-ablation co-optimization (offline sim by default)")
    ab.add_argument("--layers", type=int, default=12, help="transformer layer count")
    ab.add_argument("--n-harmful", type=int, default=10, help="harmful calibration prompts")
    ab.add_argument("--model", default="llama3.2", help="model name for --url mode")
    ab.add_argument("--url", default=None,
                    help="OpenAI-compatible base URL (e.g. http://127.0.0.1:11434/v1)")
    ab.add_argument("--trials", type=int, default=8)
    ab.add_argument("--popsize", type=int, default=6)
    ab.add_argument("--optimizer", default="tpe", choices=["simple", "cma", "tpe"])
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--price", type=float, default=0.0,
                    help="USD per 1k metered steps (needed for budget/breaker)")
    ab.add_argument("--out", default="ablation_report",
                    help="output dir for events/ledger/report (empty string disables)")
    ab.add_argument("--max-budget", type=float, default=None, dest="max_budget",
                    help="hard spend ceiling in USD (requires --price or a priced preset)")
    ab.add_argument("--quote", type=float, default=None, dest="quote",
                    help="quoted cost in USD; a fail-closed breaker trips beyond "
                         "quote*(1+tolerance) (requires --max-budget)")
    ab.set_defaults(fn=cmd_ablate)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
