# UC-8 open-core install check (self-contained).
#
# Probes the environment, runs a minimal offline search to prove the core
# works, and prints the feature matrix mapping this install to the
# commercial verticals. No network, no GPU, ~2 seconds.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "calibrix"))

from calibrix.adapters import ScriptedAdapter
from calibrix.engine import SearchConfig, SearchEngine
from calibrix.scorers import EmptyRate, KeywordRate, seed_prompts


def probe(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="UC-8 Calibrix install check")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    # 1. core smoke test: a real (tiny) offline search
    prompts = seed_prompts([
        "hello world", "tell me a fact", "explain gravity",
        "write a haiku", "name a color", "describe the ocean",
    ])
    engine = SearchEngine(
        ScriptedAdapter(n_layers=6),
        [KeywordRate(prompts), EmptyRate(prompts)],
        prompts,
        SearchConfig(n_trials=2, popsize=3, optimizer="simple", seed=0),
    )
    result = engine.run()
    core_ok = result["best"] is not None

    # 2. optional stacks
    stacks = {m: probe(m) for m in
              ("numpy", "torch", "diffusers", "cma", "optuna", "stripe")}

    # 3. vertical -> requirements -> tier
    verticals = {
        "1_reward_calibration": {
            "can_run": core_ok,
            "live_requires": ["torch", "diffusers"],
            "tier": "hosted calibration jobs ($200-2k/run)",
        },
        "2_nfe_cost_reduction": {
            "can_run": core_ok,
            "live_requires": ["torch", "diffusers"],
            "tier": "savings retainers (% of savings)",
        },
        "3_domain_kernels": {
            "can_run": core_ok,
            "live_requires": ["torch", "diffusers"],
            "tier": "per-kernel sales",
        },
        "4_kernel_marketplace": {
            "can_run": core_ok and stacks["stripe"] or core_ok,  # mock works too
            "live_requires": ["stripe"],
            "tier": "marketplace margin ($3-20/kernel)",
        },
        "5_abliteration_service": {
            "can_run": core_ok,
            "live_requires": ["torch"],
            "tier": "per-run GPU minutes",
        },
        "6_compliance_restoration": {
            "can_run": core_ok,
            "live_requires": ["torch"],
            "tier": "enterprise per-report",
        },
        "7_interp_reports": {
            "can_run": core_ok,
            "live_requires": ["torch"],
            "tier": "per-model reports",
        },
        "8_open_core": {
            "can_run": core_ok,
            "live_requires": [],
            "tier": "free funnel",
        },
    }

    report = {
        "core_search_ok": core_ok,
        "stacks": stacks,
        "verticals": verticals,
        "missing_for_full_live": sorted(
            {m for v in verticals.values() for m in v["live_requires"]
             if not stacks[m]}
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("Calibrix install check")
    print("======================")
    print(f"core offline search : {'OK' if core_ok else 'FAILED'}")
    print("optional stacks     :",
          ", ".join(f"{m}={'yes' if ok else 'no'}" for m, ok in stacks.items()))
    print()
    print("Use-case verticals runnable with THIS install:")
    for name, v in verticals.items():
        status = "demo-ready" if v["can_run"] else "unavailable"
        print(f"  {name:24s} {status:12s} -> {v['tier']}")
    missing = report["missing_for_full_live"]
    if missing:
        print(f"\ninstall {'pip install calibrix[' + ','.join(missing) + ']' if len(missing) > 1 else 'pip install ' + missing[0]} to unlock live runs")
    print("\nfree & offline core (MIT). Paid tiers: hosted jobs, retainers,")
    print("marketplace, compliance reports - see docs/product_opportunities.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
