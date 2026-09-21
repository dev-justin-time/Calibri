# Calibrix metering: explicit, monotizable compute accounting.

# "Monetizable functions" (the user's term) implemented as: every search run
# produces an auditable cost ledger (steps x steps_cost x $/step) plus an
# ex-ante budget planner that answers "what will this calibration cost?".
# Heretic/Calibri both report wall-clock only; neither can tell you what a
# run costs or whether a "better" trial just burned more compute.

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CostModel:
    """Price inputs for one hosted/local generation step.

    steps_cost: machine steps per generation unit reported by the adapter
                (NFE for diffusion, ~1 for LLM requests).
    price_per_1k_steps: USD per 1000 billed steps (0.0 for local hardware).
    """

    steps_cost: float = 1.0
    price_per_1k_steps: float = 0.0
    currency: str = "USD"

    def price(self, steps: int) -> float:
        return steps / 1000.0 * self.price_per_1k_steps


@dataclass
class UsageRecord:
    trial: int
    phase: str            # "baseline" | "search" | "validation" | "export"
    steps: int
    seconds: float
    price: float
    scorer_calls: int = 0


class Meter:
    """Tracks steps/seconds/price per phase; serializes to a ledger JSON."""

    def __init__(self, cost_model: Optional[CostModel] = None) -> None:
        self.cost_model = cost_model or CostModel()
        self.records: List[UsageRecord] = []
        self._t0 = time.perf_counter()

    def mark(self, trial: int, phase: str, steps: int, seconds: float,
             scorer_calls: int = 0) -> None:
        self.records.append(
            UsageRecord(
                trial=trial,
                phase=phase,
                steps=int(steps),
                seconds=float(seconds),
                price=self.cost_model.price(steps),
                scorer_calls=int(scorer_calls),
            )
        )

    # -- aggregations ------------------------------------------------------
    def totals(self) -> Dict[str, float]:
        t: Dict[str, float] = {"steps": 0, "seconds": 0.0, "price": 0.0,
                               "scorer_calls": 0, "trials": 0}
        for r in self.records:
            t["steps"] += r.steps
            t["seconds"] += r.seconds
            t["price"] += r.price
            t["scorer_calls"] += r.scorer_calls
            t["trials"] = max(t["trials"], r.trial + 1)
        return t

    def by_phase(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            d = out.setdefault(r.phase, {"steps": 0.0, "seconds": 0.0, "price": 0.0})
            d["steps"] += r.steps
            d["seconds"] += r.seconds
            d["price"] += r.price
        return out

    def ledger(self) -> Dict[str, Any]:
        return {
            "cost_model": asdict(self.cost_model),
            "totals": self.totals(),
            "by_phase": self.by_phase(),
            "records": [asdict(r) for r in self.records],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ledger(), f, indent=2)


# ---------------------------------------------------------------------------
# Budget planner (ex-ante estimate before launching a run)
# ---------------------------------------------------------------------------

def plan_budget(
    n_sites: int,
    n_components: int,
    n_trials: int,
    prompts_per_eval: int,
    steps_per_gen: float,
    popsize: int,
    price_per_1k_steps: float = 0.0,
    baseline_evals: int = 2,
    holdout_evals: int = 1,
) -> Dict[str, Any]:
    """Estimate the cost of a Calibrix search before running it.

    Formula: each optimizer generation evaluates `popsize` candidates; each
    candidate needs one scored evaluation pass over `prompts_per_eval`
    prompts, each generation costing ~steps_per_gen * batch inference steps.
    Kernel params are O(n_sites * n_components * 6) - the planner surfaces
    how dimensionality drives trial count (the Calibri paper's core insight).
    """
    kernel_params = n_sites * n_components * 6
    evals = n_trials * popsize + baseline_evals + holdout_evals
    steps = evals * prompts_per_eval * steps_per_gen
    price = steps / 1000.0 * price_per_1k_steps
    # Rule of thumb: >= 10 generations per free parameter for ES to converge.
    min_trials = max(10, math_ceil(10 * kernel_params / max(popsize, 1)))
    return {
        "kernel_params": kernel_params,
        "evals": evals,
        "steps": steps,
        "price": round(price, 4),
        "recommended_min_trials": min_trials,
        "underpowered": n_trials < min_trials,
        "notes": [
            f"6 params/site/component x {n_sites} sites x {n_components} comps "
            f"= {kernel_params} free parameters",
            f"{evals} scored evaluations x {prompts_per_eval} prompts "
            f"x {steps_per_gen} steps = {int(steps)} generation steps",
        ],
    }


def math_ceil(x: float) -> int:
    import math

    return math.ceil(x)


# ---------------------------------------------------------------------------
# Quote calculator: turns a plan_budget estimate into a customer-facing quote
# for hosted calibration jobs (compute at cost + margin, prep overhead,
# minimum fee, rush surcharge, power-corrected pricing).
# ---------------------------------------------------------------------------

# Presets for supported hosted models. n_sites/n_components describe the
# calibration surface; steps is default NFE; price is $/1k generation steps
# for hosted GPU time.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "flux-dev-gates": {
        "n_sites": 19, "n_components": 1, "steps": 15.0,
        "price_per_1k_steps": 1.50, "description": "FLUX.1-dev double-block gates (attn+mlp per block)",
    },
    "flux-dev-block": {
        "n_sites": 19, "n_components": 1, "steps": 15.0,
        "price_per_1k_steps": 1.50, "description": "FLUX.1-dev uniform block scaling",
    },
    "sd35-medium-gates": {
        "n_sites": 24, "n_components": 2, "steps": 15.0,
        "price_per_1k_steps": 0.90, "description": "SD3.5-Medium MM-DiT main/context gates",
    },
    "sd35-large-gates": {
        "n_sites": 38, "n_components": 2, "steps": 30.0,
        "price_per_1k_steps": 2.10, "description": "SD3.5-Large MM-DiT main/context gates",
    },
    "qwen-image-gates": {
        "n_sites": 60, "n_components": 2, "steps": 30.0,
        "price_per_1k_steps": 2.60, "description": "Qwen-Image transformer main/context gates",
    },
}


@dataclass
class QuoteConfig:
    """Business-side pricing knobs (all USD)."""

    margin: float = 0.30                 # fraction of compute charged as margin
    prep_hours: float = 1.5              # engineer time to set up prompts/scorers
    hourly_rate: float = 120.0           # $/hour for prep time
    minimum_fee: float = 250.0           # floor per job
    rush_multiplier: float = 1.5         # applied when rush=True
    rush_fee: float = 150.0              # flat adder when rush=True
    power_trials_factor: float = 2.0     # trials multiplier when fixing underpowered plans
    currency: str = "USD"


def _apply_min_power(n_trials: int, kernel_params: int, popsize: int,
                     factor: float) -> int:
    """Trials needed to reach the >= 10 generations per parameter heuristic."""
    min_trials = max(10, math_ceil(10 * kernel_params / max(popsize, 1)))
    return max(n_trials, min_trials)


def quote_job(
    n_sites: int,
    n_components: int,
    n_trials: int,
    prompts_per_eval: int,
    steps_per_gen: float,
    popsize: int,
    price_per_1k_steps: float,
    quote: Optional[QuoteConfig] = None,
    enforce_min_power: bool = True,
    baseline_evals: int = 2,
    holdout_evals: int = 1,
    rush: bool = False,
) -> Dict[str, Any]:
    """Build a customer-ready quote from a plan_budget-style estimate.

    Returns line items (compute, prep, rush) and totals with margin and
    minimum-fee logic applied. If the requested trial count is below the
    convergence heuristic and `enforce_min_power` is set, the quote is
    computed at the recommended trial count and both tiers are reported.
    """
    q = quote or QuoteConfig()

    plan = plan_budget(
        n_sites=n_sites,
        n_components=n_components,
        n_trials=n_trials,
        prompts_per_eval=prompts_per_eval,
        steps_per_gen=steps_per_gen,
        popsize=popsize,
        price_per_1k_steps=price_per_1k_steps,
        baseline_evals=baseline_evals,
        holdout_evals=holdout_evals,
    )

    requested_underpowered = plan["underpowered"]
    quoted_trials = n_trials
    if enforce_min_power and requested_underpowered:
        quoted_trials = _apply_min_power(
            n_trials, plan["kernel_params"], popsize, q.power_trials_factor
        )
        plan = plan_budget(
            n_sites=n_sites,
            n_components=n_components,
            n_trials=quoted_trials,
            prompts_per_eval=prompts_per_eval,
            steps_per_gen=steps_per_gen,
            popsize=popsize,
            price_per_1k_steps=price_per_1k_steps,
            baseline_evals=baseline_evals,
            holdout_evals=holdout_evals,
        )

    compute_cost = plan["price"]
    margin_amount = compute_cost * q.margin
    prep_cost = q.prep_hours * q.hourly_rate
    subtotal = compute_cost + margin_amount + prep_cost

    rush_fee = 0.0
    if rush:
        subtotal *= q.rush_multiplier
        rush_fee = subtotal - (compute_cost + margin_amount + prep_cost)

    total = max(subtotal, q.minimum_fee)
    minimum_applied = subtotal < q.minimum_fee

    lines = [
        {"item": "GPU compute", "detail": (
            f"{plan['evals']} evals x {prompts_per_eval} prompts x "
            f"{steps_per_gen:g} steps @ ${price_per_1k_steps:.2f}/1k steps"
        ), "amount": round(compute_cost, 2)},
        {"item": "Service margin", "detail": f"{q.margin:.0%} of compute", "amount": round(margin_amount, 2)},
        {"item": "Setup & prompt engineering", "detail": f"{q.prep_hours:g} h @ ${q.hourly_rate:.0f}/h", "amount": round(prep_cost, 2)},
    ]
    if rush:
        lines.append({"item": "Rush surcharge", "detail": f"x{q.rush_multiplier:g} subtotal", "amount": round(rush_fee, 2)})

    quote_out = {
        "currency": q.currency,
        "model": {
            "n_sites": n_sites,
            "n_components": n_components,
            "kernel_params": plan["kernel_params"],
        },
        "requested": {
            "n_trials": n_trials,
            "underpowered": requested_underpowered,
            "quoted_trials": quoted_trials,
        },
        "line_items": lines,
        "subtotal": round(subtotal, 2),
        "minimum_fee": q.minimum_fee if minimum_applied else None,
        "minimum_fee_applied": minimum_applied,
        "total": round(total, 2),
        "steps": int(plan["steps"]),
        "evals": plan["evals"],
    }
    return quote_out


def quote_from_preset(
    preset: str,
    n_trials: int = 12,
    prompts_per_eval: int = 64,
    popsize: int = 8,
    quote: Optional[QuoteConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Quote a hosted calibration job using a MODEL_PRESETS entry."""
    if preset not in MODEL_PRESETS:
        raise KeyError(
            f"unknown preset '{preset}'; available: {', '.join(sorted(MODEL_PRESETS))}"
        )
    p = MODEL_PRESETS[preset]
    return quote_job(
        n_sites=p["n_sites"],
        n_components=p["n_components"],
        n_trials=n_trials,
        prompts_per_eval=prompts_per_eval,
        steps_per_gen=p["steps"],
        popsize=popsize,
        price_per_1k_steps=p["price_per_1k_steps"],
        quote=quote,
        **kwargs,
    )


def format_quote_text(q: Dict[str, Any], preset: Optional[str] = None) -> str:
    """Render a quote dict as customer-facing plain text."""
    title = "Calibrix hosted calibration quote" + (f" - {preset}" if preset else "")
    lines_out = [title, "=" * len(title)]
    m = q["model"]
    lines_out.append(
        f"Model surface: {m['n_sites']} sites x {m['n_components']} components "
        f"= {m['kernel_params']} free parameters"
    )
    if q["requested"].get("underpowered"):
        lines_out.append(
            f"Trials: requested {q['requested']['n_trials']}, quoted "
            f"{q['requested']['quoted_trials']} (convergence-corrected)"
        )
    else:
        lines_out.append(f"Trials: {q['requested']['n_trials']}")
    lines_out.append("")
    for li in q["line_items"]:
        lines_out.append(f"  {li['item']:<28} {li['detail']:<52} ${li['amount']:>10,.2f}")
    lines_out.append("  " + "-" * 94)
    lines_out.append(f"  {'Subtotal':<28} {'':<52} ${q['subtotal']:>10,.2f}")
    if q["minimum_fee_applied"]:
        lines_out.append(f"  {'Minimum job fee':<28} {'floor applied':<52} ${q['total']:>10,.2f}")
    lines_out.append(f"  {'TOTAL':<28} {'':<52} ${q['total']:>10,.2f} {q['currency']}")
    return "\n".join(lines_out)
