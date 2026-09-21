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
