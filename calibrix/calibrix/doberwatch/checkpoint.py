# SPDX-License-Identifier: MIT
# Doberwatch Finance — checkpoint frequency: the better method.
#
# The first-order optimum for how often to checkpoint is the Young/Daly
# formula (Young 1974; Daly 2006). With a checkpoint cost ``C`` and a
# mean-time-between-failures ``mu`` for the whole job:
#
#     tau* = sqrt(2 * C * mu)        [Young/Daly]
#
# and the checkpoint overhead at that optimum is
#
#     waste = C / tau* = sqrt(C / (2 * mu)).
#
# For a cluster of N GPUs each failing at rate f (per hour),
#
#     mu = 1 / (N * f)   [hours]   =>   tau* = sqrt(2 * C / (N * f)).
#
# NOTE THE SCALING: tau* falls as 1/sqrt(N), not as 1/N (and certainly not as
# 1/N^2). A table whose optimum drops 16x when the cluster only grows 4x is
# not the Young/Daly optimum. ``check_interval_claims`` measures exactly that
# against the numbers in a source, so a wrong frequency table fails loudly
# instead of being copied into a runbook.
#
# This module also turns the formula into an actionable plan: environment caps
# (local NVMe vs network storage vs preemptible grace windows), the 5%-overhead
# rule, and the preemption-specific deadline (AWS 2 min, GCP 30 s).

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

CHECKPOINT_SCHEMA = "calibrix.doberwatch.checkpoint/1.0"

DEFAULT_FAILURE_RATE_PER_GPU_HOUR = 2 / 1000.0   # 2 failures / 1000 GPU-hours
HOURS_PER_MONTH = 720.0
OVERHEAD_ALERT_FRACTION = 0.05                   # the "5% rule"

# Provider preemption grace windows (published, verified 2026-09).
GRACE_PERIOD_S: Dict[str, float] = {"aws": 120.0, "gcp": 30.0}


@dataclass(frozen=True)
class Environment:
    name: str
    max_interval_s: float
    rationale: str
    source: str


ENVIRONMENTS: Dict[str, Environment] = {
    "cpu_memory": Environment(
        "cpu_memory", 300.0,
        "In-host CPU-memory checkpoints have a tiny write cost and near-zero "
        "throughput impact, so failure rate rather than I/O sets the interval.",
        "Gemini, SOSP'23 (Wang et al.)"),
    "local_nvme": Environment(
        "local_nvme", 6 * 3600.0,
        "Fast local disk: hours-scale intervals are economical; upload off-box "
        "asynchronously so a node loss does not lose the checkpoint.",
        "vendor/field guidance"),
    "network_storage": Environment(
        "network_storage", 4 * 3600.0,
        "Shared/network I/O contends with training traffic; keep intervals in "
        "the hours and never let writes starve the step loop.",
        "field guidance"),
    "spot": Environment(
        "spot", 900.0,
        "Preemptible capacity: the interval must fit inside the provider grace "
        "window, and the SIGTERM path must finish the write in time.",
        "AWS 2-minute notice; GCP 30-second preemption"),
}


@dataclass(frozen=True)
class FailureModel:
    n_gpus: int
    failure_rate_per_gpu_hour: float = DEFAULT_FAILURE_RATE_PER_GPU_HOUR

    def __post_init__(self) -> None:
        if self.n_gpus <= 0:
            raise ValueError("n_gpus must be positive")
        if self.failure_rate_per_gpu_hour < 0:
            raise ValueError("failure rate must be >= 0")

    def cluster_failure_rate_per_hour(self) -> float:
        return self.n_gpus * self.failure_rate_per_gpu_hour

    def mtbf_hours(self) -> float:
        rate = self.cluster_failure_rate_per_hour()
        return float("inf") if rate <= 0 else 1.0 / rate

    def mtbf_seconds(self) -> float:
        return self.mtbf_hours() * 3600.0


def young_daly_interval_s(checkpoint_cost_s: float, mtbf_s: float) -> float:
    """First-order optimal checkpoint interval: sqrt(2 * C * mu)."""
    if checkpoint_cost_s < 0 or mtbf_s < 0:
        raise ValueError("cost and MTBF must be >= 0")
    if checkpoint_cost_s == 0 or mtbf_s == 0 or math.isinf(mtbf_s):
        return float("inf")
    return math.sqrt(2.0 * checkpoint_cost_s * mtbf_s)


def expected_waste_fraction(checkpoint_cost_s: float, mtbf_s: float) -> float:
    """Checkpoint overhead at the optimum: sqrt(C / (2 * mu))."""
    if checkpoint_cost_s <= 0 or mtbf_s <= 0 or math.isinf(mtbf_s):
        return 0.0
    return math.sqrt(checkpoint_cost_s / (2.0 * mtbf_s))


def implied_overhead_s(interval_s: float, n_gpus: int,
                       failure_rate_per_gpu_hour: float = DEFAULT_FAILURE_RATE_PER_GPU_HOUR) -> float:
    """Invert Young/Daly: what checkpoint cost would make ``interval_s`` optimal?"""
    mtbf = FailureModel(n_gpus, failure_rate_per_gpu_hour).mtbf_seconds()
    return interval_s ** 2 / (2.0 * mtbf)


@dataclass
class CheckpointCostModel:
    """Young/Daly inputs: cluster size, checkpoint cost, restore cost, rate."""

    n_gpus: int
    overhead_s: float                                   # C: time to write
    restore_s: float = 0.0                              # mu_r: time to restore
    failure_rate_per_gpu_hour: float = DEFAULT_FAILURE_RATE_PER_GPU_HOUR

    def failure(self) -> FailureModel:
        return FailureModel(self.n_gpus, self.failure_rate_per_gpu_hour)

    def mtbf_s(self) -> float:
        return self.failure().mtbf_seconds()

    def optimal_interval_s(self) -> float:
        return young_daly_interval_s(self.overhead_s, self.mtbf_s())

    def waste_fraction(self) -> float:
        return expected_waste_fraction(self.overhead_s, self.mtbf_s())

    def waste_rate(self, interval_s: float) -> float:
        """Fraction of wall-clock lost to checkpointing + expected rework."""
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        mtbf = self.mtbf_s()
        if math.isinf(mtbf):
            return self.overhead_s / interval_s
        rework = (interval_s / 2.0 + self.restore_s) / mtbf
        return self.overhead_s / interval_s + rework

    def hourly_waste_usd(self, usd_per_hour: float, interval_s: Optional[float] = None) -> float:
        interval = interval_s if interval_s is not None else self.optimal_interval_s()
        if math.isinf(interval):
            return 0.0
        return max(0.0, float(usd_per_hour)) * self.waste_rate(interval)


def monthly_waste_usd(usd_per_hour: float, fraction: float,
                      hours_per_month: float = HOURS_PER_MONTH) -> float:
    """Straight arithmetic for the 'X% overhead costs Y/month' question."""
    if fraction < 0:
        raise ValueError("fraction must be >= 0")
    return max(0.0, float(usd_per_hour)) * fraction * float(hours_per_month)


def checkpoint_plan(model: CheckpointCostModel, environment: str = "local_nvme",
                    provider: Optional[str] = None,
                    usd_per_hour: float = 0.0) -> Dict[str, Any]:
    """Turn the formula into a concrete interval + operational actions."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"unknown environment {environment!r}")
    env = ENVIRONMENTS[environment]
    optimal = model.optimal_interval_s()
    cap = env.max_interval_s
    grace = GRACE_PERIOD_S.get(provider) if provider else None
    if grace is not None:
        cap = min(cap, grace)
    interval = min(optimal, cap)

    overhead_pct = (model.overhead_s / interval) if interval > 0 else 1.0
    over_tuned = overhead_pct > OVERHEAD_ALERT_FRACTION
    fits_grace = grace is None or model.overhead_s <= grace

    actions: List[str] = [
        "Write to local (or CPU-memory) storage first, then upload off-box "
        "asynchronously so a node loss does not lose the checkpoint.",
        "Keep at most 2-3 retained snapshots to bound storage cost.",
    ]
    if provider:
        actions.append(
            f"Register a SIGTERM handler that starts an async checkpoint "
            f"immediately; {provider.upper()} gives {grace:.0f}s.")
    if not fits_grace:
        actions.append(
            "Checkpoint does NOT fit the grace window: shrink the snapshot or "
            "switch to in-memory / per-iteration differential checkpointing.")
    if over_tuned:
        actions.append(
            f"Over-tuned: writes are {overhead_pct:.1%} of the loop (>5%). "
            "Lengthen the interval or cheapen the write.")

    return {
        "schema": CHECKPOINT_SCHEMA,
        "environment": environment,
        "provider": provider,
        "interval_s": interval,
        "interval_hours": (interval / 3600.0) if not math.isinf(interval) else None,
        "optimal_interval_s": optimal,
        "environment_cap_s": env.max_interval_s,
        "grace_period_s": grace,
        "waste_fraction_at_optimum": model.waste_fraction(),
        "waste_fraction_at_choice": model.waste_rate(interval) if interval > 0 else 1.0,
        "overhead_pct": overhead_pct,
        "over_tuned": over_tuned,
        "fits_grace": fits_grace,
        "optimal_hourly_waste_usd": model.hourly_waste_usd(usd_per_hour),
        "optimal_monthly_waste_usd": monthly_waste_usd(
            usd_per_hour, model.waste_fraction()),
        "actions": actions,
        "source": env.source,
        "rationale": env.rationale,
    }


# ----------------------------------------------------------------------
# fact-check a claimed frequency table against the formula
# ----------------------------------------------------------------------
# The claims below are the frequencies in the source table being checked
# (cluster size -> "optimal frequency"), all at 2 failures / 1000 GPU-hours.
CHECKPOINT_FREQUENCY_CLAIMS: Sequence[Dict[str, Any]] = (
    {"n_gpus": 4, "claimed_s": 3 * 3600.0, "claimed": "~every 3 hours"},
    {"n_gpus": 16, "claimed_s": 11 * 60.0, "claimed": "~every 11 minutes"},
    {"n_gpus": 64, "claimed_s": 3 * 60.0, "claimed": "~every 3 minutes"},
)


def check_interval_claims(claims: Sequence[Dict[str, Any]] = CHECKPOINT_FREQUENCY_CLAIMS,
                          failure_rate_per_gpu_hour: float = DEFAULT_FAILURE_RATE_PER_GPU_HOUR
                          ) -> Dict[str, Any]:
    """Do a claimed frequency table's rows follow one Young/Daly cost?

    If the 4/16/64-GPU rows came from a single (C, f) model they would imply
    the *same* checkpoint cost. This reports the cost each row implies and
    flags the spread, so a table that violates the sqrt scaling is not trusted.
    """
    rows: List[Dict[str, Any]] = []
    implied: List[float] = []
    for claim in claims:
        n = int(claim["n_gpus"])
        claimed = float(claim["claimed_s"])
        cost = implied_overhead_s(claimed, n, failure_rate_per_gpu_hour)
        predicted = young_daly_interval_s(cost, FailureModel(n, failure_rate_per_gpu_hour).mtbf_seconds())
        implied.append(cost)
        rows.append({
            "n_gpus": n,
            "claimed": claim.get("claimed", ""),
            "claimed_s": claimed,
            "implied_overhead_s": cost,
            "predicted_s_at_implied_cost": predicted,
        })
    finite = [c for c in implied if c > 0]
    spread = (max(finite) / min(finite)) if finite else 1.0

    # Cross-check: pick the cost implied by the smallest cluster and see how
    # far the other rows are from the resulting sqrt curve.
    base = implied[0] if implied else 0.0
    consistent = True
    for row in rows:
        predicted = young_daly_interval_s(
            base, FailureModel(row["n_gpus"], failure_rate_per_gpu_hour).mtbf_seconds())
        row["predicted_s_from_base_cost"] = predicted
        row["ratio_claimed_over_predicted"] = (row["claimed_s"] / predicted
                                               if predicted not in (0.0, float("inf")) else None)
        if row["ratio_claimed_over_predicted"] is not None and not (
                1 / 1.25 <= row["ratio_claimed_over_predicted"] <= 1.25):
            consistent = False
    return {
        "schema": CHECKPOINT_SCHEMA,
        "failure_rate_per_gpu_hour": failure_rate_per_gpu_hour,
        "base_overhead_s": base,
        "implied_overhead_spread": spread,
        "single_cost_consistent": spread <= 1.25,
        "sqrt_scaling_consistent": consistent,
        "verdict": ("CONSISTENT" if consistent else
                    "INCONSISTENT-WITH-YOUNG/DALY-SQRT-SCALING"),
        "rows": rows,
    }


__all__ = [
    "CHECKPOINT_SCHEMA", "DEFAULT_FAILURE_RATE_PER_GPU_HOUR", "HOURS_PER_MONTH",
    "OVERHEAD_ALERT_FRACTION", "GRACE_PERIOD_S", "Environment", "ENVIRONMENTS",
    "FailureModel", "CheckpointCostModel", "young_daly_interval_s",
    "expected_waste_fraction", "implied_overhead_s", "monthly_waste_usd",
    "checkpoint_plan", "CHECKPOINT_FREQUENCY_CLAIMS", "check_interval_claims",
]
