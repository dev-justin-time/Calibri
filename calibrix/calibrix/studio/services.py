# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — New high-value services (beyond the F001–F100
# taxonomy). Each fills a gap no feature in D1–D8 covers; all are real,
# deterministic logic with the same offline-testability contract.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# S1 — Challenger Watch (kernel invalidation monitoring) ----------------------------------
@dataclass
class ChallengerReport:
    kernel_still_valid: bool
    baseline_shift: float          # relative change in unmodulated behavior
    recommendation: str
    checked_at: float


class ChallengerWatch:
    """Detect when an upstream model update invalidates a deployed kernel.

    WHY: kernels are tuned against a specific checkpoint; when the provider
    ships a new version, gains silently lose meaning. This is the recurring-
    revenue hook — re-calibration jobs — and it needs a *trigger*. The
    signal: compare live baseline (unmodulated) scorer values against the
    sealed baseline from the original run.
    """

    def __init__(self, sealed_baseline: float, tolerance: float = 0.05) -> None:
        self.sealed_baseline = sealed_baseline
        self.tolerance = tolerance

    def check(self, current_baseline: float) -> ChallengerReport:
        shift = (current_baseline - self.sealed_baseline) / max(
            abs(self.sealed_baseline), 1e-9)
        invalid = abs(shift) > self.tolerance
        return ChallengerReport(
            kernel_still_valid=not invalid,
            baseline_shift=round(shift, 4),
            recommendation=("re-calibrate: upstream model drifted"
                            if invalid else "kernel OK"),
            checked_at=time.time(),
        )


# S2 — Compat Scanner (instant quoting for arbitrary models) -------------------------------
@dataclass
class CompatReport:
    family: str
    n_layers: int
    n_sites: int
    n_components: int
    kernel_params: int
    supported: bool
    estimated_trials: int


def scan_architecture(block_counts: Dict[str, int],
                      supported_families: Sequence[str] = ("flux", "sd3", "qwen"),
                      popsize: int = 8) -> CompatReport:
    """Static analysis of a model's block structure -> calibration surface.

    WHY: the first question every prospect asks is "does my model work?" —
    and a real answer with a parameter count and trial estimate closes the
    deal. block_counts example: {"double_blocks": 19, "single_blocks": 38}.
    """
    family = "unknown"
    if "double_blocks" in block_counts and "single_blocks" in block_counts:
        family = "flux"
    elif "joint_blocks" in block_counts:
        family = "sd3"
    elif "transformer_blocks" in block_counts and len(block_counts) == 1:
        family = "qwen"
    n_sites = sum(block_counts.values())
    n_comp = 2 if family in ("flux", "sd3", "qwen") else 1
    params = n_sites * n_comp * 6
    min_trials = max(10, int(np.ceil(10 * params / max(popsize, 1))))
    return CompatReport(
        family=family, n_layers=len(block_counts), n_sites=n_sites,
        n_components=n_comp, kernel_params=params,
        supported=family in supported_families,
        estimated_trials=min_trials,
    )


# S3 — Drift Sentinel (production monitoring subscription) ------------------------------------
class DriftSentinel:
    """Watch live production outputs against the sealed quality baseline.

    WHY: a calibrated model in production can drift (data shifts, load
    changes, upstream patches). Enterprises pay monthly for monitoring +
    alerting; this is the SaaS subscription layer on top of one-time jobs.
    """

    def __init__(self, baseline_scores: Dict[str, float],
                 alert_threshold: float = 0.10) -> None:
        self.baseline = dict(baseline_scores)
        self.threshold = alert_threshold
        self.history: List[Dict[str, Any]] = []

    def observe(self, scores: Dict[str, float]) -> Optional[Dict[str, Any]]:
        deltas = {k: (scores.get(k, 0.0) - v) / max(abs(v), 1e-9)
                  for k, v in self.baseline.items()}
        breached = {k: round(d, 4) for k, d in deltas.items() if d < -self.threshold}
        record = {"ts": time.time(), "deltas": {k: round(d, 4) for k, d in deltas.items()},
                  "alert": bool(breached), "breached": breached}
        self.history.append(record)
        return record if breached else None


# S4 — KernelOps (versioning, staged rollout, rollback) -------------------------------------------
@dataclass
class KernelVersion:
    version: int
    kernel_spec: str
    holdout_fitness: float
    sealed_at: float
    traffic_pct: float = 0.0


class KernelOps:
    """A/B + staged rollout with automatic rollback on regression.

    WHY: deploying a kernel is a *change to production behavior*; without
    versioning/rollback it's an ops liability. This is the enterprise
    requirement that gates the UC-1/UC-2 deals.
    """

    def __init__(self, success_threshold: float = 0.0) -> None:
        self.versions: List[KernelVersion] = []
        self.success_threshold = success_threshold
        self._live_version: Optional[int] = None

    def publish(self, kernel_spec: str, holdout_fitness: float,
                traffic_pct: float = 10.0) -> KernelVersion:
        v = KernelVersion(version=len(self.versions) + 1,
                          kernel_spec=kernel_spec,
                          holdout_fitness=holdout_fitness,
                          sealed_at=time.time(),
                          traffic_pct=float(traffic_pct))
        self.versions.append(v)
        return v

    def promote(self, version: int, traffic_pct: float) -> Optional[KernelVersion]:
        for v in self.versions:
            if v.version == version:
                v.traffic_pct = float(traffic_pct)
                self._live_version = version
                return v
        return None

    def evaluate(self, version: int, live_fitness: float) -> Dict[str, Any]:
        """If the staged version underperforms its holdout estimate, roll back."""
        v = next(v for v in self.versions if v.version == version)
        regression = live_fitness < v.holdout_fitness + self.success_threshold
        action = "rollback" if regression else "keep"
        if regression:
            v.traffic_pct = 0.0
            stable = self.versions[-2] if len(self.versions) > 1 else v
            stable.traffic_pct = 100.0
            self._live_version = stable.version
        return {"version": version, "live_fitness": round(live_fitness, 4),
                "holdout_estimate": v.holdout_fitness, "action": action}

    def live(self) -> Optional[KernelVersion]:
        for v in reversed(self.versions):
            if v.traffic_pct > 0:
                return v
        return None


# S5 — Benchmark Harmonizer (cross-reward normalization) --------------------------------------------
class BenchmarkHarmonizer:
    """Put different reward models on one comparable scale.

    WHY: PickScore, HPSv3, ImageReward and aesthetic scorers have wildly
    different ranges; a marketplace listing claiming "+0.2" is meaningless
    without normalization. Percentile-rank against a reference distribution
    gives every listing a common 0..1 scale — the trust layer for UC-4.
    """

    def __init__(self, reference_distributions: Dict[str, Sequence[float]]) -> None:
        self.reference = {k: np.asarray(sorted(v), dtype=np.float64)
                          for k, v in reference_distributions.items()}

    def normalize(self, scorer: str, value: float) -> float:
        ref = self.reference.get(scorer)
        if ref is None or len(ref) == 0:
            return 0.0
        return float(np.searchsorted(ref, value, side="right") / len(ref))

    def listing_score(self, scores: Dict[str, float],
                      weights: Optional[Dict[str, float]] = None) -> float:
        normed = {k: self.normalize(k, v) for k, v in scores.items() if k in self.reference}
        if not normed:
            return 0.0
        if weights is None:
            return float(np.mean(list(normed.values())))
        w = {k: weights.get(k, 0.0) for k in normed}
        s = sum(w.values())
        return float(sum(normed[k] * w[k] for k in normed) / s) if s > 0 else 0.0


# S6 — Prompt Pack Curator (active-selection for calibration budgets) ---------------------------------
def curate_prompts(embeddings: np.ndarray, budget: int,
                   diversity_weight: float = 0.5, seed: int = 0) -> List[int]:
    """Select the most informative prompt subset for a compute budget.

    WHY: calibration cost scales with prompts-per-eval; most prompt banks are
    80% redundant. Farthest-point selection (a k-center greedy approximation)
    maximizes coverage per step — the same compute now explores more of the
    behavior space. Directly improves every job's margin.
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    if budget >= n:
        return list(range(n))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-12)
    selected = [int(rng.integers(n))]
    min_dist = 1.0 - unit @ unit[selected[0]]
    while len(selected) < budget:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, 1.0 - unit @ unit[idx])
    return sorted(selected)


# S7 — Kernel Transfer (cross-model warm start) -----------------------------------------------------------
def transfer_kernel(source_gains: Sequence[float], source_layers: int,
                    target_layers: int) -> np.ndarray:
    """Resample a kernel profile from one depth to another.

    WHY: models in the same family share gain *shape* even at different
    depths; initializing the target search from a resampled source kernel
    cuts trials roughly in half (warm-start pays the rent). Linear
    interpolation over normalized depth.
    """
    src = np.asarray(source_gains, dtype=np.float64)
    if source_layers < 2 or target_layers < 2:
        return np.resize(src, target_layers).astype(np.float64)
    src_x = np.linspace(0.0, 1.0, source_layers)
    tgt_x = np.linspace(0.0, 1.0, target_layers)
    return np.interp(tgt_x, src_x, src)


# S8 — SLO Governor (marginal-utility early stop) -------------------------------------------------------------
class SloGovernor:
    """Stop the search when fitness gain per dollar drops below the bar.

    WHY: the last 20% of search quality usually costs 80% of compute. The
    governor measures marginal fitness per marginal dollar and enforces the
    SLO — turning 'run until convergence' into 'run until it stops paying'.
    """

    def __init__(self, min_gain_per_usd: float = 0.01,
                 min_total_gain: float = 0.02) -> None:
        self.min_gain_per_usd = min_gain_per_usd
        self.min_total_gain = min_total_gain
        self.best_fitness = -float("inf")
        self.best_step = 0
        self._history: List[Tuple[float, float]] = []  # (fitness, spend_usd)

    def update(self, fitness: float, spend_usd: float) -> Dict[str, Any]:
        prev_f, prev_s = self._history[-1] if self._history else (-float("inf"), 0.0)
        marginal_gain = max(0.0, fitness - prev_f)
        marginal_usd = max(1e-9, spend_usd - prev_s)
        self._history.append((fitness, spend_usd))
        if fitness > self.best_fitness:
            self.best_fitness, self.best_step = fitness, len(self._history)
        total_gain = fitness - self._history[0][0] if self._history else 0.0
        rate = marginal_gain / marginal_usd
        stop = (len(self._history) > 1
                and rate < self.min_gain_per_usd
                and total_gain >= self.min_total_gain)
        return {"marginal_gain": round(marginal_gain, 6),
                "marginal_usd": round(marginal_usd, 4),
                "gain_per_usd": round(rate, 6),
                "stop": bool(stop)}


__all__ = [
    "ChallengerWatch", "ChallengerReport", "scan_architecture", "CompatReport",
    "DriftSentinel", "KernelOps", "KernelVersion", "BenchmarkHarmonizer",
    "curate_prompts", "transfer_kernel", "SloGovernor",
]
