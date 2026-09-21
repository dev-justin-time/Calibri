# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Domain 3: Global Optimizers & Pareto Solvers.
#
# Clean-room implementations from the primary literature:
#   * Deb et al. 2002, NSGA-II (non-dominated sorting + crowding distance)
#   * Storn & Price 1997, Differential Evolution (DE/rand/1/bin)
#   * Jamieson & Talwalkar 2016, Successive Halving (Hyperband's core)
#   * Srinivas et al. 2010, GP-UCB (with a tiny Gaussian-process regressor)
#   * Deb's knee-point detection on a Pareto front
#
# All operate on plain numpy arrays; the SearchEngine composes them via the
# existing Optimizer interface (optimizers.py).

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# F031 — NSGA-II non-dominated sorting -----------------------------------------
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a dominates b (all objectives <=, at least one <) — minimization."""
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_fronts(objectives: np.ndarray) -> List[List[int]]:
    """Non-dominated sorting. Returns list of fronts (index lists)."""
    n = len(objectives)
    dom_counts = np.zeros(n, dtype=int)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
                dom_counts[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_by[j].append(i)
                dom_counts[i] += 1
    fronts: List[List[int]] = []
    current = [i for i in range(n) if dom_counts[i] == 0]
    while current:
        fronts.append(current)
        nxt: List[int] = []
        for i in current:
            for j in dominated_by[i]:
                dom_counts[j] -= 1
                if dom_counts[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(front: Sequence[int], objectives: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for one front."""
    m = len(front)
    dist = np.zeros(m)
    if m == 0:
        return dist
    obj = objectives[list(front)]
    for k in range(obj.shape[1]):
        order = np.argsort(obj[:, k])
        dist[order[0]] = dist[order[-1]] = math.inf
        lo, hi = obj[order[0], k], obj[order[-1], k]
        if hi - lo < 1e-12:
            continue
        for r in range(1, m - 1):
            dist[order[r]] += (obj[order[r + 1], k] - obj[order[r - 1], k]) / (hi - lo)
    return dist


def nsga2_select(objectives: np.ndarray, k: int) -> List[int]:
    """Select k indices: fill fronts in order, break ties by crowding."""
    chosen: List[int] = []
    for front in pareto_fronts(objectives):
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            cd = crowding_distance(front, objectives)
            order = np.argsort(-cd)
            chosen.extend(front[i] for i in order[: k - len(chosen)])
            break
    return chosen


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (the first front)."""
    mask = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        dominated = False
        for j in range(len(objectives)):
            if i != j and dominates(objectives[j], objectives[i]):
                dominated = True
                break
        mask[i] = not dominated
    return mask


# F039 — Pareto Knee-Point Selector ----------------------------------------------
def knee_point(front_objectives: np.ndarray) -> int:
    """Index (into front_objectives) of the knee: the non-dominated point
    maximizing distance from the ideal point in the achievement-scalarizing
    sense (max of normalized objectives is minimized at the knee)."""
    pts = np.asarray(front_objectives, dtype=np.float64)
    if len(pts) < 3:
        return int(np.argmin(pts.sum(axis=1)))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    scaled = (pts - lo) / span  # in [0,1]^m per axis
    # knee = min over points of the MAX normalized objective (the classic
    # achievement scalarizing value); ties broken by sum.
    achievement = scaled.max(axis=1) * 1000 + scaled.sum(axis=1) * 1e-3
    return int(np.argmin(achievement))


# F029 — CMA-ES is in calibrix/optimizers.py; here: the missing solvers ----------
# F032 — Differential Evolution (Storn & Price 1997, DE/rand/1/bin) --------------
class DifferentialEvolution:
    """DE/rand/1/bin (Storn & Price 1997). Minimizes f."""

    def __init__(self, func, dim: int, lo: float = -2.0, hi: float = 2.0,
                 popsize: int = 24, f: float = 0.8, cr: float = 0.9,
                 seed: int = 0) -> None:
        self.func, self.dim, self.lo, self.hi = func, dim, lo, hi
        self.popsize, self.f, self.cr = popsize, f, cr
        self.rng = np.random.default_rng(seed)
        self.pop = self.rng.uniform(lo, hi, size=(popsize, dim))
        self.fit = np.array([func(x) for x in self.pop])

    def step(self) -> Tuple[float, np.ndarray]:
        for i in range(self.popsize):
            a, b, c = self.rng.choice(
                [j for j in range(self.popsize) if j != i], 3, replace=False)
            mut = self.pop[a] + self.f * (self.pop[b] - self.pop[c])
            mut = np.clip(mut, self.lo, self.hi)
            cross = self.rng.random(self.dim) < self.cr
            cross[self.rng.integers(self.dim)] = True
            trial = np.where(cross, mut, self.pop[i])
            tf = self.func(trial)
            if tf <= self.fit[i]:
                self.pop[i], self.fit[i] = trial, tf
        best = int(np.argmin(self.fit))
        return float(self.fit[best]), self.pop[best].copy()


# F033 — Successive Halving / Hyperband pruner -----------------------------------
@dataclass
class _SHTrial:
    x: np.ndarray
    budget: int
    history: List[float] = field(default_factory=list)


class SuccessiveHalving:
    """Allocate more budget only to the top 1/eta of performers (Jamieson &
    Talwalkar 2016). Cuts wasted evaluations on clearly-bad candidates —
    directly reduces D5-metered cost of a search."""

    def __init__(self, eta: int = 3, min_budget: int = 1, max_budget: int = 81) -> None:
        self.eta, self.min_b, self.max_b = eta, min_budget, max_budget
        s = int(math.floor(math.log(max_budget / min_budget, eta)))
        self.rungs = [min_budget * eta ** i for i in range(s + 1)]

    def run(self, evaluate, dim: int, n_candidates: int,
            seed: int = 0) -> Tuple[float, np.ndarray]:
        rng = np.random.default_rng(seed)
        trials = [_SHTrial(x=rng.normal(size=dim), budget=self.min_b)
                  for _ in range(n_candidates)]
        for rung_budget in self.rungs:
            scored = []
            for t in trials:
                # evaluate with the rung's budget (adapter-specific meaning)
                val = evaluate(t.x, rung_budget)
                t.history.append(val)
                scored.append((val, t))
            scored.sort(key=lambda p: p[0])
            keep = max(1, len(scored) // self.eta)
            trials = [t for _, t in scored[:keep]]
            if len(trials) == 1 and rung_budget == self.rungs[-1]:
                break
        best = min(trials, key=lambda t: t.history[-1])
        return float(best.history[-1]), best.x


# F033b — Hyperband -----------------------------------------------------------------
class Hyperband:
    """Multi-bracket budget allocator over SuccessiveHalving (Li et al. 2016,
    "Hyperband: A novel bandit approach for hyperparameter optimization").

    Runs several SHA brackets with different (n_candidates, initial_budget)
    tradeoffs — bracket 0 bets on many cheap candidates, the last bracket on
    a few candidates started at high budget — and returns the best point any
    bracket found. ``evaluate(x, budget)`` has the same adapter-defined
    contract as SuccessiveHalving. ``stats()`` reports total evaluations and
    budget consumed so a D5 BudgetGuard can be charged from actuals.
    """

    def __init__(self, eta: int = 3, min_budget: int = 1, max_budget: int = 81) -> None:
        if eta < 2:
            raise ValueError("eta must be >= 2")
        self.eta, self.min_b, self.max_b = eta, min_budget, max_budget
        self.smax = int(math.floor(math.log(max_budget / min_budget, eta)))
        self.evals = 0
        self.budget_spent = 0

    def stats(self) -> Dict[str, int]:
        """Actual evaluation count and budget consumed across all brackets."""
        return {"evaluations": self.evals, "budget_spent": self.budget_spent}

    def run(self, evaluate, dim: int, seed: int = 0) -> Tuple[float, np.ndarray]:
        rng = np.random.default_rng(seed)
        best_val, best_x = math.inf, None
        for s in range(self.smax, -1, -1):
            n = int(math.ceil((self.smax + 1) / (s + 1)) * self.eta ** s)
            budget = int(self.min_b * self.eta ** s)
            trials = [_SHTrial(x=rng.normal(size=dim), budget=budget)
                      for _ in range(n)]
            while True:
                for t in trials:
                    val = evaluate(t.x, budget)
                    t.history.append(val)
                    self.evals += 1
                    self.budget_spent += budget
                    if val < best_val:
                        best_val, best_x = val, t.x
                if len(trials) <= 1 or budget >= self.max_b:
                    break
                keep = max(1, len(trials) // self.eta)
                trials.sort(key=lambda t: t.history[-1])
                trials = trials[:keep]
                budget = min(budget * self.eta, self.max_b)
        return float(best_val), best_x


# F034 — Tiny GP-UCB ---------------------------------------------------------------
class GPUCB:
    """Gaussian-process upper-confidence-bound search (Srinivas 2010) with a
    compact RBF GP implemented directly (no sklearn): O(n^2) fits are fine
    for the 10^2-10^3 evaluation counts of kernel searches."""

    def __init__(self, dim: int, bounds: Tuple[float, float] = (-2.0, 2.0),
                 lengthscale: float = 0.5, noise: float = 1e-3,
                 beta: float = 2.0, seed: int = 0) -> None:
        self.dim, self.lo, self.hi = dim, bounds[0], bounds[1]
        self.ls, self.noise, self.beta = lengthscale, noise, beta
        self.rng = np.random.default_rng(seed)
        self.xs: List[np.ndarray] = []
        self.ys: List[float] = []

    def _k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * self.ls ** 2))

    def observe(self, x: Sequence[float], y: float) -> None:
        self.xs.append(np.asarray(x, dtype=np.float64))
        self.ys.append(float(y))

    def _posterior(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xtr = np.stack(self.xs)
        ytr = np.array(self.ys)
        ktt = self._k(xtr, xtr)
        kgg = self._k(grid, grid)
        kgt = self._k(grid, xtr)
        kinv = np.linalg.inv(ktt + self.noise * np.eye(len(xtr)))
        mu = kgt @ kinv @ ytr
        var = np.clip(np.diag(kgg - kgt @ kinv @ kgt.T), 0.0, None)
        return mu, np.sqrt(var)

    def ask(self, n_grid: int = 512) -> np.ndarray:
        grid = self.rng.uniform(self.lo, self.hi, size=(n_grid, self.dim))
        if len(self.xs) < 2:
            return grid[0]
        mu, sd = self._posterior(grid)
        return grid[int(np.argmax(mu + self.beta * sd))]

    def best(self) -> Tuple[float, np.ndarray]:
        i = int(np.argmax(self.ys))
        return self.ys[i], self.xs[i]


# F035 — Random Subspace Projector --------------------------------------------------
def random_subspace(dim: int, n_active: int, seed: int = 0) -> np.ndarray:
    """Indices of the active exploration subspace (Johnson–Lindenstrauss
    style: search a small random projection first, expand if the front is
    still moving)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(dim, size=min(n_active, dim), replace=False)
    return np.sort(idx)


# F036 — Step-size adaptive decayer (1/5th rule, classic ES) ------------------------
def adapt_sigma(sigma: float, success_ratio: float,
                target: float = 0.2, factor: float = 1.5) -> float:
    """Increase sigma when success rate beats target, shrink when below."""
    if success_ratio > target:
        return sigma * factor
    if success_ratio < target:
        return sigma / factor
    return sigma


# F038 — Exploration Entropy Booster -------------------------------------------------
def entropy_inject(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian restart noise — escape valve for premature convergence."""
    return x + rng.normal(0.0, sigma, size=x.shape)


# F037 — Population Island Migrator ---------------------------------------------------
class Islands:
    """Multi-deme ES with ring migration (Cantú-Paz 1998). Each island is an
    independent population; every `interval` steps the top `migrants` copy to
    the next island. Maps naturally onto multi-GPU shards."""

    def __init__(self, n_islands: int, dim: int, pop_per_island: int,
                 lo: float = -2.0, hi: float = 2.0, seed: int = 0,
                 migrants: int = 2, interval: int = 5) -> None:
        rng = np.random.default_rng(seed)
        self.pops = [rng.uniform(lo, hi, (pop_per_island, dim))
                     for _ in range(n_islands)]
        self.fitness = [np.full(pop_per_island, math.inf) for _ in range(n_islands)]
        self.migrants, self.interval = migrants, interval

    def migrate(self, step: int, fitnesses: List[np.ndarray]) -> None:
        if step % self.interval or len(self.pops) < 2:
            return
        n = len(self.pops)
        for i in range(n):
            src, dst = i, (i + 1) % n
            top = np.argsort(fitnesses[src])[: self.migrants]
            worst = np.argsort(fitnesses[dst])[-self.migrants:]
            self.pops[dst][worst] = self.pops[src][top]

    def best(self) -> Tuple[float, np.ndarray]:
        flat_f = np.concatenate(self.fitness)
        flat_p = np.concatenate(self.pops)
        i = int(np.argmin(flat_f))
        return float(flat_f[i]), flat_p[i]


# F040 — Warm-Start Run Clone Importer -------------------------------------------------
def warm_start_from_prior(prior_vector: Sequence[float], sigma: float,
                          n: int, rng: np.random.Generator) -> np.ndarray:
    """Seed a population around a previously-verified kernel vector."""
    prior = np.asarray(prior_vector, dtype=np.float64)
    return prior[None, :] + rng.normal(0.0, sigma, size=(n, len(prior)))


__all__ = [
    "dominates", "pareto_fronts", "crowding_distance", "nsga2_select",
    "pareto_mask", "knee_point", "DifferentialEvolution", "SuccessiveHalving",
    "Hyperband", "GPUCB", "random_subspace", "adapt_sigma", "entropy_inject",
    "Islands", "warm_start_from_prior",
]
