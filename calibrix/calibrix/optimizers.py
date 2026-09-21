# Calibrix optimizers.
#
# Repurposed logic:
#   * Heretic uses Optuna's TPE over ~10 kernel params with co-objectives.
#   * Calibri uses CMA-ES over ~100 gate gains, single objective.
# Calibrix ships its own dependency-free (1+lambda) ES as the offline
# default, plus optional adapters to cma (CMA-ES, Calibri's engine) and
# optuna (TPE, Heretic's engine) when installed.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple

from .kernel import pack_spec_vector, unpack_spec_vector

Objective = Callable[[Sequence[float]], Tuple[float, ...]]
# objective returns one value per co-objective direction; direction signs
# (+1 maximize / -1 minimize) are handled by the engine wrapper.


class Optimizer(ABC):
    """Black-box optimizer over a fixed-length float vector."""

    @abstractmethod
    def ask(self) -> List[List[float]]:
        """Propose a population of candidate vectors."""

    @abstractmethod
    def tell(self, fitnesses: Sequence[float]) -> None:
        """Report scalarized fitness (higher = better) for the last ask()."""

    @property
    def best(self) -> List[float]:
        return self._best

    @property
    def sigma(self) -> float:
        return getattr(self, "_sigma", 0.0)


class SimpleES(Optimizer):
    """Dependency-free (1+lambda) evolution strategy with adaptive step.

    Robust default for offline/CI use: no compiled deps, deterministic
    given seed, behaves reasonably on 6-200 param kernels.
    """

    def __init__(
        self,
        x0: Sequence[float],
        sigma0: float = 0.15,
        lamb: int = 8,
        seed: int = 0,
        bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> None:
        import numpy as np

        self._np = np
        self.x = [float(v) for v in x0]
        self._best = list(self.x)
        self._best_f = -math.inf
        self._sigma = float(sigma0)
        self.lamb = max(2, int(lamb))
        self.rng = np.random.default_rng(seed)
        self.bounds = bounds

    def ask(self) -> List[List[float]]:
        pop = [list(self.x)]
        for _ in range(self.lamb - 1):
            cand = [
                v + self._sigma * float(self.rng.standard_normal())
                for v in self.x
            ]
            if self.bounds:
                cand = [
                    min(max(c, lo), hi) if (lo is not None and hi is not None) else c
                    for c, (lo, hi) in zip(cand, self.bounds)
                ]
            pop.append(cand)
        self._pop = pop
        return pop

    def tell(self, fitnesses: Sequence[float]) -> None:
        order = sorted(range(len(fitnesses)), key=lambda i: -fitnesses[i])
        best_i = order[0]
        if fitnesses[best_i] > self._best_f:
            self._best_f = float(fitnesses[best_i])
            self._best = list(self._pop[best_i])
            # success -> grow step slightly
            self._sigma *= 1.2
        else:
            # no improvement -> shrink toward baseline exploration
            self._sigma *= 0.85
        self._sigma = min(max(self._sigma, 1e-4), 2.0)
        # move mean toward best candidate (simple (1+lambda) update)
        best_x = self._pop[best_i]
        lr = 0.6
        self.x = [ (1 - lr) * v + lr * b for v, b in zip(self.x, best_x) ]


class CmaEsOptimizer(Optimizer):
    """CMA-ES via the `cma` package (Calibri's optimizer). Optional dep."""

    def __init__(
        self,
        x0: Sequence[float],
        sigma0: float = 0.15,
        popsize: Optional[int] = None,
        seed: int = 0,
        bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> None:
        import cma  # optional dependency (Calibri already uses it)

        lo = [b[0] for b in bounds if b is not None] if bounds else None
        hi = [b[1] for b in bounds if b is not None] if bounds else None
        opts: dict = {"seed": seed + 1, "verbose": -9}
        if popsize:
            opts["popsize"] = int(popsize)
        if lo and hi:
            opts["bounds"] = [lo, hi]
        self.es = cma.CMAEvolutionStrategy([float(v) for v in x0], sigma0, opts)
        self._best = list(x0)
        self._best_f = -math.inf

    def ask(self) -> List[List[float]]:
        self._pop = [list(map(float, x)) for x in self.es.ask()]
        return self._pop

    def tell(self, fitnesses: Sequence[float]) -> None:
        # cma minimizes; Calibrix fitness is "higher is better"
        self.es.tell(self._pop, [-f for f in fitnesses])
        b = self.es.best.__dict__["x"] if hasattr(self.es.best, "__dict__") else None
        if b is not None:
            self._best = [float(v) for v in b]


class TpeOptimizer(Optimizer):
    """Optuna TPE over the flat kernel vector (Heretic's optimizer). Optional dep."""

    def __init__(
        self,
        x0: Sequence[float],
        sigma0: float = 0.15,
        popsize: int = 8,
        seed: int = 0,
        bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> None:
        import optuna  # optional dependency (Heretic already uses it)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self.popsize = max(2, int(popsize))
        self._sigma = float(sigma0)
        self.rng_seed = seed
        self.bounds = bounds
        self.x = [float(v) for v in x0]
        self._best = list(self.x)
        self._best_f = -math.inf
        self._history: List[Tuple[List[float], float]] = []
        self._pending: List[List[float]] = []
        self._study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))

    def ask(self) -> List[List[float]]:
        import numpy as np

        self._pending = []
        lo_hi = [
            (b if b else (v - 5 * self._sigma, v + 5 * self._sigma))
            for v, b in zip(self.x, self.bounds or [None] * len(self.x))
        ]
        for _ in range(self.popsize):
            t = self._study.ask()
            cand = [
                float(t.suggest_float(f"p{i}", lo, hi))
                for i, (lo, hi) in enumerate(lo_hi)
            ]
            self._pending.append(cand)
        return self._pending

    def tell(self, fitnesses: Sequence[float]) -> None:
        for cand, f in zip(self._pending, fitnesses):
            t = self._study.ask()  # reuse flow: create trial then set values
            self._study.tell(t, [-f])  # TPE minimizes
            self._history.append((cand, float(f)))
            if f > self._best_f:
                self._best_f = float(f)
                self._best = list(cand)


def make_optimizer(
    name: str,
    x0: Sequence[float],
    bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    seed: int = 0,
    popsize: int = 8,
    sigma0: float = 0.15,
) -> Optimizer:
    """Factory. Falls back to SimpleES when optional deps are missing."""
    name = name.lower()
    if name in ("cma", "cmaes"):
        try:
            return CmaEsOptimizer(x0, sigma0=sigma0, popsize=popsize, seed=seed, bounds=bounds)
        except ImportError:
            return SimpleES(x0, sigma0=sigma0, lamb=popsize, seed=seed, bounds=bounds)
    if name in ("tpe", "optuna"):
        try:
            return TpeOptimizer(x0, sigma0=sigma0, popsize=popsize, seed=seed, bounds=bounds)
        except ImportError:
            return SimpleES(x0, sigma0=sigma0, lamb=popsize, seed=seed, bounds=bounds)
    return SimpleES(x0, sigma0=sigma0, lamb=popsize, seed=seed, bounds=bounds)
