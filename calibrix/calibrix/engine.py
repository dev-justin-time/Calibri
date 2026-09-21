# Calibrix search engine.
#
# Combines Heretic's co-objective pattern (behavior scorer vs. drift scorer,
# Pareto trial menu) with Calibri's CMA-ES over gains and adds what both
# lack: a train/holdout split with an overfitting alarm, compute metering
# per trial, and lossless export of everything needed to reproduce a run.

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .adapters import Adapter
from .events import EventLog
from .kernel import KernelParams, ModulationSpec, pack_spec_vector, total_param_count
from .metering import Meter, CostModel
from .optimizers import make_optimizer
from .scorers import Prompt, Scorer, ScorerContext, Score


@dataclass
class SearchConfig:
    n_trials: int = 12                  # optimizer generations
    popsize: int = 8                    # candidates per generation
    optimizer: str = "simple"           # "simple" | "cma" | "tpe"
    seed: int = 0
    batch_size: int = 8
    # scalarization: fitness = sum(weight_i * dir_i * value_i) over scorers
    # whose .optimization is set. Report-only scorers are logged only.
    objective_weights: Dict[str, float] = field(default_factory=dict)
    holdout_fraction: float = 0.25
    overfit_margin: float = 0.15        # relative train/holdout gap that flags overfitting
    price_per_1k_steps: float = 0.0
    log_path: Optional[str] = None      # JSONL event log
    meter_path: Optional[str] = None    # cost ledger JSON


@dataclass
class TrialResult:
    trial: int
    generation: int
    vector: List[float]
    objectives: Dict[str, float]        # train values by scorer name
    holdout: Dict[str, float]           # holdout values (filled after search)
    fitness: float
    steps_used: int
    seconds: float


class SearchEngine:
    """Co-optimizes modulation kernels against a panel of scorers."""

    def __init__(
        self,
        adapter: Adapter,
        scorers: List[Scorer],
        train_prompts: List[Prompt],
        config: Optional[SearchConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.all_scorers = list(scorers)
        self.cfg = config or SearchConfig()
        self.events = EventLog(self.cfg.log_path)
        self.meter = Meter(CostModel(price_per_1k_steps=self.cfg.price_per_1k_steps))

        # --- split prompts into train / holdout ---------------------------
        n_hold = max(1, int(len(train_prompts) * self.cfg.holdout_fraction))
        n_hold = min(n_hold, max(len(train_prompts) - 1, 1))
        cut = len(train_prompts) - n_hold
        self.train_prompts = list(train_prompts[:cut]) or list(train_prompts)
        self.holdout_prompts = list(train_prompts[cut:]) or list(train_prompts)

        # --- per-split scorer clones (frozen baselines live here) ---------
        self.train_scorers = [copy.deepcopy(s) for s in self.all_scorers]
        self.holdout_scorers = [copy.deepcopy(s) for s in self.all_scorers]

    # ------------------------------------------------------------------
    def _objectives(self) -> List[Tuple[str, float, float]]:
        """(name, weight, direction) for scorers that are objectives."""
        out = []
        for s in self.all_scorers:
            if s.optimization in ("maximize", "minimize"):
                w = self.cfg.objective_weights.get(s.score_name, s.weight)
                d = 1.0 if s.optimization == "maximize" else -1.0
                out.append((s.score_name, float(w), d))
        return out

    # ------------------------------------------------------------------
    def _evaluate_with(
        self, prompts: List[Prompt], scorers: List[Scorer]
    ) -> Tuple[Dict[str, Score], int]:
        ctx = ScorerContext(self.adapter, batch_size=self.cfg.batch_size)
        for s in scorers:
            s.prompts_spec = prompts  # bind this split's prompts
            s.init(ctx)               # caching scorers freeze baselines here
        scores: Dict[str, Score] = {}
        for s in scorers:
            scores[s.score_name] = s.get_score(ctx)
        return scores, ctx.steps_used

    def _identity_specs(self) -> List[ModulationSpec]:
        specs = self.adapter.spec_sites()
        identity = []
        for s in specs:
            s2 = copy.deepcopy(s)
            s2.kernel = KernelParams(weight=1.0, position=s.kernel.position,
                                     focus=1e6, floor=1.0, ripple=0.0, phase=0.0)
            s2.explicit = None
            identity.append(s2)
        return identity

    def _baseline(self) -> Dict[str, Dict[str, Score]]:
        """Baseline = identity modulation. Also freezes scorer baselines
        (KL reference logits, length/diversity references) at the original
        model state, mirroring Heretic's baseline protocol."""
        self.adapter.apply_modulation(self._identity_specs())
        train_scores, _ = self._evaluate_with(self.train_prompts, self.train_scorers)
        hold_scores, _ = self._evaluate_with(self.holdout_prompts, self.holdout_scorers)
        return {"train": train_scores, "holdout": hold_scores}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        t_start = time.perf_counter()
        self.events.log(
            "run_start",
            adapter=type(self.adapter).__name__,
            scorers=[s.score_name for s in self.all_scorers],
            n_train=len(self.train_prompts),
            n_holdout=len(self.holdout_prompts),
            n_trials=cfg.n_trials,
            popsize=cfg.popsize,
            optimizer=cfg.optimizer,
        )

        baseline = self._baseline()
        self.meter.mark(0, "baseline", 0, 0.0, len(self.train_scorers) * 2)
        self.events.log(
            "baseline",
            train={k: v.value for k, v in baseline["train"].items()},
            holdout={k: v.value for k, v in baseline["holdout"].items()},
        )

        specs = self.adapter.spec_sites()
        x0 = pack_spec_vector(specs)
        n_params = total_param_count(specs)
        self.events.log("search_space", n_params=n_params)

        objectives = self._objectives()
        if not objectives:
            raise ValueError(
                "No optimization objectives configured. At least one scorer must "
                "set optimization='maximize' or 'minimize'."
            )

        opt = make_optimizer(cfg.optimizer, x0, seed=cfg.seed, popsize=cfg.popsize)

        trials: List[TrialResult] = []
        trial_counter = 0

        for gen in range(cfg.n_trials):
            pop = opt.ask()
            fitnesses: List[float] = []

            for vec in pop:
                trial_counter += 1
                t0 = time.perf_counter()

                self.adapter.apply_to_vector(vec)
                train_scores, steps_used = self._evaluate_with(
                    self.train_prompts, self.train_scorers
                )

                fit = 0.0
                for name, weight, direction in objectives:
                    sc = train_scores.get(name)
                    if sc is not None:
                        fit += weight * direction * sc.value

                seconds = time.perf_counter() - t0
                self.meter.mark(trial_counter, "search", steps_used, seconds,
                                scorer_calls=len(self.train_scorers))
                trials.append(TrialResult(
                    trial=trial_counter,
                    generation=gen,
                    vector=list(vec),
                    objectives={k: v.value for k, v in train_scores.items()},
                    holdout={},
                    fitness=fit,
                    steps_used=steps_used,
                    seconds=seconds,
                ))
                fitnesses.append(fit)
                self.events.log("trial", trial=trial_counter, generation=gen,
                                fitness=fit,
                                objectives=trials[-1].objectives)

            opt.tell(fitnesses)
            self.events.log("generation", generation=gen,
                            sigma=getattr(opt, "sigma", None),
                            best_fitness=max(fitnesses) if fitnesses else None)

        # ---- holdout evaluation of the Pareto set + overall best ----------
        pareto = self._pareto(trials)
        to_validate = {id(t): t for t in pareto[:6]}
        best = max(trials, key=lambda t: t.fitness)
        to_validate[id(best)] = best

        for t in to_validate.values():
            self.adapter.apply_to_vector(t.vector)
            hold_scores, steps_used = self._evaluate_with(
                self.holdout_prompts, self.holdout_scorers
            )
            t.holdout = {k: v.value for k, v in hold_scores.items()}
            self.meter.mark(t.trial, "validation", steps_used, 0.0,
                            len(self.holdout_scorers))
        self.events.log("holdout_evaluated", n=len(to_validate))

        alarm = self._overfit_alarm(list(to_validate.values()))
        self.events.log("overfit_alarm", **alarm)

        elapsed = time.perf_counter() - t_start
        self.meter.mark(0, "run", 0, elapsed, 0)
        if self.cfg.meter_path:
            self.meter.save(self.cfg.meter_path)

        result: Dict[str, Any] = {
            "baseline_train": {k: v.value for k, v in baseline["train"].items()},
            "baseline_holdout": {k: v.value for k, v in baseline["holdout"].items()},
            "n_params": n_params,
            "n_evals": trial_counter,
            "trials": trials,
            "pareto": pareto,
            "best": best,
            "overfit": alarm,
            "elapsed_seconds": elapsed,
        }
        self.events.log("run_end", best_fitness=best.fitness, elapsed=elapsed)
        return result

    # ------------------------------------------------------------------
    def _pareto(self, trials: List[TrialResult]) -> List[TrialResult]:
        """Non-dominated set over (maximize fitness, minimize steps)."""
        def dominates(a: TrialResult, b: TrialResult) -> bool:
            ge = (a.fitness >= b.fitness) and (a.steps_used <= b.steps_used)
            strict = (a.fitness > b.fitness) or (a.steps_used < b.steps_used)
            return ge and strict

        pareto = [
            a for a in trials
            if not any(dominates(b, a) for b in trials if b is not a)
        ]
        seen: set = set()
        out: List[TrialResult] = []
        for t in sorted(pareto, key=lambda t: -t.fitness):
            key = tuple(round(v, 9) for v in t.vector)
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _overfit_alarm(self, validated: List[TrialResult]) -> Dict[str, Any]:
        """Flags when the selected trial's holdout degraded vs. its train values."""
        if not validated:
            return {"flagged": False, "details": []}
        best = max(validated, key=lambda t: t.fitness)
        flagged = []
        for name, tr in best.objectives.items():
            ho = best.holdout.get(name)
            if ho is None:
                continue
            denom = max(abs(tr), 1e-6)
            gap = (tr - ho) / denom if tr >= 0 else (ho - tr) / denom
            if gap > self.cfg.overfit_margin:
                flagged.append({"metric": name, "train": round(tr, 4),
                                "holdout": round(ho, 4), "gap": round(gap, 4)})
        return {
            "flagged": bool(flagged),
            "details": flagged,
            "margin": self.cfg.overfit_margin,
            "note": "holdout gap exceeds margin: selection overfit the train "
                    "split; increase holdout_fraction or reduce n_trials",
        }
