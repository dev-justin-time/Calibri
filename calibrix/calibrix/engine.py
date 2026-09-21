# SPDX-License-Identifier: MIT
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
from .studio.billing import BudgetGuard, CostBreaker


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
    # --- budget guards (studio.billing wired into the run loop) -----------
    # max_budget_usd: hard ceiling on metered step spend. 0/None = unbounded
    #   (legacy behavior). When set, the engine preflights the requested
    #   trials, skips trials it cannot afford, and HALTS the search the
    #   moment spend would exceed the ceiling — results found so far are
    #   still validated and reported.
    max_budget_usd: Optional[float] = None
    # quoted_usd: when set (and > 0), a fail-closed CostBreaker trips once
    #   spend exceeds quote * (1 + tolerance) and further work is refused
    #   outright (not merely reduced).
    quoted_usd: Optional[float] = None
    breaker_tolerance: float = 0.10
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

        # --- financial guards (calibrix.studio.billing) --------------------
        # BudgetGuard: preflight + per-trial affordability; CostBreaker:
        # fail-closed killswitch. price_per_1k_steps == 0 means spend never
        # grows, so both guards are naturally inert unless a quote is set.
        self.budget = BudgetGuard(
            max_usd=self.cfg.max_budget_usd
            if self.cfg.max_budget_usd not in (None, 0) else float("inf"),
            price_per_1k_steps=self.cfg.price_per_1k_steps,
        )
        self.breaker = (
            CostBreaker(self.cfg.quoted_usd, tolerance=self.cfg.breaker_tolerance)
            if self.cfg.quoted_usd and self.cfg.quoted_usd > 0 else None
        )
        if self.breaker is not None and self.cfg.price_per_1k_steps <= 0:
            raise ValueError(
                "quoted_usd requires price_per_1k_steps > 0: without a price, "
                "spend is always 0 and the breaker can never trip."
            )

    # -- budget hooks ------------------------------------------------------
    def _spend(self) -> float:
        """USD spent on metered steps so far."""
        return self.budget.spent_usd

    def _charge(self, steps: int) -> None:
        self.budget.charge(steps)
        if self.breaker is not None and self.breaker.check(self._spend()):
            self.events.log(
                "budget_breaker_tripped",
                spent_usd=round(self._spend(), 6),
                limit_usd=round(self.breaker.limit, 6),
            )

    def _halted(self) -> bool:
        """True once the budget/breaker forbids any further paid work."""
        if self.breaker is not None and self.breaker.tripped:
            return True
        return self.budget.remaining_usd <= 0.0

    def _budget_summary(self, halted: bool, skipped: int) -> Dict[str, Any]:
        """Serializable financial outcome of the run (result + event log)."""
        bounded = self.budget.max_usd != float("inf")
        return {
            "halted": halted,
            "trials_skipped_budget": skipped,
            "spent_usd": round(self._spend(), 6),
            "max_budget_usd": round(self.budget.max_usd, 6) if bounded else None,
            "remaining_usd": round(self.budget.remaining_usd, 6) if bounded else None,
            "breaker_tripped": bool(self.breaker and self.breaker.tripped),
        }

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
            s.init(ctx)               # seal-once; baselines anchor at identity
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

        if self.budget.max_usd != float("inf"):
            # Preflight: a budget that cannot cover ONE evaluation should
            # fail before spending anything, not after.
            est_first = max(len(self.train_prompts), 1)
            if not self.budget.can_afford(est_first):
                raise ValueError(
                    f"max_budget_usd ({self.cfg.max_budget_usd}) cannot cover a "
                    f"single evaluation at price_per_1k_steps "
                    f"({self.cfg.price_per_1k_steps}); raise the budget, lower "
                    "the price, or reduce prompts per evaluation."
                )
            self.events.log(
                "budget_preflight",
                max_budget_usd=round(self.budget.max_usd, 6),
                price_per_1k_steps=round(self.budget.price, 6),
                estimated_eval_steps=est_first,
            )

        baseline = self._baseline()
        baseline_steps = 0  # baseline runs at identity; meter reports usage
        self.meter.mark(0, "baseline", baseline_steps, 0.0, len(self.train_scorers) * 2)
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
        halted = False
        skipped_trials = 0

        for gen in range(cfg.n_trials):
            if halted:
                break
            pop = opt.ask()
            fitnesses: List[float] = []
            evaluated_this_gen = 0

            for vec in pop:
                est = self._estimate_eval_steps()
                if not self.budget.can_afford(est) or self._halted():
                    skipped_trials += 1
                    continue
                trial_counter += 1
                t0 = time.perf_counter()

                self.adapter.apply_to_vector(vec)
                train_scores, steps_used = self._evaluate_with(
                    self.train_prompts, self.train_scorers
                )
                self._charge(steps_used)

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
                evaluated_this_gen += 1
                self.events.log("trial", trial=trial_counter, generation=gen,
                                fitness=fit,
                                objectives=trials[-1].objectives)
                if self._halted():
                    halted = True
                    self.events.log(
                        "budget_halt",
                        at_trial=trial_counter,
                        spent_usd=round(self._spend(), 6),
                        remaining_usd=round(self.budget.remaining_usd, 6),
                        breaker=self.breaker.tripped if self.breaker else False,
                    )
                    break

            # Optimizers expect one tell() per ask(); with an all-skipped
            # generation there is nothing to report, and CMA-ES rejects an
            # empty tell outright.
            if fitnesses:
                opt.tell(fitnesses)
            n_skipped = (len(pop) - evaluated_this_gen) if pop else 0
            self.events.log(
                "generation", generation=gen,
                sigma=getattr(opt, "sigma", None),
                best_fitness=max(fitnesses) if fitnesses else None,
                evaluated=evaluated_this_gen,
                skipped=n_skipped,
            )
            if pop and evaluated_this_gen == 0:
                # Could not afford a single evaluation this generation:
                # further generations cannot make progress either.
                halted = True
                self.events.log(
                    "budget_halt",
                    at_trial=trial_counter,
                    spent_usd=round(self._spend(), 6),
                    remaining_usd=round(self.budget.remaining_usd, 6),
                    breaker=self.breaker.tripped if self.breaker else False,
                    reason="insufficient budget for any further evaluation",
                )

        if not trials:
            # Budget exhausted before any trial completed: return a
            # well-formed, report-safe result instead of crashing.
            elapsed = time.perf_counter() - t_start
            summary = self._budget_summary(True, skipped_trials)
            self.meter.mark(0, "run", 0, elapsed, 0)
            if self.cfg.meter_path:
                self.meter.save(self.cfg.meter_path)
            self.events.log("budget_exhausted_no_trials", **summary)
            return {
                "baseline_train": {k: v.value for k, v in baseline["train"].items()},
                "baseline_holdout": {k: v.value for k, v in baseline["holdout"].items()},
                "n_params": n_params,
                "n_evals": 0,
                "trials": [],
                "pareto": [],
                "best": None,
                "overfit": {"flagged": False, "details": []},
                "budget": summary,
                "elapsed_seconds": elapsed,
            }

        # ---- holdout evaluation of the Pareto set + overall best ----------
        pareto = self._pareto(trials)
        to_validate = {id(t): t for t in pareto[:6]}
        best = max(trials, key=lambda t: t.fitness)
        to_validate[id(best)] = best

        validated, validation_skipped = 0, 0
        for t in to_validate.values():
            if self.budget.can_afford(self._estimate_eval_steps()) and not self._halted():
                self.adapter.apply_to_vector(t.vector)
                hold_scores, steps_used = self._evaluate_with(
                    self.holdout_prompts, self.holdout_scorers
                )
                self._charge(steps_used)
                t.holdout = {k: v.value for k, v in hold_scores.items()}
                self.meter.mark(t.trial, "validation", steps_used, 0.0,
                                len(self.holdout_scorers))
                validated += 1
            else:
                validation_skipped += 1
        self.events.log("holdout_evaluated", n=len(to_validate),
                        validated=validated, skipped=validation_skipped)

        alarm = self._overfit_alarm(list(to_validate.values()))
        self.events.log("overfit_alarm", **alarm)

        elapsed = time.perf_counter() - t_start
        self.meter.mark(0, "run", 0, elapsed, 0)
        if self.cfg.meter_path:
            self.meter.save(self.cfg.meter_path)

        budget_summary = self._budget_summary(halted, skipped_trials)
        self.events.log("budget_summary", **budget_summary)

        result: Dict[str, Any] = {
            "baseline_train": {k: v.value for k, v in baseline["train"].items()},
            "baseline_holdout": {k: v.value for k, v in baseline["holdout"].items()},
            "n_params": n_params,
            "n_evals": trial_counter,
            "trials": trials,
            "pareto": pareto,
            "best": best,
            "overfit": alarm,
            "budget": budget_summary,
            "elapsed_seconds": elapsed,
        }
        self.events.log("run_end", best_fitness=best.fitness, elapsed=elapsed)
        return result

    # ------------------------------------------------------------------
    def _estimate_eval_steps(self) -> int:
        """Steps a single evaluation of the train split is expected to cost.

        Derived from the last observed train evaluation (actual usage,
        which includes adapter-side per-call overheads); falls back to one
        step per prompt when nothing has been measured yet. Used for
        preflight affordability decisions, never for accounting — spend is
        always charged from reported actuals.
        """
        observed = [
            r.steps for r in self.meter.records
            if r.phase == "search" and r.steps > 0
        ]
        if observed:
            return int(observed[-1])
        return max(len(self.train_prompts), 1)

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
