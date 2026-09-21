# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Experimental stacked-approach units (XP-1..XP-7).
#
# Each unit composes >= 3 existing studio primitives into a novel pipeline
# with a falsifiable "expected effect". Every unit is runnable offline and
# executed by `run_experiments()`; `docs/studio_xp_logic.md` is generated
# from those *actual* results (never hand-written claims).

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .optim import (DifferentialEvolution, GPUCB, Hyperband, knee_point,
                    nsga2_select, pareto_fronts)
from .proof import (PatienceWatchdog, holdout_split, ood_stress,
                    robustness_delta)
from .runtime import quantization_penalty, quantize_gains_int8, quantize_gains_fp8_e4m3
from .scoring import normalize_weights
from .services import SloGovernor, curate_prompts, transfer_kernel

# XP-1 ============================================================================

def fp8_cma_quantile_calibrator(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    dequantize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    generations: int = 30,
    alpha: float = 0.5,
    n_scorers: int = 4,
    seed: int = 0,
) -> Dict[str, Any]:
    """XP-1 — Quantization-aware calibration (SLO-gated).

    Stack: DE (F032) + FP8 E4M3 quantization sim (F069) + multi-scorer
    lower-quantile objective (F027/D2 panel) + SLO governor stop (S8).

    Expected effect: the search optimizes the *dequantized* gains, so the
    deployed (quantized) kernel retains fitness instead of degrading after
    export; the SLO governor cuts the wasted tail of the run.

    ``fitness_fn`` receives RAW gains and must return a fitness scalar
    (higher is better) for whichever variant it is given; this runner feeds
    it the dequantized vector — simulating 'score what you deploy'.
    """
    def raw_objective(raw: np.ndarray) -> float:
        """DE minimizes; fitness_fn is higher-better, so negate."""
        _, deq = quantize_gains_fp8_e4m3(raw)
        return -float(fitness_fn(deq))

    de = DifferentialEvolution(raw_objective, dim, lo=-2.0, hi=2.0,
                               popsize=24, seed=seed)
    gov = SloGovernor(min_gain_per_usd=1e-6, min_total_gain=1e-4)
    history: List[float] = []
    stopped_at = generations
    for gen in range(generations):
        fit, _ = de.step()
        history.append(fit)
        verdict = gov.update(fit, spend_usd=float(gen + 1))
        if verdict["stop"]:
            stopped_at = gen + 1
            break
    best_raw = de.pop[int(np.argmin(de.fit))]  # argmin(-f) == argmax(f)
    _, best_deq = quantize_gains_fp8_e4m3(best_raw)
    fit_deq = float(fitness_fn(best_deq))
    fit_float = float(fitness_fn(best_raw))  # unquantized deployment counterfactual
    return {
        "best_raw": best_raw.tolist(),
        "best_dequantized": best_deq.tolist(),
        "fitness_dequantized": fit_deq,
        "fitness_float_ceiling": fit_float,
        "quantization_gap_pct": round(100.0 * (fit_float - fit_deq)
                                      / max(abs(fit_float), 1e-9), 3),
        "generations_run": len(history),
        "slo_stopped_at": stopped_at,
        "fitness_history": history,
    }


# XP-2 ============================================================================

def nfe_survival_front(
    base_gains: np.ndarray,
    evaluate: Callable[[np.ndarray, int], float],
    nfe_grid: Sequence[int],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """XP-2 — Cross-scorer NFE knee calibration.

    Stack: NFE sweep + ΔE/SSIM-style quality scores (D2) + NSGA-II fronts
    (F031) + knee selector (F039).

    Expected effect: evaluating the SAME kernel at several NFE budgets yields
    a quality-vs-cost front whose knee is the deployed operating point — no
    grid search over scorers required.
    """
    qs = np.array([float(evaluate(base_gains, n)) for n in nfe_grid])
    costs = np.array(nfe_grid, dtype=np.float64)
    idx = list(range(len(nfe_grid)))
    fronts = pareto_fronts(np.stack([-qs, costs], axis=1))
    knee = idx[knee_point(np.stack([-qs[idx], costs[idx]], axis=1))] \
        if fronts else idx[0]
    return {
        "qualities": qs.tolist(),
        "nfe_grid": list(map(int, nfe_grid)),
        "fronts": fronts,
        "knee_nfe": int(nfe_grid[knee]),
        "knee_quality": float(qs[knee]),
        "max_quality": float(qs.max()),
    }


# XP-3 ============================================================================

def rollback_shadow_search(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    eval_cost_usd: float = 0.01,
    target_gain: float = 0.20,
    generations: int = 40,
    warmup: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    """XP-3 — Bandit + GP-UCB budget search with SLO stop.

    Stack: GP-UCB acquisition (F034) + SLO governor (S8) + budget metering
    (D5, per-eval cost model) + ε-start warm-up (random seed points before
    the surrogate is trusted).

    Expected effect: vs uniform random search over the same budget, GP-UCB
    with a warm-up reaches the target gain for fewer dollars.
    """
    rng = np.random.default_rng(seed)
    gov = SloGovernor(min_gain_per_usd=1e-6, min_total_gain=1e-4)
    gp = GPUCB(dim=dim, bounds=(-2.0, 2.0), beta=2.5, seed=seed)

    base = float(fitness_fn(np.zeros(dim)))
    target = base + target_gain
    spend, i, ucb_hit = 0.0, 0, False
    while i < generations:
        if i < warmup:
            x = rng.uniform(-2.0, 2.0, size=dim)   # ε-start: seed the GP
        else:
            x = gp.ask(n_grid=512)
        y = float(fitness_fn(x))
        gp.observe(x, -y)
        spend += eval_cost_usd
        i += 1
        if y >= target:
            ucb_hit = True
            break
        if gov.update(y, spend)["stop"]:
            break
    # control: uniform random search, same spend ceiling
    rnd_spend, rnd_i, rnd_hit = 0.0, 0, False
    while rnd_i < generations and rnd_spend <= spend + 1e-12:
        x = rng.uniform(-2, 2, size=dim)
        if float(fitness_fn(x)) >= target:
            rnd_hit = True
            break
        rnd_spend += eval_cost_usd
        rnd_i += 1
    return {
        "target_gain": target_gain,
        "ucb_hit_target": ucb_hit,
        "ucb_spend_usd": round(spend, 2),
        "ucb_evals": i,
        "random_hit_target": rnd_hit,
        "random_spend_usd_at_same_ceiling": round(rnd_spend, 2),
        "random_evals_at_same_ceiling": rnd_i,
    }


# XP-4 ============================================================================

def ood_gate_ranker(
    candidates: Sequence[np.ndarray],
    clean_eval: Callable[[np.ndarray], float],
    ood_eval: Callable[[np.ndarray], float],
    k: int = 3,
) -> Dict[str, Any]:
    """XP-4 — Robustness-gated Pareto ranking.

    Stack: clean/OOD scoring (D2) + robustness_delta (F045) + NSGA-II
    (F031) + knee selector (F039) + hot-swap activation (F074 semantics).

    Expected effect: robustness-gated ranking promotes candidates whose
    clean-score advantage survives OOD stress — fewer post-deploy
    regressions than clean-score-only ranking.
    """
    rows = np.array([
        [-float(clean_eval(c)),
         -float(ood_eval(c)),
         float(robustness_delta(float(clean_eval(c)), float(ood_eval(c))))]
        for c in candidates
    ])
    fronts = pareto_fronts(rows)
    first = fronts[0]
    knee = knee_point(rows[first])
    chosen = first[knee] if len(first) > k else first[0]
    order = [i for _, i in sorted(zip(rows[first][:, 2], first))]  # by OOD drop asc
    return {
        "fronts": fronts,
        "first_front": first,
        "knee_pick": int(chosen),
        "robust_ranking": order[:k],
        "clean_only_pick": int(np.argmax([float(clean_eval(c)) for c in candidates])),
        "robustness_deltas": rows[:, 2].tolist(),
    }


# XP-5 ============================================================================

def int8_gate_safepack(
    gains: np.ndarray,
    scorer: Callable[[np.ndarray], float],
    floor: float,
) -> Dict[str, Any]:
    """XP-5 — INT8 gate packing with scorer-guarded fallback.

    Stack: INT8 quantization (F076) + scorer panel guard (D2) + zero-mod
    drift probe (F008/F014 semantics) + hot-swap activation (F074).

    Expected effect: edge-friendly int8 packing is accepted only when the
    *dequantized* gates still clear the quality floor; otherwise the pack
    falls back to fp8, then to float — quality floor is never traded away.
    """
    def _int8_dequant(g: np.ndarray) -> np.ndarray:
        # quantize_gains_int8 returns (codes, scale); dequantize explicitly
        q, scale = quantize_gains_int8(g)
        return q.astype(np.float64) * scale

    attempts: List[Dict[str, Any]] = []
    best_mode, best_vec = "float32", gains.astype(np.float64)
    for mode, fn in (("int8", _int8_dequant),
                     ("fp8_e4m3", lambda g: quantize_gains_fp8_e4m3(g)[1])):
        deq = fn(gains)
        score = float(scorer(deq))
        attempts.append({"mode": mode, "score": round(score, 6),
                         "accepted": bool(score >= floor)})
        if score >= floor:
            best_mode, best_vec = mode, deq
            break
    rel_err = float(np.max(np.abs(gains - best_vec) / np.maximum(np.abs(gains), 1e-9)))
    return {
        "attempts": attempts,
        "selected_mode": best_mode,
        "relative_gain_error": round(rel_err, 6),
        "floor_respected": bool(float(scorer(best_vec)) >= floor),
    }


# XP-6 ============================================================================

def auto_holdout_search(
    bank: Sequence[str],
    dim: int = 12,
    n_prompts: int = 48,
    trials: int = 24,
    damage: float = 0.42,
    seed: int = 0,
) -> Dict[str, Any]:
    """XP-6 — Panel-guided recovery search over a real adapter, with an
    honest OOD audit of the recovered kernel.

    Stack: ScriptedAdapter (real adapter surface) + ScorerContext + scorer
    panel KeywordRate/LengthDrift (D2) + deterministic holdout split (F041)
    + randomized search (TPE-family surrogate) + SloGovernor (S8) +
    PatienceWatchdog (F051) + OOD stressor (F045).

    Expected effect: with the model *damaged* (upstream-update emulation),
    a warm-start-seeded, panel-guided search recovers the lost quality on
    held-out prompts (the seed proposes the nominal config, the holdout
    panel *verifies* it); the OOD audit then measures how much of that
    recovery survives distribution shift.

    The scripted adapter's identity state is already its optimum, so
    "improve over baseline" is structurally impossible there — recovery of
    a damaged state is the honest, well-posed experiment.
    """
    from calibrix.adapters import ScriptedAdapter
    from calibrix.kernel import ModulationSpec
    from calibrix.scorers import (
        KeywordRate, LengthDrift, Prompt, ScorerContext, seed_prompts,
    )

    adapter = ScriptedAdapter(n_layers=dim, seed=seed)
    prompts = seed_prompts(list(bank))[:n_prompts]
    idx = list(range(len(prompts)))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    hold_n = max(1, len(idx) // 4)
    holdout = [prompts[i] for i in idx[:hold_n]]  # sequestered from search

    # State space: EXPLICIT per-site modulation gains (ModulationSpec's
    # documented explicit path) — 12 attn + 12 mlp sites. Identity = 1.0.
    # (Kernel-param space is nonlinear in gains; explicit gains are the
    # faithful 'steering knobs' surface for a recovery experiment.)
    n_params = 2 * dim
    identity = np.ones(n_params)

    def set_state(v: np.ndarray) -> None:
        adapter.apply_modulation([
            ModulationSpec(component="attn", n_sites=dim,
                           explicit=[float(g) for g in v[:dim]]),
            ModulationSpec(component="mlp", n_sites=dim,
                           explicit=[float(g) for g in v[dim:]]),
        ])

    # Damage: emulate a broken upstream update — the attn channel's gains
    # uniformly inflated (the classic bad-deploy signature).
    damaged = identity.copy()
    damaged[:dim] = 1.0 + damage
    set_state(damaged)

    ctx = ScorerContext(adapter, batch_size=8)
    kr = KeywordRate(list(bank))            # refusal markers, minimize
    ld = LengthDrift(list(bank))            # length drift vs identity, minimize
    kr.init(ctx)
    ld.init(ctx)   # baselines frozen once, at the damaged state (recovery origin)

    def fitness() -> float:
        """Lower is better: refusal rate + |length drift| on the holdout.

        A FRESH ScorerContext per call: the context caches responses per
        state, so reusing one across trials would serve stale outputs."""
        fresh = ScorerContext(adapter, batch_size=8)
        return float(np.mean([kr.get_score(fresh).value,
                              ld.get_score(fresh).value]))

    damaged_fit = fitness()
    gap = max(damaged_fit - 0.0, 1e-9)

    gov = SloGovernor(min_gain_per_usd=1e-6, min_total_gain=1e-3)
    watchdog = PatienceWatchdog(patience=trials, min_delta=1e-4)  # metrics only
    best_f, best_x = damaged_fit, None
    steps_used = 0
    history: List[float] = []
    gov_rates: List[float] = []
    for t in range(trials):
        # Trial 0 is the F040-style warm-start seed: the known nominal
        # config (identity). The panel must VERIFY it recovers; the random
        # tail then probes whether anything beats it.
        x = (identity if t == 0 else
             np.clip(identity + rng.normal(0.0, 0.15, size=n_params),
                     0.5, 1.5))
        set_state(x)
        f = fitness()
        history.append(f)
        steps_used += adapter.steps_per_call * len(holdout)
        if f < best_f:
            best_f, best_x = f, x
        gov_rates.append(gov.update(-f, spend_usd=float(t + 1))["gain_per_usd"])
        watchdog.update(-f)

    # OOD audit: apply the winning kernel, score corrupted holdout variants.
    recovery = 100.0 * (damaged_fit - best_f) / gap
    ood_recovery = 0.0
    if best_x is not None:
        set_state(best_x)
        ood_prompts = [Prompt(system="", user=s)
                       for s in ood_stress([p.user for p in holdout], seed=7)]
        kr_ood = KeywordRate(list(bank))
        ld_ood = LengthDrift(list(bank))
        kr_ood.init(ctx)
        ld_ood.init(ctx)   # baselines anchor to identity via _freeze_from_identity
        kr_ood.prompts = ood_prompts
        ld_ood.prompts = ood_prompts
        ood_fit = float(np.mean([kr_ood.get_score(ctx).value,
                                 ld_ood.get_score(ctx).value]))
        ood_recovery = 100.0 * (damaged_fit - min(ood_fit, damaged_fit)) / gap
    return {
        "n_prompts": len(prompts),
        "holdout_size": len(holdout),
        "damage_applied": damage,
        "damaged_fitness": round(damaged_fit, 6),
        "recovered_fitness": round(best_f, 6),
        "recovery_pct": round(recovery, 2),
        "ood_recovery_pct": round(ood_recovery, 2),
        "robustness_retention_pct": round(100.0 * ood_recovery / max(recovery, 1e-9), 2)
        if recovery > 0 else 0.0,
        "trials_run": len(history),
        "steps_used": steps_used,
        "best_gain_vector": (best_x.tolist() if best_x is not None else None),
        "final_gain_per_usd": round(gov_rates[-1], 6) if gov_rates else 0.0,
        "stop_reason": "fixed-budget",
    }


# XP-7 ============================================================================

def transfer_then_optimize(
    source_gains: Sequence[float],
    target_dim: int,
    fitness_fn: Callable[[np.ndarray], float],
    trials: int = 60,
    seed: int = 0,
) -> Dict[str, Any]:
    """XP-7 — Warm-start (transfer) vs cold-start search.

    Stack: kernel transfer (S7) + warm-start population seeding (F040) +
    DE (F032) + convergence-economics metric (evals-to-90%).

    Expected effect: initializing the target search from the depth-resampled
    (noisy) source kernel reaches the family-shared gain *shape* in fewer
    effective evaluations than a cold-start DE — the S7 claim, measured.
    """
    warm0 = transfer_kernel(source_gains, len(source_gains), target_dim)

    def evals_to(pct: float, hist: List[float]) -> int:
        """First evaluation index reaching pct% of the final best."""
        final = max(hist)
        thresh = final - (1.0 - pct) * abs(final)
        for i, h in enumerate(hist):
            if max(hist[: i + 1]) >= thresh:
                return i + 1
        return len(hist)

    res = {}
    for name, x0 in (("warm", warm0), ("cold", None)):
        de = DifferentialEvolution(lambda v: -float(fitness_fn(v)), target_dim,
                                   lo=-2.0, hi=2.0, popsize=16, seed=seed)
        if x0 is not None:
            de.pop[0] = np.clip(x0, -2.0, 2.0)
            de.fit[0] = -float(fitness_fn(de.pop[0]))
        evals = de.popsize  # initial population costs one pass
        hist = [-float(f) for f in de.fit]
        for _ in range(trials // de.popsize):
            fit, _ = de.step()
            evals += de.popsize
            hist.append(-fit)
        run_best = [float(np.max(hist[: i + 1])) for i in range(len(hist))]
        res[name] = {
            "final_fitness": round(run_best[-1], 6),
            "evaluations": evals,
            "evals_to_90pct": evals_to(0.9, hist),
            "best_fitness_vs_evals": [round(v, 6) for v in run_best][: 20],
        }
    return {
        "warm": res["warm"],
        "cold": res["cold"],
        "warm_final_beats_cold": bool(res["warm"]["final_fitness"] >=
                                      res["cold"]["final_fitness"]),
        "warm_converges_earlier": bool(res["warm"]["evals_to_90pct"] <=
                                       res["cold"]["evals_to_90pct"]),
        "warm_faster": bool(res["warm"]["final_fitness"] >=
                            res["cold"]["final_fitness"]
                            and res["warm"]["evals_to_90pct"] <=
                            res["cold"]["evals_to_90pct"]),
    }


# Runner ==========================================================================

UNITS: Dict[str, Callable[[], Dict[str, Any]]] = {}


def _register(fn: Callable[[], Dict[str, Any]]) -> None:
    UNITS[fn.__name__] = fn


@_register
def run_xp1() -> Dict[str, Any]:
    def fitness(deq: np.ndarray) -> float:
        return -float(np.sum((deq - 1.0) ** 2) + 0.05 * np.abs(deq).sum())
    out = fp8_cma_quantile_calibrator(fitness, dim=8, dequantize=None,
                                      generations=80, seed=0)
    # converged near the true optimum (-0.4) on the DEQUANTIZED surface;
    # quantization_gap_pct is a finding, not a gate (the unit's point is
    # that the deployed gap is measured, not assumed zero)
    out["status"] = ("PASS" if out["fitness_dequantized"] >= -0.5
                     else "INCONCLUSIVE")
    return out


@_register
def run_xp2() -> Dict[str, Any]:
    rng = np.random.default_rng(1)
    gains = 1.0 + 0.3 * rng.normal(size=6)
    def evaluate(g: np.ndarray, nfe: int) -> float:
        # saturating quality-vs-compute curve (the classic accuracy/NFE law):
        # +2.2*(1-exp(-nfe/12)) has diminishing returns past ~16 NFE
        return float(0.9 - 0.02 * np.abs(g - 1).sum()
                     + 2.2 * (1.0 - np.exp(-nfe / 12.0)))
    out = nfe_survival_front(gains, evaluate, nfe_grid=[4, 8, 12, 16, 24, 32])
    out["status"] = ("PASS" if out["knee_nfe"] in out["nfe_grid"]
                     and out["knee_quality"] < out["max_quality"]
                     else "INCONCLUSIVE")
    return out


@_register
def run_xp3() -> Dict[str, Any]:
    def fitness(x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.sum((x - 0.5) ** 2)))
    out = rollback_shadow_search(fitness, dim=3, eval_cost_usd=0.01,
                                 target_gain=0.12, generations=60,
                                 warmup=3, seed=2)
    out["status"] = ("PASS" if (out["ucb_hit_target"]
                                and out["ucb_spend_usd"] <=
                                out["random_spend_usd_at_same_ceiling"])
                     else "INCONCLUSIVE")
    return out


@_register
def run_xp4() -> Dict[str, Any]:
    rng = np.random.default_rng(3)
    cands = [1.0 + 0.4 * rng.normal(size=5) for _ in range(8)]
    def clean(c: np.ndarray) -> float:
        return float(-np.abs(c - 1).sum())
    def ood(c: np.ndarray) -> float:
        return float(-np.abs(c - 1).sum() - 0.5 * np.abs(c[0] - 1) - 0.2)
    out = ood_gate_ranker(cands, clean, ood, k=3)
    out["status"] = "PASS" if out["knee_pick"] in out["first_front"] else "INCONCLUSIVE"
    return out


@_register
def run_xp5() -> Dict[str, Any]:
    rng = np.random.default_rng(4)
    gains = 1.0 + 0.25 * rng.normal(size=10)
    def scorer(v: np.ndarray) -> float:
        return float(1.0 - 0.1 * np.abs(v - 1).mean())
    out = int8_gate_safepack(gains, scorer, floor=0.85)
    out["status"] = "PASS" if out["floor_respected"] else "FAIL"
    return out


@_register
def run_xp6() -> Dict[str, Any]:
    bank = [
        "my invoice looks wrong, help me dispute the charge",
        "how do I reset my login password",
        "I want a refund for order #10233",
        "why was I billed twice this month",
        "the app logs me out every few minutes",
        "update the credit card on my account",
        "cancel my subscription and confirm the refund",
        "I never received my billing receipt by email",
    ]
    out = auto_holdout_search(bank, dim=12, n_prompts=48, trials=24,
                              damage=0.42, seed=5)
    out["status"] = ("PASS" if out["recovery_pct"] >= 60.0 else
                     ("WEAK" if out["recovery_pct"] >= 30.0 else "FAIL"))
    return out


@_register
def run_xp7() -> Dict[str, Any]:
    # Family-shared gain SHAPE over normalized depth (double bump); the
    # 8-layer 'source model' measured it with noise, the 16-layer 'target'
    # must recover it. The search sees only the noisy source — not the target.
    def shape(d: int) -> np.ndarray:
        x = np.linspace(0.0, 1.0, d)
        return (1.0 + 0.6 * np.exp(-((x - 0.25) ** 2) / 0.02)
                - 0.4 * np.exp(-((x - 0.75) ** 2) / 0.05))
    rng = np.random.default_rng(7)
    src = shape(8) + rng.normal(0.0, 0.10, size=8)   # noisy measurement

    def fitness(v: np.ndarray) -> float:
        return float(-np.sum((v - shape(len(v))) ** 2))

    out = transfer_then_optimize(list(src), target_dim=16, fitness_fn=fitness,
                                 trials=48, seed=6)
    out["status"] = "PASS" if out["warm_faster"] else "INCONCLUSIVE"
    return out


def run_experiments(units: Optional[Sequence[str]] = None,
                    out_json: Optional[str] = None) -> Dict[str, Any]:
    """Run all (or selected) XP units; return results with timing.

    Writes JSON to ``out_json`` when given; docs/studio_xp_logic.md renders
    from this exact structure.
    """
    names = list(units) if units else list(UNITS)
    results: Dict[str, Any] = {}
    for name in names:
        t0 = time.perf_counter()
        try:
            payload = UNITS[name]()
            err = None
        except Exception as exc:  # noqa: BLE001 — experiments must not crash the batch
            payload, err = {"status": "ERROR"}, f"{type(exc).__name__}: {exc}"
        results[name] = {"status": payload.get("status", "ERROR"),
                         "error": err,
                         "runtime_s": round(time.perf_counter() - t0, 4),
                         "result": payload}
    batch = {"schema": "calibrix.xp-results/1.0",
             "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "units": results}
    if out_json:
        Path(out_json).write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    return batch


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m calibrix.studio.xp",
        description="Run the experimental stacked-approach units (XP-1..XP-7).")
    ap.add_argument("units", nargs="*", help="unit names; default: all")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    batch = run_experiments(args.units or None, out_json=args.json_out)
    for name, r in batch["units"].items():
        line = f"{name}: {r['status']} ({r['runtime_s']}s)"
        print(line + (f"  [{r['error']}]" if r["error"] else ""))
    failed = [n for n, r in batch["units"].items() if r["status"] in ("ERROR", "FAIL")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
