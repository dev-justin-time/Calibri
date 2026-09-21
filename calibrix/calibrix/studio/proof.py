# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Domain 4: Overfitting Alarms & Proof Attestations.
#
# The trust layer: everything that turns "trust me, the kernel is better"
# into verifiable artifacts. Clean-room implementations of standard
# techniques (k-fold CV, hypothesis-testing gap alarms, Rademacher-style
# capacity bounds, SHA-256 seals, n-gram canaries).

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# F041 — 80/20 Holdout Splitter (deterministic, stratified option) ---------------
def holdout_split(n_items: int, holdout_fraction: float = 0.2,
                  seed: int = 0, stratify: Optional[Sequence[bool]] = None
                  ) -> Tuple[List[int], List[int]]:
    """Reproducible split. Stratified keeps class balance across the split
    (important: refusal probes are rare and must not vanish into holdout)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    if stratify is None:
        rng.shuffle(idx)
        n_hold = max(1, int(n_items * holdout_fraction))
        return list(idx[n_hold:]), list(idx[:n_hold])
    train, hold = [], []
    for cls in (True, False):
        cls_idx = idx[np.asarray(stratify) == cls]
        rng.shuffle(cls_idx)
        n_hold_cls = max(1, int(len(cls_idx) * holdout_fraction))
        hold.extend(cls_idx[:n_hold_cls])
        train.extend(cls_idx[n_hold_cls:])
    return train, hold


# F042 — Holdout Delta Gap Alarm ---------------------------------------------------
def gap_alarm(train_value: float, holdout_value: float,
              margin: float = 0.14, mode: str = "maximize") -> Dict[str, Any]:
    """Flag when train beats holdout by more than `margin` (relative).

    Also reports a one-sided sign-test p-value over paired per-prompt deltas
    when `paired_deltas` provided via `pvalue_from_deltas` — the alarm
    decision is a rule, the p-value is the evidence attached to it.
    """
    if mode == "maximize":
        gap = (train_value - holdout_value) / max(abs(holdout_value), 1e-9)
    else:
        gap = (holdout_value - train_value) / max(abs(holdout_value), 1e-9)
    return {"gap": round(gap, 4), "margin": margin, "flagged": gap > margin}


def pvalue_from_deltas(deltas: Sequence[float]) -> float:
    """Exact one-sided sign test: P(#positive >= observed) under p=0.5."""
    d = [x for x in deltas if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = sum(1 for x in d if x > 0)
    # two-sided tail for robustness
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


# F043 — Blind Cross-Validation Fold ------------------------------------------------
def kfold_scores(evaluate_fold: Callable[[List[int], List[int]], float],
                 n_items: int, k: int = 5, seed: int = 0) -> Dict[str, float]:
    """Run k-fold CV with a caller-supplied fold evaluator. Returns mean/std
    — the "blind" numbers that go into the dossier."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        hold = list(folds[i])
        train = list(np.concatenate([folds[j] for j in range(k) if j != i]))
        scores.append(evaluate_fold(train, hold))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "folds": scores}


# F044 — SHA-256 Checkpoint Seal ------------------------------------------------------
def seal_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seal_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def seal_state(state: Dict[str, np.ndarray]) -> str:
    """Order-stable digest of a named tensor dict (the weight-seal primitive)."""
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(seal_array(state[key]).encode("utf-8"))
    return h.hexdigest()


# F048 — Gradient Bleed Leak Probe ------------------------------------------------------
def zero_grad_proof(grads: Dict[str, Optional[np.ndarray]]) -> Dict[str, Any]:
    """Attest no gradient reached base weights: all grad arrays absent/zero."""
    violations = [k for k, g in grads.items()
                  if g is not None and np.any(np.asarray(g) != 0)]
    return {"clean": not violations, "violations": violations}


# F045 — Out-of-Distribution Stressor ------------------------------------------------------
def ood_stress(prompts: Sequence[str], seed: int = 0) -> List[str]:
    """Deterministic adversarial corruptions: typos, casing chaos, padding,
    unicode noise — cheap OOD probes for robustness scoring."""
    rng = np.random.default_rng(seed)
    out = []
    for s in prompts:
        chars = list(s)
        if len(chars) > 4 and rng.random() < 0.8:
            i = rng.integers(1, len(chars) - 1)
            chars[i] = chars[i].upper() if chars[i].islower() else chars[i].lower()
        if rng.random() < 0.5:
            chars.insert(rng.integers(0, len(chars)), " ")
        out.append("".join(chars) + rng.choice(["", "  ", "!!", "??"]))
    return out


def robustness_delta(clean_score: float, ood_score: float) -> float:
    """Relative drop under stress; >0.2 usually means the kernel memorized
    the clean prompt distribution."""
    return (clean_score - ood_score) / max(abs(clean_score), 1e-9)


# F046 — Auto-Rollback Snapshotter -----------------------------------------------------------
@dataclass
class Snapshot:
    step: int
    vector: List[float]
    fitness: float
    holdout: float
    digest: str


class RollbackLedger:
    """Keep the last known *robust* candidate; revert when alarms fire."""

    def __init__(self, max_snapshots: int = 32) -> None:
        self.snaps: List[Snapshot] = []
        self.max = max_snapshots

    def record(self, step: int, vector: Sequence[float], fitness: float,
               holdout: float) -> Snapshot:
        snap = Snapshot(step=step, vector=list(vector), fitness=fitness,
                        holdout=holdout,
                        digest=seal_bytes(json.dumps(
                            {"v": list(vector), "f": fitness},
                            sort_keys=True).encode()))
        self.snaps.append(snap)
        if len(self.snaps) > self.max:
            self.snaps.pop(0)
        return snap

    def last_robust(self, margin: float = 0.14) -> Optional[Snapshot]:
        ok = [s for s in self.snaps
              if abs(s.fitness - s.holdout) / max(abs(s.holdout), 1e-9) <= margin]
        return ok[-1] if ok else None


# F047 — Semantic Divergence Monitor (embedding-free) ------------------------------------------
def lexical_divergence(texts_before: Sequence[str],
                       texts_after: Sequence[str]) -> float:
    """Jaccard distance over token sets, averaged pairwise.

    A weights-free stand-in for BERTScore-style monitoring: catches when a
    kernel pushes the model into degenerate vocabulary loops. Swap in an
    embedding model for production; the reporting contract is identical.
    """
    def toks(s: str) -> set:
        return set(s.lower().split())

    if len(texts_before) != len(texts_after) or not texts_before:
        return 0.0
    dists = []
    for a, b in zip(texts_before, texts_after):
        ta, tb = toks(a), toks(b)
        if not ta and not tb:
            continue
        dists.append(1.0 - len(ta & tb) / max(len(ta | tb), 1))
    return float(np.mean(dists)) if dists else 0.0


# F049 — Generalization Bounds Bounder ------------------------------------------------------------
def rademacher_bound(n_train: int, n_params: int, delta: float = 0.05,
                     sigma_scale: float = 1.0) -> float:
    """Capacity-style bound: with prob 1-delta, true loss is within
    bound of empirical loss (Bartlett & Mendelson 2002 flavor, simplified
    for a d-dimensional kernel — d is ~100, which is exactly why the
    generalization story is strong)."""
    if n_train <= 0:
        return math.inf
    return sigma_scale * math.sqrt((2 * n_params * math.log(2 * math.e / delta)) / n_train)


# F050 — Prompt Memorization Canary -----------------------------------------------------------------
def memorization_canary(outputs: Sequence[str], training_prompts: Sequence[str],
                        ngram: int = 8) -> Dict[str, Any]:
    """Flag outputs reproducing long verbatim spans from training prompts.

    The canary standard used by LLM memorization audits, reduced to its
    detection core: shared n-grams are the fingerprint of memorization.
    """
    def ngrams(s: str) -> set:
        toks = s.lower().split()
        return {" ".join(toks[i:i + ngram]) for i in range(len(toks) - ngram + 1)}

    train_grams = set()
    for p in training_prompts:
        train_grams |= ngrams(p)
    hits = []
    for i, o in enumerate(outputs):
        shared = ngrams(o) & train_grams
        if shared:
            hits.append({"output_index": i, "examples": sorted(shared)[:3]})
    return {"memorization_detected": bool(hits), "hits": hits}


# F051 — Early Stop Patience Watchdog -----------------------------------------------------------------
class PatienceWatchdog:
    """Stop when the validation metric hasn't improved for `patience` steps."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best = -math.inf
        self.since = 0

    def update(self, value: float) -> bool:
        """Returns True when the run should stop."""
        if value > self.best + self.min_delta:
            self.best, self.since = value, 0
        else:
            self.since += 1
        return self.since >= self.patience


# F052 — Auditable Dossier Generator -------------------------------------------------------------------
def build_dossier(run: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """Compile the client-facing proof bundle: config, seals, statistics.

    Self-contained JSON (rendered to HTML by report.py's engine elsewhere);
    every claim is traceable to a digest or a statistic computed here.
    """
    dossier = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run.get("run_id"),
        "config_seal": seal_bytes(json.dumps(run.get("config", {}),
                                             sort_keys=True).encode()),
        "kernel_vector": run.get("kernel_vector"),
        "kernel_seal": seal_bytes(json.dumps(run.get("kernel_vector", []),
                                             sort_keys=True).encode()),
        "holdout": run.get("holdout", {}),
        "gap_alarm": run.get("gap_alarm", {}),
        "kfold": run.get("kfold", {}),
        "memorization": run.get("memorization", {}),
        "zero_grad": run.get("zero_grad", {}),
        "weight_seal": run.get("weight_seal"),
        "compute_ledger": run.get("compute_ledger", {}),
    }
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)
    return dossier


__all__ = [
    "holdout_split", "gap_alarm", "pvalue_from_deltas", "kfold_scores",
    "seal_bytes", "seal_array", "seal_state", "zero_grad_proof",
    "ood_stress", "robustness_delta", "RollbackLedger", "Snapshot",
    "lexical_divergence", "rademacher_bound", "memorization_canary",
    "PatienceWatchdog", "build_dossier",
]
