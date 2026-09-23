# SPDX-License-Identifier: MIT
# Doberwatch Finance — expert task rubrics & evidence-backed grading.
#
# A watchdog that complains about a "bad" answer is useless unless "bad" is
# defined *before* the answer arrives and unless the grade is bound to
# evidence. This module implements that contract:
#
#   * a Rubric is a fixed set of weighted criteria (+ required evidence);
#   * a criterion that requires evidence cannot score above EVIDENCE_CAP
#     without it, and a rubric with expected evidence cannot reach PASS at all
#     without it (the evidence gate);
#   * an anti-drift penalty discounts answers that ignore the request,
#     measured deterministically against a reference answer.
#
# Nothing here talks to a model, a network, or a clock: grade in, grade out.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .similarity import DEFAULT_DIM, similarity

# A criterion that asserts a fact but supplies no evidence keeps at most half
# credit: an unproven claim is worth less than a proven one, never zero, so
# the rubric still distinguishes "unsupported but plausible" from "absent".
EVIDENCE_CAP = 0.5

PASS, GROWL, BARK, BITE = "PASS", "GROWL", "BARK", "BITE"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Criterion:
    """One gradeable requirement: what it is, how much it counts, and whether
    it must be backed by evidence to count in full."""

    name: str
    weight: float
    description: str = ""
    requires_evidence: bool = False
    evidence_kind: str = ""
    # Corroboration is stronger than evidence: evidence says a source was
    # cited, corroboration says independent sources actually agree.
    requires_corroboration: bool = False


@dataclass(frozen=True)
class Rubric:
    """An expert rubric for one (domain, task) pair.

    Verdict bands are ordered cut points on the final 0..1 score:
    ``score >= pass_threshold`` is PASS; the GROWL band sits just below it;
    BARK below that; anything under ``bite_threshold`` is BITE.
    """

    rubric_id: str
    domain: str
    task: str
    criteria: Tuple[Criterion, ...]
    pass_threshold: float = 0.70
    growl_threshold: float = 0.55
    bite_threshold: float = 0.35
    expected_evidence: Tuple[str, ...] = ()
    drift_floor: float = 0.35
    max_drift_penalty: float = 0.30

    def normalized_weights(self) -> Dict[str, float]:
        """Weights rescaled to sum to 1.0 (a rubric that does not total 1 is
        a data-entry slip, not a reason to silently change the verdict)."""
        total = sum(c.weight for c in self.criteria)
        if total <= 0.0:
            raise ValueError(f"rubric {self.rubric_id!r} has no positive weight")
        return {c.name: c.weight / total for c in self.criteria}

    @property
    def required_evidence(self) -> Tuple[str, ...]:
        return tuple(c.evidence_kind or c.name for c in self.criteria
                     if c.requires_evidence)


@dataclass
class Grade:
    """The evidence-backed result of grading one response."""

    rubric_id: str
    domain: str
    score: float
    verdict: str
    complaint: bool
    raw_score: float
    criteria: List[Dict[str, Any]]
    missing_evidence: List[str]
    evidence_supplied: List[str]
    evidence_coverage: float
    evidence_gate_applied: bool
    missing_corroboration: List[str]
    corroboration_coverage: float
    corroboration_gate_applied: bool
    drift: float
    drift_penalty: float
    contradiction_veto: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "domain": self.domain,
            "score": round(self.score, 6),
            "verdict": self.verdict,
            "complaint": self.complaint,
            "raw_score": round(self.raw_score, 6),
            "criteria": self.criteria,
            "missing_evidence": list(self.missing_evidence),
            "evidence_supplied": list(self.evidence_supplied),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "evidence_gate_applied": self.evidence_gate_applied,
            "missing_corroboration": list(self.missing_corroboration),
            "corroboration_coverage": round(self.corroboration_coverage, 4),
            "corroboration_gate_applied": self.corroboration_gate_applied,
            "drift": round(self.drift, 6),
            "drift_penalty": round(self.drift_penalty, 6),
            "contradiction_veto": self.contradiction_veto,
        }


def verdict_for(score: float, rubric: Rubric) -> str:
    """Map a final score onto the Growl / Bark / Bite band."""
    if score >= rubric.pass_threshold:
        return PASS
    if score >= rubric.growl_threshold:
        return GROWL
    if score >= rubric.bite_threshold:
        return BARK
    return BITE


def _truthy_evidence(refs: Optional[Sequence[Any]]) -> List[Any]:
    return [r for r in (refs or []) if r]


def _any_evidence(evidence: Mapping[str, Sequence[Any]]) -> bool:
    return any(_truthy_evidence(refs) for refs in evidence.values())


def requireable(rubric: Rubric) -> bool:
    """True when the rubric declares criteria that demand evidence."""
    return any(c.requires_evidence for c in rubric.criteria)


def corroborates(rubric: Rubric) -> bool:
    """True when the rubric declares criteria that require corroboration."""
    return any(c.requires_corroboration for c in rubric.criteria)


def grade_response(rubric: Rubric,
                   scores: Mapping[str, float],
                   evidence: Optional[Mapping[str, Sequence[Any]]] = None,
                   response: Optional[str] = None,
                   reference: Optional[str] = None,
                   corroboration: Optional[Mapping[str, float]] = None,
                   contradicted: bool = False,
                   dim: int = DEFAULT_DIM) -> Grade:
    """Grade one response against one rubric, with evidence and anti-drift.

    ``scores`` maps criterion name -> 0..1 judgement (a human or an automated
    judge supplies these; this module owns only the aggregation rules).
    ``evidence`` maps criterion name -> non-empty references (policy ids,
    receipt ids, citations).    ``corroboration`` maps criterion name -> 0..1
    independent-source agreement (see ``verification.verify_claim``); a
    criterion that requires it cannot be credited above that value.
    ``contradicted`` says an independent source refutes a material claim in
    the response: a refuted fact is worse than an unverified one, so it vetoes
    PASS outright rather than merely discounting the criterion it maps to.
    ``response``/``reference`` are optional texts; when both are present the
    response's similarity to the reference sets the anti-drift penalty.
    """
    weights = rubric.normalized_weights()
    evidence = evidence or {}
    corroboration = corroboration or {}

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    supplied: List[str] = []
    missing_corr: List[str] = []
    supplied_corr: List[str] = []
    raw_score = 0.0
    required_total = 0.0
    required_met = 0.0
    corr_total = 0.0
    corr_met = 0.0

    for criterion in rubric.criteria:
        refs = _truthy_evidence(evidence.get(criterion.name))
        score_in = _clamp01(scores.get(criterion.name, 0.0))
        capped = score_in
        if criterion.requires_evidence:
            required_total += weights[criterion.name]
            if refs:
                required_met += weights[criterion.name]
                supplied.append(criterion.name)
            else:
                capped = min(capped, EVIDENCE_CAP)
                missing.append(criterion.name)
        applied_corr: Optional[float] = None
        if criterion.requires_corroboration:
            corr_total += weights[criterion.name]
            raw_corr = corroboration.get(criterion.name)
            if raw_corr is None:
                missing_corr.append(criterion.name)
                capped = min(capped, EVIDENCE_CAP)
            else:
                applied_corr = _clamp01(raw_corr)
                capped = min(capped, applied_corr)
                corr_met += weights[criterion.name] * applied_corr
                supplied_corr.append(criterion.name)
        rows.append({
            "criterion": criterion.name,
            "weight": round(weights[criterion.name], 6),
            "judgement": round(score_in, 4),
            "credited": round(capped, 4),
            "requires_evidence": criterion.requires_evidence,
            "evidence_kind": criterion.evidence_kind,
            "evidence_refs": len(refs),
            "requires_corroboration": criterion.requires_corroboration,
            "corroboration": (round(applied_corr, 4)
                              if applied_corr is not None else None),
        })
        raw_score += capped * weights[criterion.name]

    # Evidence coverage is weighted over the criteria that demand evidence;
    # for a rubric with none, it simply records whether any evidence was given.
    if required_total > 0.0:
        coverage = required_met / required_total
    else:
        coverage = 1.0 if _any_evidence(evidence) else 0.0

    gate_applied = False
    score = raw_score
    if rubric.expected_evidence and not supplied and not _any_evidence(evidence):
        # A response that makes claims of this kind with no evidence anywhere
        # can be at most a warning (GROWL) — never a clean PASS.
        gate_applied = True
        score = min(score, rubric.growl_threshold)

    # The corroboration gate: a rubric that demands independent verification
    # cannot PASS on an unverified response, no matter how confident it reads.
    # A *contradicted* material claim triggers the same cap: zeroing the one
    # criterion it happens to map to is not enough when the rest of the answer
    # still reads well, because the answer as a whole is then not true.
    corr_gate_applied = False
    if corroborates(rubric) and (not supplied_corr or contradicted):
        corr_gate_applied = True
        score = min(score, rubric.growl_threshold)

    if corr_total > 0.0:
        corr_coverage = corr_met / corr_total
    else:
        corr_coverage = 1.0

    drift, penalty = 1.0, 0.0
    if response is not None and reference:
        drift = similarity(response, reference, dim=dim)
        if drift < rubric.drift_floor:
            penalty = ((rubric.drift_floor - drift) / rubric.drift_floor
                       * rubric.max_drift_penalty)
        score *= (1.0 - penalty)

    score = _clamp01(score)
    verdict = verdict_for(score, rubric)
    return Grade(
        rubric_id=rubric.rubric_id,
        domain=rubric.domain,
        score=score,
        verdict=verdict,
        complaint=verdict != PASS,
        raw_score=_clamp01(raw_score),
        criteria=rows,
        missing_evidence=missing,
        evidence_supplied=supplied,
        evidence_coverage=coverage,
        evidence_gate_applied=gate_applied,
        missing_corroboration=missing_corr,
        corroboration_coverage=corr_coverage,
        corroboration_gate_applied=corr_gate_applied,
        drift=drift,
        drift_penalty=penalty,
        contradiction_veto=bool(contradicted and corroborates(rubric)),
    )


__all__ = ["EVIDENCE_CAP", "PASS", "GROWL", "BARK", "BITE", "Criterion",
           "Rubric", "Grade", "verdict_for", "grade_response", "requireable",
           "corroborates"]
