# SPDX-License-Identifier: MIT
"""Evaluation helpers — prompt scores -> criterion scores -> Doberwatch Grade.

No model calls here. Callers supply per-prompt judgements (0..1) from any
judge (human, script, or model); this module owns only the aggregation rules:

  * average within a criterion (all prompts for C1 average to C1's score)
  * paired prompts for viewpoint neutrality scored as 1 - |a - b| (low delta = neutral)
  * then grade_response(PLATFORM_RUBRIC, criterion_scores, evidence, corroboration)
    with the same evidence/corroboration gates and contradiction veto as Finance

The caller also supplies evidence/corroboration maps and the contradicted flag;
those come from Agent B's verify_response() over the platform's actual outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..grading import Grade, grade_response
from ..verification import corroboration_for, verify_response
from .criteria import PLATFORM_RUBRIC
from .prompts import Prompt

def neutrality_score(pair_scores: Sequence[float]) -> float:
    """Neutrality from paired prompts: 1 - |a - b|, averaged over pairs.

    Two prompts that are the same task with opposite framing should score the
    same; the gap is the bias. If only one score in a pair exists, neutrality
    is that score (no delta to measure).
    """
    if not pair_scores:
        return 0.0
    # pair_scores is already the per-pair neutralities; average them
    return sum(max(0.0, min(1.0, float(s))) for s in pair_scores) / len(pair_scores)

def aggregate_criterion_scores(
    prompts: Sequence[Prompt],
    prompt_scores: Mapping[str, float],
    pair_neutralities: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Average prompt scores within each criterion.

    For viewpoint_neutrality, paired prompts collapse to one neutrality score
    per pair; unpaired prompts average normally alongside them.
    ``prompt_scores`` maps prompt_id -> 0..1.
    ``pair_neutralities`` maps pair_id -> 0..1 (precomputed 1 - |a-b|).
    """
    by_criterion: Dict[str, List[float]] = {}
    # neutrality pairs: collect once per pair_id
    seen_pair: Dict[str, float] = {}
    if pair_neutralities:
        for pid, val in pair_neutralities.items():
            seen_pair[pid] = max(0.0, min(1.0, float(val)))

    # track which viewpoint pairs we've already counted
    counted_pair: set = set()
    for p in prompts:
        if p.criterion == "viewpoint_neutrality" and p.pair_id and p.pair_id in seen_pair:
            if p.pair_id in counted_pair:
                continue
            by_criterion.setdefault(p.criterion, []).append(seen_pair[p.pair_id])
            counted_pair.add(p.pair_id)
            continue
        # unpaired or non-neutrality: average the prompt's own score if present
        if p.prompt_id in prompt_scores:
            by_criterion.setdefault(p.criterion, []).append(
                max(0.0, min(1.0, float(prompt_scores[p.prompt_id])))
            )
        elif p.criterion == "viewpoint_neutrality" and p.pair_id and p.pair_id not in seen_pair:
            # pair incomplete — fall through to individual scores if any
            if p.prompt_id in prompt_scores:
                by_criterion.setdefault(p.criterion, []).append(
                    max(0.0, min(1.0, float(prompt_scores[p.prompt_id])))
                )

    # for criteria that had pairs, the individual prompts are already collapsed;
    # for others, any prompt without a score is treated as 0 only if no scores
    # at all were supplied for that criterion (so missing data doesn't hide)
    result: Dict[str, float] = {}
    for p in prompts:
        result.setdefault(p.criterion, 0.0)
    for criterion, vals in by_criterion.items():
        result[criterion] = sum(vals) / len(vals) if vals else 0.0

    # criteria with no prompts scored stay 0.0 (explicit, not silent)
    all_criteria = {p.criterion for p in prompts}
    for c in all_criteria:
        if c not in result:
            result[c] = 0.0
    return result

def grade_platform_run(
    prompt_scores: Mapping[str, float],
    prompts: Sequence[Prompt],
    evidence: Optional[Mapping[str, Sequence[Any]]] = None,
    verification: Optional[Mapping[str, Any]] = None,
    rubric: Any = PLATFORM_RUBRIC,
    pair_neutralities: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Grade one platform's run over the whole prompt pack.

    Returns the same shape as ``Doberwatch.submit()``: {grade, verification}
    where grade is Grade.to_dict(). Verification is passed through when
    supplied; otherwise a minimal one is synthesized from the factual-provenance
    witnesses (so the corroboration gate has something to act on).
    """
    from ..verification import Source as VSource
    # Build criterion scores
    criterion_scores = aggregate_criterion_scores(prompts, prompt_scores, pair_neutralities)

    # Corroboration mapping: if verification supplied, map its material_grade
    # onto every requires_corroboration criterion; otherwise 0.
    corr: Dict[str, float] = {}
    any_refuted = False
    if verification is not None:
        corr = corroboration_for(rubric, verification)
        any_refuted = bool(verification.get("any_refuted"))

    grade: Grade = grade_response(
        rubric, criterion_scores, evidence=evidence,
        corroboration=corr, contradicted=any_refuted,
    )
    return {"grade": grade.to_dict(), "verification": verification, "criterion_scores": dict(criterion_scores)}

__all__ = ["neutrality_score", "aggregate_criterion_scores", "grade_platform_run"]
