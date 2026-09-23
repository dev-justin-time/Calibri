# SPDX-License-Identifier: MIT
# Doberwatch Finance — fact-check every response; verify with other sources.
#
# A single response is a claim, not a fact. This module grades a claim by how
# many *independent* sources support it and whether any source contradicts it,
# entirely deterministically (no model, no network):
#
#   support      : topically close AND same factual polarity
#   contradict   : topically close AND (negation mismatch OR numeric conflict)
#   neutral      : neither
#
# Numeric conflict is the key trick: if a claim and a close source assert
# *different* numbers and share none, they disagree (2 minutes vs 30 seconds).
# Corroboration is what unlocks a full grade: one source caps at
# SINGLE_SOURCE_CAP, contradictions zero it out. ``cross_model_agreement``
# does the same for several models answering the same prompt.
#
# Materiality decides what "everything that matters" means: claims carrying
# numbers, amounts, deadlines, or legal/citation language are graded first and
# carry the response's overall verification grade.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .similarity import DEFAULT_DIM, cosine, hash_embedding, similarity

VERIFICATION_SCHEMA = "calibrix.doberwatch.verification/1.0"

SUPPORT_SIM = 0.45
CONTRADICTION_SIM = 0.40
MIN_SOURCES = 2
SINGLE_SOURCE_CAP = 0.5

# Prior trust by evidence kind (used when a Source does not state its own).
TRUST_BY_KIND: Dict[str, float] = {
    "authority": 0.90,   # primary docs, regulators, official APIs
    "citation": 0.80,    # a cited paper/spec
    "document": 0.70,    # internal doc, statement, receipt
    "model": 0.50,       # another model's output — one witness, not truth
    "unverified": 0.30,
}

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
# A sentence ends at . ! ? or ; — but *not* at a decimal point. Splitting on
# every full stop turned "section 2.1 of the agreement" into the fragments
# "section 2" and "1 of the agreement", so a claim about a numbered clause
# could never match a source that spells the number the one correct way.
_SENTENCE = re.compile(r"(?:\d+[.,]\d+|[^.!?\n;])+")
# Negations decide polarity, and a polarity mismatch turns an agreeing source
# into a contradiction, so this list is load-bearing.
_NEGATIONS = frozenset({
    "not", "no", "never", "none", "cannot", "false", "incorrect", "wrong",
    "denied", "unsupported", "unverified", "invalid", "misleading", "disputed",
    "refuted",
    # Contractions, stored without the apostrophe: has_negation() normalizes
    # "isn't" -> "isnt" before looking up, because matching the raw token
    # against an apostrophe-free set meant every contraction was invisible.
    "cant", "wont", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt",
    "werent", "hasnt", "havent", "hadnt", "shouldnt", "wouldnt", "couldnt",
    "aint",
})
_UNITS = frozenset({
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "step", "steps",
    "gpu", "gpus", "percent", "%",
})
_LEGAL = frozenset({
    "policy", "clause", "section", "act", "rule", "rules", "regulation", "law",
    "deadline", "fee", "fees", "charge", "charges", "refund", "rate", "rates",
    "must", "required", "entitled", "liable", "liability", "warranty", "guarantee",
})
_MONEY = re.compile(r"[$€£]\s?\d")


@dataclass(frozen=True)
class Source:
    """One independent witness. ``provider``/``derived_from`` mark provenance."""

    source_id: str
    text: str
    kind: str = "document"
    trust: Optional[float] = None
    independent: bool = True
    url: str = ""
    derived_from: str = ""

    def weight(self) -> float:
        base = self.trust if self.trust is not None else TRUST_BY_KIND.get(
            self.kind, TRUST_BY_KIND["document"])
        base = max(0.0, min(1.0, float(base)))
        if not self.independent:
            base *= 0.5
        if self.derived_from:
            base *= 0.5   # borrowed from another witness: not independent
        return base


@dataclass
class ClaimVerification:
    claim: str
    status: str                    # corroborated|contested|single_source|contradicted|unverified
    grade: float
    support: List[str] = field(default_factory=list)
    contradict: List[str] = field(default_factory=list)
    neutral: int = 0
    independent_support: int = 0
    support_strength: float = 0.0
    agreement: float = 0.0
    material: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "grade": round(self.grade, 6),
            "support": list(self.support),
            "contradict": list(self.contradict),
            "neutral": self.neutral,
            "independent_support": self.independent_support,
            "support_strength": round(self.support_strength, 4),
            "agreement": round(self.agreement, 4),
            "material": self.material,
            "reasons": list(self.reasons),
        }


# ----------------------------------------------------------------------
# cheap deterministic primitives
# ----------------------------------------------------------------------
def normalize_number(token: str) -> str:
    """Canonical form of a numeric token.

    ``,`` is a thousands separator and a trailing ``.`` is punctuation; trailing
    zeros are dropped *only after a decimal point*. Stripping them off integers
    (the old behaviour: ``"30" -> "3"``) made different numbers compare equal,
    so ``"within 30 days"`` and ``"within 3 days"`` were read as agreement — a
    contradiction the watchdog exists to catch, silently passing as support.
    """
    text = token.replace(",", "").rstrip(".")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def extract_numbers(text: str) -> frozenset:
    return frozenset(normalize_number(t) for t in _NUM.findall(text))


def has_negation(text: str) -> bool:
    """True when the text asserts a negative (in either contraction form).

    The lookup strips apostrophes, so "isn't", "isnt" and "is not" all count;
    matching the raw token against an apostrophe-free set missed every
    contracted negation, which silently flipped polarity — and a polarity
    mismatch is read as *contradiction*, not support.
    """
    for word in re.findall(r"[a-z']+", text.lower()):
        if word in _NEGATIONS or word.replace("'", "") in _NEGATIONS:
            return True
    return False


def numeric_conflict(claim: str, source: str) -> bool:
    """Close but disagreeing numbers: both assert numbers and share none."""
    a, b = extract_numbers(claim), extract_numbers(source)
    return bool(a) and bool(b) and a.isdisjoint(b)


def material_signals(text: str) -> Dict[str, int]:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return {
        "numbers": len(extract_numbers(text)),
        "units": len(words & _UNITS),
        "legal": len(words & _LEGAL),
        "money": len(_MONEY.findall(text)),
    }


def is_material(text: str) -> bool:
    """Does this claim carry a fact worth grading (number/deadline/citation)?"""
    return any(material_signals(text).values())


def materiality_score(text: str) -> float:
    signals = material_signals(text)
    score = sum(signals.values()) / 6.0
    return max(0.0, min(1.0, score))


# ----------------------------------------------------------------------
# claim vs source
# ----------------------------------------------------------------------
def compare_claim_to_source(claim: str, source_text: str,
                            dim: int = DEFAULT_DIM) -> str:
    """Classify one source as 'support', 'contradict', or 'neutral'.

    A close source that disagrees on polarity OR asserts different numbers is
    a *contradiction*, never support — agreeing in tone is not agreeing in
    fact.
    """
    closeness = similarity(claim, source_text, dim=dim)
    same_polarity = has_negation(claim) == has_negation(source_text)
    conflicting_numbers = numeric_conflict(claim, source_text)
    if closeness >= SUPPORT_SIM and same_polarity and not conflicting_numbers:
        return "support"
    if closeness >= CONTRADICTION_SIM and (not same_polarity or conflicting_numbers):
        return "contradict"
    return "neutral"


def verify_claim(claim: str, sources: Sequence[Source],
                 min_sources: int = MIN_SOURCES,
                 dim: int = DEFAULT_DIM) -> ClaimVerification:
    """Grade one claim by independent corroboration and contradiction."""
    supporting = [s for s in sources if compare_claim_to_source(claim, s.text, dim) == "support"]
    contradicting = [s for s in sources
                     if compare_claim_to_source(claim, s.text, dim) == "contradict"]
    neutral = len(sources) - len(supporting) - len(contradicting)

    weights = [s.weight() for s in supporting]
    support_strength = float(np.mean(weights)) if weights else 0.0
    independent = len([s for s in supporting if s.independent and not s.derived_from])
    w_s = sum(weights)
    w_c = sum(s.weight() for s in contradicting)
    agreement = w_s / (w_s + w_c) if (w_s + w_c) > 0 else 0.0

    reasons: List[str] = []
    if contradicting and not supporting:
        status = "contradicted"
    elif contradicting:
        status = "contested"
    elif independent >= min_sources:
        status = "corroborated"
    elif supporting:
        status = "single_source"
    else:
        status = "unverified"

    factor = min(1.0, independent / max(1, min_sources))
    if status in ("contradicted", "unverified"):
        grade = 0.0
    else:
        grade = support_strength * factor * (0.5 if status == "contested" else 1.0)
    if status == "single_source":
        grade = min(grade, SINGLE_SOURCE_CAP)
        reasons.append(f"only {independent} independent source(s); "
                       f"capped at {SINGLE_SOURCE_CAP}")
    if contradicting:
        reasons.append("contradicted by " + ", ".join(s.source_id for s in contradicting))
    if neutral and not supporting:
        reasons.append("no source was close enough to support the claim")

    return ClaimVerification(
        claim=claim, status=status, grade=max(0.0, min(1.0, grade)),
        support=[s.source_id for s in supporting],
        contradict=[s.source_id for s in contradicting],
        neutral=neutral, independent_support=independent,
        support_strength=support_strength, agreement=agreement,
        material=is_material(claim), reasons=reasons,
    )


# ----------------------------------------------------------------------
# response-level verification
# ----------------------------------------------------------------------
def split_claims(text: str, min_words: int = 3) -> List[str]:
    parts = [p.strip() for p in _SENTENCE.findall(text)]
    return [p for p in parts if len(p.split()) >= min_words]


def verify_response(response: str, sources: Sequence[Source],
                    claims: Optional[Sequence[str]] = None,
                    dim: int = DEFAULT_DIM) -> Dict[str, Any]:
    """Verify every material claim in a response; grade what matters."""
    segments = list(claims) if claims is not None else split_claims(response)
    graded = [s for s in segments if is_material(s)]
    targets = graded or segments
    verifications = [verify_claim(c, sources, dim=dim) for c in targets]

    material_grades = [v.grade for v in verifications if v.material] or \
                      [v.grade for v in verifications]
    material_grade = float(np.mean(material_grades)) if material_grades else 0.0
    weakest = (min(verifications, key=lambda v: v.grade)
               if verifications else None)
    contradictions = [v for v in verifications if v.status == "contradicted"]
    # A *contested* claim (one witness refutes it, another supports it) is no
    # safer than a contradicted one: a material fact some independent source
    # denies must not pass just because a second source agreed. Both are
    # reported separately — "refuted" is the set a grader should act on.
    refuted = [v for v in verifications
               if v.status in ("contradicted", "contested")]
    counts: Dict[str, int] = {}
    for v in verifications:
        counts[v.status] = counts.get(v.status, 0) + 1

    return {
        "schema": VERIFICATION_SCHEMA,
        "claim_count": len(verifications),
        "material_count": len(graded),
        "material_grade": material_grade,
        "all_corroborated": bool(verifications) and all(
            v.status == "corroborated" for v in verifications),
        "status_counts": counts,
        "contradictions": [v.to_dict() for v in contradictions],
        "refuted": [v.to_dict() for v in refuted],
        "any_refuted": bool(refuted),
        "weakest": weakest.to_dict() if weakest else None,
        "claims": [v.to_dict() for v in
                   sorted(verifications, key=lambda v: (not v.material, v.grade))],
    }


def corroboration_for(rubric: Any, verification: Mapping[str, Any]) -> Dict[str, float]:
    """Map a response-level verification onto every criterion that demands it."""
    grade = float(verification.get("material_grade", 0.0))
    return {c.name: grade for c in rubric.criteria if c.requires_corroboration}


# ----------------------------------------------------------------------
# cross-model agreement
# ----------------------------------------------------------------------
def cross_model_agreement(responses: Mapping[str, str],
                          dim: int = DEFAULT_DIM) -> Dict[str, Any]:
    """How much do several models agree? Outliers are flagged, not trusted."""
    names = list(responses)
    if not names:
        return {"schema": VERIFICATION_SCHEMA, "models": [], "agreement": 0.0,
                "pairwise": {}, "consensus": {}, "divergent": []}
    embeddings = {n: hash_embedding(responses[n], dim) for n in names}
    pairwise: Dict[str, float] = {}
    consensus: Dict[str, float] = {}
    for i, a in enumerate(names):
        others = []
        for j, b in enumerate(names):
            if i == j:
                continue
            key = "|".join(sorted((a, b)))
            if key not in pairwise:
                pairwise[key] = cosine(embeddings[a], embeddings[b])
            others.append(pairwise[key])
        consensus[a] = float(np.mean(others)) if others else 1.0
    agreement = float(np.mean(list(pairwise.values()))) if pairwise else 1.0
    mean_c = float(np.mean(list(consensus.values()))) if consensus else 0.0
    spread = float(np.std(list(consensus.values()))) if consensus else 0.0
    divergent = [n for n, value in consensus.items()
                 if value < mean_c - max(spread, 0.05)]
    return {
        "schema": VERIFICATION_SCHEMA,
        "models": names,
        "agreement": agreement,
        "pairwise": {k: round(v, 6) for k, v in pairwise.items()},
        "consensus": {k: round(v, 6) for k, v in consensus.items()},
        "divergent": sorted(divergent),
    }


__all__ = [
    "VERIFICATION_SCHEMA", "SUPPORT_SIM", "CONTRADICTION_SIM", "MIN_SOURCES",
    "SINGLE_SOURCE_CAP", "TRUST_BY_KIND", "Source", "ClaimVerification",
    "normalize_number", "extract_numbers", "has_negation", "numeric_conflict",
    "material_signals", "is_material", "materiality_score",
    "compare_claim_to_source", "verify_claim", "split_claims",
    "verify_response", "corroboration_for", "cross_model_agreement",
]
