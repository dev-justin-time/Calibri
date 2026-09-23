# SPDX-License-Identifier: MIT
# Doberwatch Finance — fair value, refund strategy, and the escalation ladder.
#
# A complaint is only actionable if the user can say *how much* was lost and
# *what to ask for*. Both are computed here:
#
#   fair_value_usd = reference_price * min(1, score / pass_threshold)
#
# i.e. a response that just clears its rubric is worth the reference price; a
# response at half the pass bar is worth half of it. ``value_ratio`` is
# price_paid / fair_value, so > 1 means "paid more than it was worth".
# Below/at/above banding turns that ratio into a plain-language verdict and a
# refund ask; the escalation ladder says who to ask and by when.
#
# This is dispute *arithmetic from the evidence*, not legal advice: the ask is
# always anchored to the rubric failure and the audit export.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .grading import BARK, BITE, GROWL

DEFAULT_TOLERANCE = 0.15
OVERPAID_CEILING = 1.75

# Recommended rung per severity; refund *fraction* is of the price paid.
REFUND_TABLE: Dict[str, Dict[str, Any]] = {
    GROWL: {"fraction": 0.15, "escalation_rung": 1,
            "strategy": "rework_or_credit",
            "rationale": "Minor rubric failure: the cheapest correct remedy is a "
                         "reworked answer or a goodwill credit, not a refund."},
    BARK: {"fraction": 0.50, "escalation_rung": 2,
           "strategy": "partial_refund",
           "rationale": "Major rubric failure: the answer has to be redelivered "
                        "and part of the fee refunded for the shortfall."},
    BITE: {"fraction": 1.00, "escalation_rung": 4,
           "strategy": "full_refund_then_chargeback",
           "rationale": "Severe failure: recover the full amount, then escalate "
                        "to the regulator / payment reversal if refused."},
}

ESCALATION_LADDER: List[Dict[str, Any]] = [
    {"rung": 1, "actor": "merchant_frontline", "deadline_days": 14,
     "action": "Send a written request citing the rubric failure and the evidence; "
               "ask for rework, a credit, or a refund."},
    {"rung": 2, "actor": "merchant_supervisor", "deadline_days": 14,
     "action": "Escalate to a named supervisor or complaints team with the same "
               "evidence and a stated deadline."},
    {"rung": 3, "actor": "external_ombudsman", "deadline_days": 30,
     "action": "Refer the case to the provider's dispute-resolution scheme or an "
               "industry/financial ombudsman."},
    {"rung": 4, "actor": "regulator", "deadline_days": 30,
     "action": "File a complaint with the consumer-finance regulator for the "
               "jurisdiction, attaching the audit export."},
    {"rung": 5, "actor": "chargeback_or_claims", "deadline_days": 45,
     "action": "Request a card chargeback / payment reversal, or file in "
               "small-claims, with the audit export as the exhibit."},
]


@dataclass
class ValueAssessment:
    """What a response was worth versus what it cost the user."""

    score: float
    pass_threshold: float
    price_paid_usd: float
    reference_price_usd: float
    fair_value_usd: float
    value_ratio: float
    overpaid_usd: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "pass_threshold": round(self.pass_threshold, 6),
            "price_paid_usd": round(self.price_paid_usd, 2),
            "reference_price_usd": round(self.reference_price_usd, 2),
            "fair_value_usd": round(self.fair_value_usd, 2),
            "value_ratio": round(self.value_ratio, 4),
            "overpaid_usd": round(self.overpaid_usd, 2),
            "verdict": self.verdict,
        }


def delivered_quality_fraction(score: float, pass_threshold: float = 0.70) -> float:
    """Fraction of the promised quality actually delivered (capped at 1.0)."""
    if pass_threshold <= 0.0:
        raise ValueError("pass_threshold must be positive")
    return max(0.0, min(1.0, float(score) / float(pass_threshold)))


def assess_fair_value(price_paid_usd: float, score: float,
                      pass_threshold: float = 0.70,
                      reference_price_usd: Optional[float] = None,
                      tolerance: float = DEFAULT_TOLERANCE) -> ValueAssessment:
    """Compare what was charged against what the graded response was worth."""
    if price_paid_usd < 0:
        raise ValueError("price_paid_usd must be >= 0")
    reference = float(reference_price_usd if reference_price_usd is not None
                      else price_paid_usd)
    fair = reference * delivered_quality_fraction(score, pass_threshold)
    if fair <= 0.0:
        ratio = float("inf") if price_paid_usd > 0 else 0.0
    else:
        ratio = float(price_paid_usd) / fair
    if ratio <= 1.0 - tolerance:
        verdict = "underpriced"
    elif ratio <= 1.0 + tolerance:
        verdict = "fair"
    elif ratio <= OVERPAID_CEILING:
        verdict = "overpriced"
    else:
        verdict = "grossly_overpriced"
    return ValueAssessment(
        score=float(score), pass_threshold=float(pass_threshold),
        price_paid_usd=float(price_paid_usd), reference_price_usd=reference,
        fair_value_usd=fair, value_ratio=ratio,
        overpaid_usd=max(0.0, float(price_paid_usd) - fair), verdict=verdict,
    )


def compare_offers(offers: Sequence[Mapping[str, Any]],
                   pass_threshold: float = 0.70,
                   tolerance: float = DEFAULT_TOLERANCE) -> List[Dict[str, Any]]:
    """Rank offers by fair value: best value first (lowest price/fair ratio).

    Each offer is ``{"offer_id", "price_usd", "score"}`` with an optional
    ``reference_price_usd``. Ties break on ``offer_id`` so the ranking is
    reproducible.
    """
    ranked: List[Dict[str, Any]] = []
    for offer in offers:
        assessment = assess_fair_value(
            price_paid_usd=float(offer["price_usd"]),
            score=float(offer["score"]),
            pass_threshold=pass_threshold,
            reference_price_usd=offer.get("reference_price_usd"),
            tolerance=tolerance,
        )
        ranked.append({"offer_id": str(offer.get("offer_id", "")),
                       **assessment.to_dict()})
    ranked.sort(key=lambda row: (row["value_ratio"], row["offer_id"]))
    return ranked


def escalation_path(upto_rung: int = 1) -> List[Dict[str, Any]]:
    """The ladder rungs from the first up to and including ``upto_rung``."""
    return [dict(rung) for rung in ESCALATION_LADDER if rung["rung"] <= upto_rung]


def refund_strategy(severity: str, price_paid_usd: float = 0.0,
                    value: Optional[ValueAssessment] = None,
                    domain: str = "") -> Dict[str, Any]:
    """The refund ask + escalation route for one severity."""
    if severity not in REFUND_TABLE:
        raise ValueError(f"unknown severity {severity!r}")
    spec = REFUND_TABLE[severity]
    ask = (0.0 if price_paid_usd <= 0 else
           float(price_paid_usd) if severity == BITE else
           round(float(price_paid_usd) * spec["fraction"], 2))
    ask = min(ask, max(0.0, float(price_paid_usd)))
    rung = int(spec["escalation_rung"])
    rationale = spec["rationale"]
    if value is not None:
        rationale += (f" Measured fair value {value.fair_value_usd:.2f} against "
                      f"price paid {value.price_paid_usd:.2f} "
                      f"({value.verdict}, ratio {value.value_ratio:.2f}).")
    return {
        "severity": severity,
        "domain": domain,
        "strategy": spec["strategy"],
        "ask_usd": ask,
        "ask_fraction": 1.0 if severity == BITE and price_paid_usd > 0
        else (spec["fraction"] if price_paid_usd > 0 else 0.0),
        "escalation_rung": rung,
        "escalation_path": escalation_path(rung),
        "next_rung": next((r for r in ESCALATION_LADDER if r["rung"] == rung + 1), None),
        "rationale": rationale,
        "template": (
            f"Re: {domain or 'this service'} — rubric failure ({severity}). "
            f"I am requesting {('a refund of %.2f' % ask) if ask else 'corrective action'} "
            "within 14 days, with the attached rubric grade and audit trail as evidence."
        ),
        "evidence_required": list(value.to_dict().keys()) if value is not None else [
            "rubric grade", "audit export"],
    }


__all__ = ["DEFAULT_TOLERANCE", "OVERPAID_CEILING", "REFUND_TABLE",
           "ESCALATION_LADDER", "ValueAssessment", "delivered_quality_fraction",
           "assess_fair_value", "compare_offers", "escalation_path",
           "refund_strategy"]
