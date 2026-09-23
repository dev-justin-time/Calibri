# SPDX-License-Identifier: MIT
# Doberwatch Finance — the watchdog itself: escalation + checkpoint economics.
#
# Two independent watchdogs live here.
#
# 1. Growl / Bark / Bite — the complaint ladder for a *bad response*. A
#    response graded below the rubric's pass threshold is a complaint, and the
#    rubric's verdict band sets the severity:
#
#        PASS   -> no complaint (the response did its job)
#        GROWL  -> minor: ask for rework / a credit
#        BARK   -> major: partial refund + formal written complaint
#        BITE   -> severe: full refund, then chargeback / regulator
#
#    (The refund *amounts* are computed in ``value.py``; this module owns the
#    workflow state so escalation is auditable and replayable.)
#
# 2. Checkpoint-cost watchdog — a checkpoint is any resumable save point in a
#    metered run (spot-instance state, an agent run, a billing checkpoint).
#    The watchdog answers "is it worth checkpointing *now*?" from these
#    formulas, which are the whole point of the component:
#
#        write_usd            = n_bytes / 1e9 * usd_per_gb
#        expected_loss_usd    = eviction_prob * (seconds_since / 3600) * usd_per_hour
#        breakeven_interval_s = write_usd / (usd_per_hour * eviction_prob / 3600)
#
#    Checkpoint when an eviction signal fires, or when the *expected* loss of
#    the work since the last checkpoint already exceeds the cost of writing
#    one. A cumulative budget breaker (D5/F059 semantics, reused) trips
#    fail-closed if checkpointing itself runs past its ceiling.

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..studio.billing import CostBreaker
from .grading import BARK, BITE, GROWL

SEVERITIES = (GROWL, BARK, BITE)

DEFAULT_USD_PER_GB = 0.02
DEFAULT_USD_PER_HOUR = 2.0


# ----------------------------------------------------------------------
# Growl / Bark / Bite — complaint workflow
# ----------------------------------------------------------------------
@dataclass
class Complaint:
    """A filed complaint about one graded, failing response."""

    complaint_id: str
    domain: str
    severity: str
    score: float
    reason: str
    rubric_id: str = ""
    missing_evidence: List[str] = field(default_factory=list)
    stage: int = 1
    status: str = "open"          # open | resolved | escalated | withdrawn
    created_at: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complaint_id": self.complaint_id,
            "domain": self.domain,
            "severity": self.severity,
            "score": round(float(self.score), 6),
            "reason": self.reason,
            "rubric_id": self.rubric_id,
            "missing_evidence": list(self.missing_evidence),
            "stage": self.stage,
            "status": self.status,
            "created_at": self.created_at,
            "history": list(self.history),
        }


class GrowlBarkBite:
    """The complaint ladder: file, act, escalate, resolve — all auditable."""

    def __init__(self, audit: Optional[Callable[[str, Dict[str, Any]], Any]] = None) -> None:
        self._complaints: Dict[str, Complaint] = {}
        self._audit = audit

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self._audit is not None:
            self._audit(event, payload)

    def file(self, domain: str, grade: Any, reason: str = "") -> Complaint:
        """Open a complaint for a *failing* grade (never for a PASS)."""
        if not getattr(grade, "complaint", False):
            raise ValueError("a passing response cannot be complained about")
        severity = getattr(grade, "verdict", GROWL)
        if severity not in SEVERITIES:
            severity = BARK
        complaint = Complaint(
            complaint_id="cmp_" + uuid.uuid4().hex[:12],
            domain=domain,
            severity=severity,
            score=float(getattr(grade, "score", 0.0)),
            reason=reason or (f"{severity}: {getattr(grade, 'rubric_id', '')} "
                              f"scored {getattr(grade, 'score', 0.0):.2f}"),
            rubric_id=str(getattr(grade, "rubric_id", "")),
            missing_evidence=list(getattr(grade, "missing_evidence", []) or []),
        )
        complaint.history.append({"action": "filed", "severity": severity,
                                  "at": complaint.created_at})
        self._complaints[complaint.complaint_id] = complaint
        self._emit("complaint.filed", {"complaint_id": complaint.complaint_id,
                                       "domain": domain, "severity": severity})
        return complaint

    def get(self, complaint_id: str) -> Optional[Complaint]:
        return self._complaints.get(complaint_id)

    def act(self, complaint_id: str, action: str, note: str = "") -> Complaint:
        complaint = self._complaints[complaint_id]
        complaint.history.append({"action": action, "note": note, "at": time.time()})
        self._emit("complaint.action", {"complaint_id": complaint_id,
                                        "action": action})
        return complaint

    def escalate(self, complaint_id: str, note: str = "") -> Complaint:
        complaint = self._complaints[complaint_id]
        complaint.stage += 1
        complaint.status = "escalated"
        complaint.history.append({"action": "escalated", "stage": complaint.stage,
                                  "note": note, "at": time.time()})
        self._emit("complaint.escalated", {"complaint_id": complaint_id,
                                           "stage": complaint.stage})
        return complaint

    def resolve(self, complaint_id: str, note: str = "") -> Complaint:
        complaint = self._complaints[complaint_id]
        complaint.status = "resolved"
        complaint.history.append({"action": "resolved", "note": note,
                                  "at": time.time()})
        self._emit("complaint.resolved", {"complaint_id": complaint_id})
        return complaint

    def open_complaints(self) -> List[Complaint]:
        return [c for c in self._complaints.values() if c.status == "open"]

    def counts(self) -> Dict[str, int]:
        out = {sev: 0 for sev in SEVERITIES}
        for complaint in self._complaints.values():
            out[complaint.severity] = out.get(complaint.severity, 0) + 1
        out["open"] = len(self.open_complaints())
        out["total"] = len(self._complaints)
        return out


# ----------------------------------------------------------------------
# Checkpoint economics (the watchdog's formulas)
# ----------------------------------------------------------------------
def checkpoint_write_usd(n_bytes: float, usd_per_gb: float = DEFAULT_USD_PER_GB) -> float:
    """Cost of writing one checkpoint of ``n_bytes``."""
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")
    if usd_per_gb < 0:
        raise ValueError("usd_per_gb must be >= 0")
    return float(n_bytes) / 1e9 * float(usd_per_gb)


def expected_eviction_loss_usd(seconds_since_checkpoint: float,
                               usd_per_hour: float,
                               eviction_prob: float = 1.0) -> float:
    """Expected value of the metered work lost if eviction hits right now."""
    if not 0.0 <= eviction_prob <= 1.0:
        raise ValueError("eviction_prob must be in [0, 1]")
    return (max(0.0, float(seconds_since_checkpoint)) / 3600.0
            * float(usd_per_hour) * float(eviction_prob))


def breakeven_checkpoint_interval_s(n_bytes: float, usd_per_gb: float,
                                    usd_per_hour: float,
                                    eviction_prob: float = 1.0) -> float:
    """Seconds of exposed work at which checkpointing pays for itself."""
    rate_per_s = float(usd_per_hour) * float(eviction_prob) / 3600.0
    write = checkpoint_write_usd(n_bytes, usd_per_gb)
    if rate_per_s <= 0.0:
        return float("inf")
    return write / rate_per_s


class CheckpointWatchdog:
    """Decide *when* to checkpoint and *how much* checkpointing has cost."""

    def __init__(self, usd_per_gb: float = DEFAULT_USD_PER_GB,
                 usd_per_hour: float = DEFAULT_USD_PER_HOUR,
                 budget_usd: Optional[float] = None,
                 eviction_prob: float = 1.0,
                 min_interval_s: float = 0.0) -> None:
        if not 0.0 <= eviction_prob <= 1.0:
            raise ValueError("eviction_prob must be in [0, 1]")
        self.usd_per_gb = float(usd_per_gb)
        self.usd_per_hour = float(usd_per_hour)
        self.eviction_prob = float(eviction_prob)
        self.min_interval_s = float(min_interval_s)
        self.checkpoints = 0
        self.bytes_written = 0.0
        self.spend_usd = 0.0
        self._breaker = CostBreaker(budget_usd) if budget_usd is not None else None
        self._last_checkpoint_at: Optional[float] = None

    @property
    def tripped(self) -> bool:
        return bool(self._breaker and self._breaker.tripped)

    def should_checkpoint(self, seconds_since_checkpoint: float, n_bytes: float,
                          eviction_signal: bool = False,
                          eviction_prob: Optional[float] = None) -> Dict[str, Any]:
        """Decide whether to write a checkpoint now, and why."""
        prob = self.eviction_prob if eviction_prob is None else float(eviction_prob)
        write = checkpoint_write_usd(n_bytes, self.usd_per_gb)
        loss = expected_eviction_loss_usd(seconds_since_checkpoint,
                                          self.usd_per_hour, prob)
        breakeven = breakeven_checkpoint_interval_s(n_bytes, self.usd_per_gb,
                                                    self.usd_per_hour, prob)
        if eviction_signal:
            due, reason = True, "eviction_signal"
        elif loss >= write:
            due, reason = True, "expected_loss_exceeds_write"
        elif seconds_since_checkpoint < self.min_interval_s:
            due, reason = False, "below_min_interval"
        else:
            due, reason = False, "below_breakeven"
        return {
            "checkpoint": due,
            "reason": reason,
            "write_usd": round(write, 6),
            "expected_loss_usd": round(loss, 6),
            "net_usd": round(loss - write, 6),
            "breakeven_s": round(breakeven, 3) if breakeven != float("inf") else None,
            "eviction_prob": prob,
        }

    def note_checkpoint(self, n_bytes: float, now: Optional[float] = None) -> Dict[str, Any]:
        """Record a checkpoint actually written; trip the breaker past budget."""
        cost = checkpoint_write_usd(n_bytes, self.usd_per_gb)
        self.checkpoints += 1
        self.bytes_written += float(n_bytes)
        self.spend_usd += cost
        self._last_checkpoint_at = now if now is not None else time.time()
        if self._breaker is not None:
            self._breaker.check(self.spend_usd)
        return {"checkpoints": self.checkpoints,
                "write_usd": round(cost, 6),
                "spend_usd": round(self.spend_usd, 6),
                "tripped": self.tripped}

    def report(self) -> Dict[str, Any]:
        budget = self._breaker.limit if self._breaker is not None else None
        return {
            "checkpoints": self.checkpoints,
            "bytes_written": self.bytes_written,
            "spend_usd": round(self.spend_usd, 6),
            "budget_usd": round(budget, 6) if budget is not None else None,
            "remaining_usd": (round(max(0.0, budget - self.spend_usd), 6)
                              if budget is not None else None),
            "tripped": self.tripped,
            "avg_write_usd": (round(self.spend_usd / self.checkpoints, 6)
                              if self.checkpoints else 0.0),
        }


__all__ = [
    "SEVERITIES", "DEFAULT_USD_PER_GB", "DEFAULT_USD_PER_HOUR",
    "Complaint", "GrowlBarkBite",
    "checkpoint_write_usd", "expected_eviction_loss_usd",
    "breakeven_checkpoint_interval_s", "CheckpointWatchdog",
]
