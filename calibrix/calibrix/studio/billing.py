# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Domain 5: Financial Metering & Retainer Arbitrage.
#
# The margin-protecting layer. calibrix.metering records usage; this module
# makes *decisions* with it: hard ceilings, spot-market bidding, failover,
# quotas, gainshare math, invoicing. Everything is deterministic and
# unit-testable — billing logic must never "probably work".

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# F053 — plan_budget Verification Core --------------------------------------------
class BudgetGuard:
    """Hard ceiling around any metered activity (search, eval, export)."""

    def __init__(self, max_usd: float, price_per_1k_steps: float) -> None:
        if max_usd < 0:
            raise ValueError("max_usd must be >= 0")
        self.max_usd = float(max_usd)
        self.price = float(price_per_1k_steps)
        self.steps_used = 0

    def charge(self, steps: int) -> None:
        self.steps_used += int(steps)

    @property
    def spent_usd(self) -> float:
        return self.steps_used / 1000.0 * self.price

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd)

    def can_afford(self, steps: int) -> bool:
        return self.spent_usd + steps / 1000.0 * self.price <= self.max_usd

    def max_affordable_steps(self) -> int:
        return int(self.remaining_usd / self.price * 1000)


# F059 — Hard Cost Limit Circuit Breaker -------------------------------------------
class CostBreaker:
    """Killswitch: trips when spend exceeds quote by `tolerance`. Once
    tripped, stays tripped (fail-closed) until explicitly reset."""

    def __init__(self, quoted_usd: float, tolerance: float = 0.10) -> None:
        self.limit = float(quoted_usd) * (1.0 + tolerance)
        self.tripped = False
        self.tripped_at: Optional[float] = None

    def check(self, spent_usd: float) -> bool:
        if not self.tripped and spent_usd > self.limit:
            self.tripped = True
            self.tripped_at = time.time()
        return self.tripped


# F054 — RunPod Spot Instance Bidder (provider-agnostic pricing core) --------------
@dataclass
class SpotQuote:
    provider: str
    gpu: str
    on_demand_usd_hr: float
    spot_usd_hr: float

    @property
    def discount(self) -> float:
        return 1.0 - self.spot_usd_hr / max(self.on_demand_usd_hr, 1e-9)


def best_spot(catalog: Sequence[SpotQuote], min_discount: float = 0.4
              ) -> Optional[SpotQuote]:
    """Pick the deepest discount above the reliability floor. Real providers
    expose live prices via API; this core consumes any of them uniformly."""
    eligible = [q for q in catalog if q.discount >= min_discount]
    return min(eligible, key=lambda q: q.spot_usd_hr) if eligible else None


def job_cost_usd(quote: SpotQuote, gpu_hours: float) -> float:
    return quote.spot_usd_hr * max(gpu_hours, 0.0)


# F055 — Retainer Margin Arbitrage ---------------------------------------------------
def retainer_margin(retainer_usd: float, compute_usd: float) -> Dict[str, float]:
    """Retainer minus compute = margin. Negative margin surfaces *before*
    month-end, which is the whole point of metering."""
    return {
        "retainer_usd": round(retainer_usd, 2),
        "compute_usd": round(compute_usd, 2),
        "margin_usd": round(retainer_usd - compute_usd, 2),
        "margin_pct": round((retainer_usd - compute_usd) / max(retainer_usd, 1e-9), 4),
    }


# F063 — Dynamic Gainshare Calculator -------------------------------------------------
def gainshare(baseline_cost_usd: float, calibrated_cost_usd: float,
              share: float = 0.25) -> Dict[str, float]:
    """Client keeps (1-share) of verified savings; you bill share."""
    savings = max(0.0, baseline_cost_usd - calibrated_cost_usd)
    return {
        "savings_usd": round(savings, 2),
        "client_keeps_usd": round(savings * (1 - share), 2),
        "provider_fee_usd": round(savings * share, 2),
        "share": share,
    }


# F091 shares math with F063 (settlement.py re-exports it with contract ids)

# F056 — Per-Trial Token Cost Ticker ----------------------------------------------------
@dataclass
class MicroMeter:
    """FLOP/token-level meter: trial -> micro-USD. Feeds per-candidate cost
    into the Pareto selection (a 'better' trial that costs 10x must win by
    enough — otherwise the cheaper one is the right answer)."""

    usd_per_1k_flops: float = 1e-9

    def __init__(self, usd_per_1k_flops: float = 1e-9) -> None:
        self.usd_per_1k_flops = usd_per_1k_flops
        self.trial_costs: Dict[int, float] = {}

    def charge_trial(self, trial: int, flops: float) -> float:
        cost = flops / 1000.0 * self.usd_per_1k_flops
        self.trial_costs[trial] = self.trial_costs.get(trial, 0.0) + cost
        return cost

    def total(self) -> float:
        return sum(self.trial_costs.values())


# F057 — Spot Eviction Failover Gate --------------------------------------------------------
@dataclass
class EvictionPlan:
    checkpoint_every_s: int
    fallback_provider: str
    max_resume_minutes: int


def should_checkpoint(seconds_since_last: int, plan: EvictionPlan,
                      eviction_signal: bool = False) -> bool:
    """Checkpoint on signal, or on cadence (whichever first)."""
    return eviction_signal or seconds_since_last >= plan.checkpoint_every_s


# F058 — Lambda Labs BYOC Key Relay (credential hygiene core) --------------------------------
class ByocRelay:
    """Hold customer cloud keys for the duration of a job only.

    Keys live in memory, are never logged, zeroized on release. This is the
    *minimum viable* handling; production adds an external secrets manager
    (D7/F090) — the interface here matches so it swaps in.
    """

    def __init__(self) -> None:
        self._keys: Dict[str, str] = {}

    def submit(self, job_id: str, api_key: str) -> None:
        self._keys[job_id] = api_key

    def fetch(self, job_id: str) -> Optional[str]:
        return self._keys.get(job_id)

    def release(self, job_id: str) -> None:
        self._keys.pop(job_id, None)

    def zeroize_all(self) -> None:
        for k in self._keys:
            self._keys[k] = ""
        self._keys.clear()


# F060 — SaaS Multi-Tenant Quota Guard ----------------------------------------------------------
class QuotaGuard:
    """Concurrent-job and monthly-compute quotas per workspace."""

    def __init__(self, max_concurrent: int, monthly_step_cap: int) -> None:
        self.max_concurrent = max_concurrent
        self.monthly_step_cap = monthly_step_cap
        self._active: Dict[str, int] = {}
        self._monthly: Dict[str, int] = {}

    def try_admit(self, org_id: str) -> bool:
        if self._active.get(org_id, 0) >= self.max_concurrent:
            return False
        self._active[org_id] = self._active.get(org_id, 0) + 1
        return True

    def release(self, org_id: str) -> None:
        self._active[org_id] = max(0, self._active.get(org_id, 0) - 1)

    def charge_steps(self, org_id: str, steps: int) -> bool:
        """False = cap exceeded; caller must stop or upgrade the plan."""
        new_total = self._monthly.get(org_id, 0) + steps
        if new_total > self.monthly_step_cap:
            return False
        self._monthly[org_id] = new_total
        return True


# F061 — VRAM Idle Reclaim Automator -----------------------------------------------------------------
class IdleReclaimer:
    """Track per-model idle seconds; emit unload signals past the threshold."""

    def __init__(self, idle_threshold_s: int = 300) -> None:
        self.idle_threshold_s = idle_threshold_s
        self._last_used: Dict[str, float] = {}

    def touch(self, model_id: str, now: Optional[float] = None) -> None:
        self._last_used[model_id] = now if now is not None else time.time()

    def should_unload(self, model_id: str, now: Optional[float] = None) -> bool:
        if model_id not in self._last_used:
            return False
        t = now if now is not None else time.time()
        return (t - self._last_used[model_id]) > self.idle_threshold_s


# F062 — CO2 Emissions Carbon Ticker --------------------------------------------------------------------
# Regional grid intensity (kg CO2e / kWh), published averages — data, not code.
GRID_INTENSITY = {
    "us-east": 0.00036, "us-west": 0.00023, "eu-central": 0.00028,
    "eu-north": 0.00003, "ap-south": 0.00071,
}


def co2_grams(kwh: float, region: str = "us-east") -> float:
    return kwh * GRID_INTENSITY.get(region, 0.0004) * 1000.0


def kwh_from_gpu_hours(gpu_hours: float, gpu_watts: float = 400.0,
                       pue: float = 1.2) -> float:
    """Energy = power * hours * PUE (datacenter overhead factor)."""
    return gpu_hours * gpu_watts / 1000.0 * pue


# F064 — Net-30 Invoice Generator -----------------------------------------------------------------------
def build_invoice(line_items: Sequence[Dict[str, Any]], terms_days: int = 30,
                  currency: str = "USD") -> Dict[str, Any]:
    """Itemized invoice from metered line items: [{desc, qty, unit_usd}]."""
    lines, total = [], 0.0
    for li in line_items:
        amount = round(float(li["qty"]) * float(li["unit_usd"]), 2)
        total += amount
        lines.append({**li, "amount_usd": amount})
    return {
        "invoice_id": f"INV-{time.strftime('%Y%m%d-%H%M%S')}",
        "issued": time.strftime("%Y-%m-%d"),
        "net_due_days": terms_days,
        "currency": currency,
        "lines": lines,
        "total_usd": round(total, 2),
    }


# F065 — Credit Balance Auto-Debit -----------------------------------------------------------------------
class CreditWallet:
    """Pre-funded wallet: reserve before dispatch, settle actuals after."""

    def __init__(self, balance_usd: float = 0.0) -> None:
        self.balance = float(balance_usd)
        self._reserved: Dict[str, float] = {}

    def fund(self, usd: float) -> None:
        self.balance += float(usd)

    def reserve(self, job_id: str, usd: float) -> bool:
        if usd > self.balance:
            return False
        self.balance -= usd
        self._reserved[job_id] = usd
        return True

    def settle(self, job_id: str, actual_usd: float) -> Dict[str, float]:
        reserved = self._reserved.pop(job_id, 0.0)
        refund = max(0.0, reserved - actual_usd)
        extra = max(0.0, actual_usd - reserved)
        self.balance -= extra
        self.balance += refund
        return {"settled_usd": round(actual_usd, 2),
                "refund_usd": round(refund, 2),
                "balance_usd": round(self.balance, 2)}


__all__ = [
    "BudgetGuard", "CostBreaker", "SpotQuote", "best_spot", "job_cost_usd",
    "retainer_margin", "gainshare", "MicroMeter", "EvictionPlan",
    "should_checkpoint", "ByocRelay", "QuotaGuard", "IdleReclaimer",
    "co2_grams", "kwh_from_gpu_hours", "build_invoice", "CreditWallet",
]
