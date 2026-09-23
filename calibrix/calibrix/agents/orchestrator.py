# SPDX-License-Identifier: MIT
"""Orchestrator: assembles the agent team and runs one mission.

The mission completes when the pipeline reaches a fixed point (no pending
messages) or the bus round cap trips — both are reported honestly in the
summary. The Auditor's hash chain seals the whole run.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

from .bus import Bus
from .roles import (Artificer, Auditor, Calibrator, ComplaintClerk, Merchant,
                    Planner, Validator)
from .tools import ToolSurface


class Orchestrator:
    """Wires bus + tools + the seven roles; runs one goal to completion."""

    def __init__(self, work_dir: str = "agent_workspace",
                 secret: str | None = None) -> None:
        self.bus = Bus()
        self.tools = ToolSurface(work_dir, secret)
        self.planner = Planner(self.bus, self.tools)
        self.calibrator = Calibrator(self.bus, self.tools)
        self.validator = Validator(self.bus, self.tools)
        self.artificer = Artificer(self.bus, self.tools)
        self.merchant = Merchant(self.bus, self.tools)
        self.clerk = ComplaintClerk(self.bus, self.tools)
        self.auditor = Auditor(self.bus, self.tools)

    def run(self, goal: str = "calibrate and ship a kernel",
            complaint: Dict[str, Any] | None = None,
            **overrides) -> Dict[str, Any]:
        """Run the kernel mission, then (optionally) one watchdog case.

        ``complaint`` is a ``complaint.request`` payload: the clerk advises
        first and grades the response if one is supplied. Both legs run on
        the same bus, so the Auditor seals them into one chain.
        """
        t0 = time.time()
        self.bus.publish("orchestrator.plan", {"goal": goal, **overrides},
                         sender="orchestrator")
        rounds = self.bus.drain()
        if complaint:
            self.bus.publish("complaint.request", dict(complaint),
                             sender="orchestrator")
            rounds += self.bus.drain()
        outcome = self._collect()
        outcome["rounds"] = rounds
        outcome["complaint_case"] = self._complaint_outcome()
        # Checked, not asserted: every routed message must appear in the chain.
        outcome["bus_audit"] = self.tools.audit_bus(self.bus.log)
        outcome["audit"] = self.tools.audit_verify()
        outcome["audit_entries"] = len(self.tools.chain)
        outcome["messages"] = len(self.bus.log)
        outcome["elapsed_seconds"] = round(time.time() - t0, 3)
        return outcome

    def _complaint_outcome(self) -> Dict[str, Any] | None:
        """Summarize the watchdog case traffic, if a case was opened."""
        advice = [m for m in self.bus.log if m.topic == "complaint.advice"]
        pending = [m for m in self.bus.log if m.topic == "complaint.pending"]
        resolved = [m for m in self.bus.log if m.topic == "complaint.resolved"]
        escalated = [m for m in self.bus.log if m.topic == "complaint.escalated"]
        rejected = [m for m in self.bus.log if m.topic == "complaint.rejected"]
        if not (advice or pending or resolved or rejected):
            return None
        last = resolved[-1].payload if resolved else None
        if resolved:
            status = "RESOLVED"
        elif rejected:
            status = "REJECTED"
        elif pending:
            status = "AWAITING-INFORMATION"
        else:
            status = "ADVISED"
        return {
            "cases": len(advice) + len(rejected),
            "advised": len(advice),
            "pending": len(pending),
            "resolved": len(resolved),
            "escalated": len(escalated),
            "rejected": len(rejected),
            "status": status,
            "last": {
                "case_id": last["case_id"], "verdict": last["verdict"],
                "score": last["score"], "complaint": last["complaint"],
                "refund_plan": last["refund_plan"],
                "escalation_path": last["escalation_path"],
            } if last else None,
            "rejections": [m.payload for m in rejected],
        }

    def _collect(self) -> Dict[str, Any]:
        """Extract the mission outcome from bus traffic (deterministic)."""
        rejected = [m for m in self.bus.log if m.topic == "kernel.rejected"]
        ready = [m for m in self.bus.log if m.topic == "kernel.ready"]
        sales = [m for m in self.bus.log if m.topic == "sale.complete"]

        if sales:
            p = sales[-1].payload
            return {"status": "SOLD", "spec": p["spec"],
                    "listing": p["listing"], "sale": p["sale"],
                    "verify": p.get("verify"), "export": p.get("export"),
                    "best_fitness": p.get("calibration", {}).get("best_fitness"),
                    "trials": (p.get("calibration") or {}).get("trials"),
                    "profile_fit_rmse": (p.get("profile_fit") or {}).get("rmse")}
        if ready:
            p = ready[-1].payload
            return {"status": "READY-NOT-SOLD", "spec": p["spec"],
                    "verify": p.get("verify"), "export": p.get("export")}
        if rejected:
            return {"status": "REJECTED", "reason": rejected[-1].payload.get("reason")}
        return {"status": "NO-OUTCOME",
                "reason": "pipeline produced no terminal message"}


def run_pipeline(work_dir: str = "agent_workspace",
                 complaint: Dict[str, Any] | None = None,
                 **overrides) -> Dict[str, Any]:
    orch = Orchestrator(work_dir=work_dir)
    return orch.run(complaint=complaint, **overrides)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m calibrix.agents",
        description="Run the Calibrix multi-agent pipeline end-to-end.")
    ap.add_argument("--goal", default="calibrate and ship a kernel")
    ap.add_argument("--work-dir", default="agent_workspace")
    ap.add_argument("--n-trials", type=int, default=6)
    ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--price-cents", type=int, default=4900)
    args = ap.parse_args(argv)

    summary = run_pipeline(work_dir=args.work_dir, goal=args.goal,
                           n_trials=args.n_trials, popsize=args.popsize,
                           seed=args.seed, price_cents=args.price_cents)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("status") in ("SOLD", "READY-NOT-SOLD") else 1


if __name__ == "__main__":
    raise SystemExit(main())
