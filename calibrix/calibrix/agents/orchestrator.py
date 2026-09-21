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
from .roles import Artificer, Auditor, Calibrator, Merchant, Planner, Validator
from .tools import ToolSurface


class Orchestrator:
    """Wires bus + tools + the six roles; runs one goal to completion."""

    def __init__(self, work_dir: str = "agent_workspace",
                 secret: str | None = None) -> None:
        self.bus = Bus()
        self.tools = ToolSurface(work_dir, secret)
        self.planner = Planner(self.bus, self.tools)
        self.calibrator = Calibrator(self.bus, self.tools)
        self.validator = Validator(self.bus, self.tools)
        self.artificer = Artificer(self.bus, self.tools)
        self.merchant = Merchant(self.bus, self.tools)
        self.auditor = Auditor(self.bus, self.tools)

    def run(self, goal: str = "calibrate and ship a kernel",
            **overrides) -> Dict[str, Any]:
        t0 = time.time()
        self.bus.publish("orchestrator.plan", {"goal": goal, **overrides},
                         sender="orchestrator")
        rounds = self.bus.drain()
        outcome = self._collect()
        outcome["rounds"] = rounds
        outcome["audit"] = self.tools.audit_verify()
        outcome["audit_entries"] = len(self.tools.chain)
        outcome["messages"] = len(self.bus.log)
        outcome["elapsed_seconds"] = round(time.time() - t0, 3)
        return outcome

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


def run_pipeline(work_dir: str = "agent_workspace", **overrides) -> Dict[str, Any]:
    orch = Orchestrator(work_dir=work_dir)
    return orch.run(**overrides)


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
