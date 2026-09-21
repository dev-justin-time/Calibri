# SPDX-License-Identifier: MIT
"""Role agents for the Calibrix pipeline.

Mission flow (one message round-trip per hop, deterministic):

  Planner    orchestrator.plan   -> mission brief          [calibrate.request]
  Calibrator calibrate.request   -> offline search run     [calibrate.done]
  Validator  calibrate.done      -> overfit gate           [kernel.approved | kernel.rejected]
  Artificer  kernel.approved     -> spec + workflow + fit  [kernel.ready]
  Merchant   kernel.ready        -> listing + sale cycle   [sale.complete]
  Auditor    "*"                 -> hash-chain every event (seals the run)

Each agent owns exactly one concern and communicates only through the bus —
no direct imports of other agents, which keeps every role independently
testable with a stub bus.
"""

from __future__ import annotations

from typing import Tuple

from .base import Agent
from .bus import Bus, Message


class Planner(Agent):
    """Turns a goal into a concrete mission brief (the only policy holder)."""

    role = "planner"
    subscribes: Tuple[str, ...] = ("orchestrator.plan",)

    def handle(self, msg: Message, bus: Bus) -> None:
        goal = msg.payload.get("goal", "calibrate and ship a kernel")
        mission = {
            "goal": goal,
            "model": msg.payload.get("model", "ScriptedAdapter-12L"),
            "n_layers": msg.payload.get("n_layers", 12),
            "n_trials": msg.payload.get("n_trials", 6),
            "popsize": msg.payload.get("popsize", 6),
            "seed": msg.payload.get("seed", 0),
            "checkpoint": msg.payload.get("checkpoint", "calibrix_demo.safetensors"),
            "price_cents": msg.payload.get("price_cents", 4900),
            "buyer_email": msg.payload.get("buyer_email",
                                           "buyer@example.invalid"),
        }
        self.say(bus, "calibrate.request", mission)


class Calibrator(Agent):
    """Offline surface worker: runs the hosted calibration search."""

    role = "calibrator"
    subscribes: Tuple[str, ...] = ("calibrate.request",)

    def handle(self, msg: Message, bus: Bus) -> None:
        summary = self.tools.calibrate(msg.payload)
        self.say(bus, "calibrate.done", {**msg.payload,
                                         "calibration": summary})


class Validator(Agent):
    """Quality gate: rejects overfit kernels BEFORE anything ships."""

    role = "validator"
    subscribes: Tuple[str, ...] = ("calibrate.done",)

    def handle(self, msg: Message, bus: Bus) -> None:
        summary = msg.payload.get("calibration", {})
        if summary.get("best_vector") is None:
            self.say(bus, "kernel.rejected",
                     {"reason": "no kernel found in search",
                      "stage": "validator"})
            return
        check = self.tools.check_overfit(summary)
        if check["flagged"]:
            self.say(bus, "kernel.rejected",
                     {"reason": f"overfit alarm: gap {check.get('gap')}",
                      "stage": "validator"})
            return
        self.say(bus, "kernel.approved",
                 {**msg.payload, "validation": check})


class Artificer(Agent):
    """Comfy-surface worker: spec verification, workflow export, profile fit."""

    role = "artificer"
    subscribes: Tuple[str, ...] = ("kernel.approved",)

    def handle(self, msg: Message, bus: Bus) -> None:
        vector = msg.payload["calibration"]["best_vector"]
        fit = self.tools.fit_profile(vector)
        verify = self.tools.verify_spec(fit["spec_string"],
                                        n_sites=int(msg.payload.get("n_layers", 12)))
        if not verify["channels"]:
            self.say(bus, "kernel.rejected",
                     {"reason": "spec failed to parse", "stage": "artificer"})
            return
        export = self.tools.export_workflow(
            fit["spec_string"], msg.payload.get("checkpoint", "calibrix.safetensors"))
        attest = self.tools.build_attestation({
            "goal": msg.payload.get("goal"),
            "kernel_vector": vector,
            "spec": fit["spec_string"],
            "best_fitness": msg.payload["calibration"]["best_fitness"],
            "parity": verify["max_abs_diff"],
        }, name="kernel_dossier")
        self.say(bus, "kernel.ready", {**msg.payload,
                                       "spec": fit["spec_string"],
                                       "profile_fit": fit,
                                       "verify": verify,
                                       "export": export,
                                       "attestation": attest})


class Merchant(Agent):
    """Online-surface worker: listing, sale cycle, license verification."""

    role = "merchant"
    subscribes: Tuple[str, ...] = ("kernel.ready",)

    def handle(self, msg: Message, bus: Bus) -> None:
        spec = msg.payload["spec"]
        listing = self.tools.list_kernel(
            spec, title=f"Calibrated kernel — {msg.payload.get('model', 'model')}",
            model=msg.payload.get("model", "model"),
            price_cents=int(msg.payload.get("price_cents", 4900)),
            metrics={"best_fitness": msg.payload["calibration"]["best_fitness"]},
        )
        sale = self.tools.sell_kernel(listing["listing_id"],
                                      msg.payload.get("buyer_email",
                                                      "buyer@example.invalid"))
        self.say(bus, "sale.complete", {**msg.payload,
                                        "listing": listing,
                                        "sale": sale})


class Auditor(Agent):
    """Observes ALL traffic and seals the run into the hash chain."""

    role = "auditor"
    subscribes: Tuple[str, ...] = ("*",)

    def handle(self, msg: Message, bus: Bus) -> None:
        # chain the payload, not raw ts/seq, so verification is stable
        self.tools.audit_append(self.role, msg.topic, msg.payload)
