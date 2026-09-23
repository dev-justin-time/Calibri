# SPDX-License-Identifier: MIT
"""Calibrix multi-agent orchestration layer.

Role agents (Planner, Calibrator, Validator, Artificer, Merchant,
ComplaintClerk, Auditor) cooperate over a deterministic in-process message
bus, driving four tool surfaces:

  * offline  — search engine, proof/dossier machinery, heretic-bridge fitter
  * comfy    — ComfyUI workflow export, spec-parse verification, Rust parity
  * online   — marketplace store, orders, fulfillment, license verification
  * watchdog — Doberwatch advise/grade/fact-check/complain

Run the whole pipeline with `python -m calibrix.agents`.
"""

from .bus import Bus, Message
from .base import Agent
from .tools import ToolSurface
from .roles import (Planner, Calibrator, Validator, Artificer, Merchant,
                    ComplaintClerk, Auditor)
from .orchestrator import Orchestrator, run_pipeline

__all__ = [
    "Bus", "Message", "Agent", "ToolSurface",
    "Planner", "Calibrator", "Validator", "Artificer", "Merchant",
    "ComplaintClerk", "Auditor",
    "Orchestrator", "run_pipeline",
]
