# SPDX-License-Identifier: MIT
"""Calibrix Logic Studio: real implementations of the 100-feature taxonomy.

Eight domains, one module each, all clean-room (published equations only,
no runtime code copied), numpy-first, torch-optional, fully offline-testable:

  D1 steering    - modulation primitives (QK gains, RoPE, softmax temp, ...)
  D2 scoring     - scorer panel math (CIELAB dE, SSIM, sharpness, patch-LPIPS)
  D3 optim       - optimizers & Pareto solvers (NSGA-II, DE, halving, GP-UCB)
  D4 proof       - overfit alarms & attestation (k-fold, seals, canaries)
  D5 billing     - budget guards, spot bidding, breakers, gainshare
  D6 runtime     - artifact/export layer (ComfyUI JSON, safetensors metadata)
  D7 governance  - cleanroom scan, zero-leak proof, audit chain, PII
  D8 settlement  - gainshare engine, escrow, royalty splits, SLA credits
"""

from . import steering, scoring, optim, proof, billing, runtime, governance, settlement

__all__ = ["steering", "scoring", "optim", "proof", "billing", "runtime",
           "governance", "settlement"]
