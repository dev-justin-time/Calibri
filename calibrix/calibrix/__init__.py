# SPDX-License-Identifier: MIT
"""Calibrix: model-agnostic parametric modulation + black-box co-optimization.

Repurposes the core logic of two projects in this repository:

- Heretic  (heritic/) : generalized weight-kernel over transformer layers,
    scorer-plugin co-optimization of behavior vs. drift, reproducibility.
- Calibri  (src/)     : per-block output-gain calibration of frozen diffusion
    transformers, CMA-ES black-box search, ~100-parameter efficiency.

Calibrix makes the pattern general: ANY frozen transformer + ANY set of
behavioral scorers + a 6-parameter modulation kernel per component, searched
by a co-objective optimizer. It adds the missing pieces both lack: holdout
validation, Pareto trial selection, cost metering, and drift alarms.
"""

__version__ = "0.1.0"

from .kernel import (
    KernelParams,
    ModulationSpec,
    pack_spec_vector,
    unpack_spec_vector,
    total_param_count,
)
from .scorers import (
    Scorer,
    Score,
    ScorerContext,
    KeywordRate,
    KLDrift,
    LengthDrift,
    EmptyRate,
    DiversityDrop,
    NFETrap,
    seed_prompts,
)
from .engine import SearchEngine, SearchConfig
from .report import build_report, write_report

__all__ = [
    "__version__",
    "KernelParams",
    "ModulationSpec",
    "pack_spec_vector",
    "unpack_spec_vector",
    "total_param_count",
    "Scorer",
    "Score",
    "ScorerContext",
    "KeywordRate",
    "KLDrift",
    "LengthDrift",
    "EmptyRate",
    "DiversityDrop",
    "NFETrap",
    "seed_prompts",
    "SearchEngine",
    "SearchConfig",
    "build_report",
    "write_report",
]
