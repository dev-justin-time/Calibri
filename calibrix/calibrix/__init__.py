# SPDX-License-Identifier: MIT
"""Calibrix: model-agnostic parametric modulation + black-box co-optimization.

Repurposes the core logic of two projects in this repository:

- Heretic  (heritic/) : generalized weight-kernel over transformer layers,
    scorer-plugin co-optimization of behavior vs. drift, reproducibility.
- Calibri  (src/)     : per-block output-gain calibration of frozen diffusion
    transformers, CMA-ES black-box search, ~100-parameter efficiency.

Calibrix makes the pattern general: ANY frozen transformer + ANY panel of
behavioral scorers + a 6-parameter modulation kernel per component, searched
by a co-objective optimizer. It adds the missing pieces both lack: holdout
validation with an overfitting alarm, Pareto trial selection, explicit cost
metering/planning, an offline scripted mode, and a self-contained HTML report
that works from file:// or any free static host.
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
    Prompt,
    seed_prompts,
)
from .adapters import (
    Adapter,
    ScriptedAdapter,
    OpenAICompatAdapter,
    HFInferenceAdapter,
    CalibriFluxAdapter,
)
from .judge import OfflineJudge, RemoteLLMJudge
from .optimizers import make_optimizer
from .metering import Meter, CostModel, plan_budget, quote_job, quote_from_preset, format_quote_text, QuoteConfig, MODEL_PRESETS
from .events import EventLog
from .engine import SearchEngine, SearchConfig, TrialResult
from .report import build_report, write_report

__all__ = [
    "__version__",
    # kernel
    "KernelParams", "ModulationSpec", "pack_spec_vector",
    "unpack_spec_vector", "total_param_count",
    # scorers
    "Scorer", "Score", "ScorerContext", "Prompt", "seed_prompts",
    "KeywordRate", "KLDrift", "LengthDrift", "EmptyRate",
    "DiversityDrop", "NFETrap",
    # adapters
    "Adapter", "ScriptedAdapter", "OpenAICompatAdapter",
    "HFInferenceAdapter", "CalibriFluxAdapter",
    # judge
    "RemoteLLMJudge", "OfflineJudge",
    # optimizers
    "make_optimizer",
    # metering
    "Meter", "CostModel", "plan_budget", "EventLog",
    # quoting
    "quote_job", "quote_from_preset", "format_quote_text", "QuoteConfig", "MODEL_PRESETS",
    # engine
    "SearchEngine", "SearchConfig", "TrialResult",
    # report
    "build_report", "write_report",
]
