# Calibrix — generalized modulation kernel.
#
# Repurposed logic:
#   * Heretic controls *where* an intervention acts via a smooth weight kernel
#     (max_weight / max_weight_position / min_weight / min_weight_distance)
#     over transformer layers, and optimizes those 4 numbers per component.
#   * Calibri controls *what* the intervention is: per-block output gains
#     (gate scales) applied to attention / MLP sub-outputs, optimized as a
#     ~100-dimensional vector by CMA-ES.
#
# Calibrix unifies both: the intervention is "multiply a component's
# contribution by g(l)" where g(l) is a 3-parameter piecewise-linear kernel:
#
#     g(l) = w_max - (w_max - w_min) * clip(|l - p| / d, 0, 1)
#
# plus an optional cosine-modulated ripple controlled by `ripple` and a
# phase free per component. This collapses Heretic's 4-parameter kernel and
# Calibri's per-layer free gains into one parametric family:
#
#   - ripple = 0, d = >= max layers  -> Heretic's smooth kernel (4 params)
#   - per-layer phases + ripple      -> approximates Calibri's free per-layer
#     gains with 10-100x fewer parameters, so search stays sample-efficient.
#
# Kernels are pure numpy and dependency-free.

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class KernelParams:
    """Parameters of the modulation kernel for ONE component.

    Attributes:
        weight:      global gain scale multiplied into the whole kernel.
        position:    layer index of the kernel peak (float, interpolable).
        focus:       distance (in layers) over which the kernel decays from
                     `weight` down to `floor`. Heretic calls this
                     min_weight_distance; large focus == "all layers".
        floor:       asymptotic gain away from the peak (Heretic's min_weight).
        ripple:      amplitude of a cosine ripple added on top of the linear
                     decay. 0 disables it (Heretic-equivalent).
        phase:       phase of the ripple in cycles. Free per component; lets
                     the optimizer place constructive/destructive interference
                     at specific depths (new research capability).
    """

    weight: float = 1.0
    position: float = 0.5
    focus: float = 1e6
    floor: float = 1.0
    ripple: float = 0.0
    phase: float = 0.0

    def validate(self) -> None:
        if not (math.isfinite(self.weight) and math.isfinite(self.position)
                and math.isfinite(self.focus) and math.isfinite(self.floor)
                and math.isfinite(self.ripple) and math.isfinite(self.phase)):
            raise ValueError(f"kernel params must be finite: {self!r}")
        if self.focus <= 0:
            raise ValueError("focus must be > 0")
        if self.weight < 0:
            raise ValueError("weight must be >= 0 (negative gains are undefined)")
        if self.floor < 0:
            raise ValueError("floor must be >= 0")

    @staticmethod
    def from_heretic(
        max_weight: float,
        max_weight_position: float,
        min_weight: float,
        min_weight_distance: float,
        n_sites: int = 1,
    ) -> "KernelParams":
        """Exact conversion from Heretic's AbliterationParameters.

        Heretic's position/distance are absolute layer indices; Calibrix
        stores position normalized to [0, 1] so specs are model-agnostic.
        """
        last = max(n_sites - 1, 1)
        return KernelParams(
            weight=max_weight,
            position=max_weight_position / last,
            focus=max(min_weight_distance, 1e-6) / last,
            floor=min_weight,
            ripple=0.0,
            phase=0.0,
        )

    def to_heretic(self) -> Dict[str, float]:
        """Round-trip back to Heretic parameter names (ripple/phase lost)."""
        return {
            "max_weight": self.weight,
            "max_weight_position": self.position,
            "min_weight": self.floor,
            "min_weight_distance": self.focus,
        }


@dataclass
class ModulationSpec:
    """Declarative description of one modulation channel.

    A channel is (component kind, target module indices, kernel). Adapters
    (FLUX / LLM / audio / any transformer) emit a list of these; the engine
    only manipulates the kernel parameters, adapters do the tensor work.
    """

    component: str                     # e.g. "attn", "mlp", "o_proj", "down_proj"
    n_sites: int                       # number of layers/blocks this channel spans
    kernel: KernelParams = field(default_factory=KernelParams)
    # optional explicit per-site gains (Calibri-style). When empty, gains are
    # generated from the kernel. Mixing both is allowed: kernel * explicit.
    explicit: Optional[List[float]] = None
    # direction_index for direction-based interventions (Heretic-style
    # difference-of-means direction selection). None = use site's own direction.
    direction_index: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_sites <= 0:
            raise ValueError(f"n_sites must be > 0, got {self.n_sites}")
        if self.explicit is not None and len(self.explicit) != self.n_sites:
            raise ValueError(
                f"explicit gains length {len(self.explicit)} != n_sites {self.n_sites}"
            )

    # ------------------------------------------------------------------
    def gains(self) -> List[float]:
        """Materialize per-site gains: kernel(l) * explicit(l)."""
        self.kernel.validate()
        k = self.kernel
        last = max(self.n_sites - 1, 1)
        pos = k.position * last  # position is normalized 0..1
        out = []
        for l in range(self.n_sites):
            distance = abs(l - pos)
            f = k.focus * last  # focus normalized -> absolute layer distance
            decay = min(distance / f, 1.0) if f > 0 else 1.0
            base = k.weight - (k.weight - k.floor) * decay
            if k.ripple != 0.0:
                base += k.ripple * math.cos(
                    2.0 * math.pi * (l / max(self.n_sites, 1))
                    + 2.0 * math.pi * k.phase
                )
            g = base
            if self.explicit is not None:
                g *= self.explicit[l]
            out.append(g)
        return out

    # ------------------------------------------------------------------
    def n_params(self) -> int:
        """Free parameter count of this channel (6 kernel params)."""
        return 6

    def to_vector(self) -> List[float]:
        k = self.kernel
        return [k.weight, k.position, k.focus, k.floor, k.ripple, k.phase]

    def from_vector(self, vec: List[float]) -> None:
        if len(vec) != 6:
            raise ValueError(f"expected 6 kernel params, got {len(vec)}")
        self.kernel = KernelParams(*[float(v) for v in vec])

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gains"] = self.gains()
        return d


def pack_spec_vector(specs: List[ModulationSpec]) -> List[float]:
    """Concatenate all channel kernel vectors into one flat vector."""
    vec: List[float] = []
    for s in specs:
        vec.extend(s.to_vector())
    return vec


def unpack_spec_vector(specs: List[ModulationSpec], vec: List[float]) -> None:
    """Distribute a flat vector into the specs' kernels (in place)."""
    need = sum(s.n_params() for s in specs)
    if len(vec) != need:
        raise ValueError(f"vector length {len(vec)} != total kernel params {need}")
    i = 0
    for s in specs:
        s.from_vector(vec[i:i + 6])
        i += 6


def total_param_count(specs: List[ModulationSpec]) -> int:
    return sum(s.n_params() for s in specs)
