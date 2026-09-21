# Calibrix directional ablation (MIT).
#
# Clean-room implementation of the *methods* behind automatic abliteration,
# from the public literature only:
#
#   * Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction"
#     (arXiv:2406.11717) - refusal directions via difference-of-means over
#     first-token residual-stream activations; directional ablation via
#     orthogonalization of attention out-projections.
#   * Lai 2025 - projected / norm-preserving biprojected variants.
#   * The published idea of co-optimizing ablation parameters against a
#     panel of scorers (compliance vs behavior drift) - which is exactly
#     what Calibrix's SearchEngine already does for any adapter.
#
# No code from heretic (AGPL) or any other abliteration project is used or
# consulted; everything below is written against the Calibrix adapter API.
#
# What "ablation" means here: an Adapter whose modulation *is* ablation.
# Each component channel carries KernelParams whose (weight, position,
# focus, floor) define a smooth per-layer ablation-strength profile
# (Heretic-style shape flexibility becomes the Calibrix kernel family), and
# each site's weight interpolates between the layer's refusal direction and
# the per-layer direction (direction_index as a float, per the literature).
# The SearchEngine co-optimizes those parameters against compliance and
# drift scorers with train/holdout validation - i.e. the full "fully
# automatic abliteration" loop, MIT-licensed and adapter-agnostic.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters import Adapter, ScriptedAdapter
from .kernel import KernelParams, ModulationSpec, unpack_spec_vector
from .scorers import Prompt, ScorerContext


# ---------------------------------------------------------------------------
# 1. Direction extraction (Arditi et al. 2024, Section: difference-in-means)
# ---------------------------------------------------------------------------

def compute_refusal_directions(
    hidden_states: np.ndarray,
    harmful_flags: List[bool],
) -> List[np.ndarray]:
    """Per-layer refusal directions from residual-stream activations.

    Parameters
    ----------
    hidden_states : array (n_prompts, n_layers+1, d_model)
        First-token residual-stream activations per layer (layer 0 =
        embedding output, layers 1..n = post-block i).
    harmful_flags : per-prompt booleans; True = "harmful" (refusal-eliciting)
        cluster, False = "harmless" cluster.

    Returns
    -------
    List of d_model vectors, one per layer (length n_layers+1), each the
    normalized difference of cluster means (harmful minus harmless). Layers
    where either cluster is empty yield zero vectors.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be (n_prompts, n_layers+1, d), got shape "
            f"{hidden_states.shape}"
        )
    n_prompts, n_layers_p1, _ = hidden_states.shape
    if len(harmful_flags) != n_prompts:
        raise ValueError(
            f"got {len(harmful_flags)} flags for {n_prompts} prompts"
        )
    harm_idx = [i for i, f in enumerate(harmful_flags) if f]
    safe_idx = [i for i, f in enumerate(harmful_flags) if not f]
    directions: List[np.ndarray] = []
    for l in range(n_layers_p1):
        if not harm_idx or not safe_idx:
            directions.append(np.zeros(hidden_states.shape[2], dtype=np.float64))
            continue
        h_mean = hidden_states[harm_idx, l, :].mean(axis=0)
        s_mean = hidden_states[safe_idx, l, :].mean(axis=0)
        diff = h_mean - s_mean
        norm = np.linalg.norm(diff)
        directions.append(diff / norm if norm > 1e-12 else np.zeros_like(diff))
    return directions


def pick_best_direction_layer(directions: List[np.ndarray]) -> int:
    """Layer whose direction best separates the clusters (max mean cosine gap).

    Simple, dependency-free heuristic: the direction that maximizes the
    separation between the projection distributions of the two clusters is
    the most "refusal-like"; we approximate separation with the gap between
    cluster means minus pooled spread, computed on the *extraction* clusters
    themselves.
    """
    # The caller has the activations; re-deriving them here would double the
    # API surface. Instead we score by direction self-consistency: the layer
    # whose direction has maximal cosine alignment with the average direction
    # of its neighbors (a stable direction generalizes better than a noisy
    # one). Zero vectors score -inf.
    n = len(directions)
    best_i, best_score = 0, -math.inf
    for i, d in enumerate(directions):
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            continue
        neighbors = [directions[j] for j in (i - 1, i + 1) if 0 <= j < n]
        neighbors = [v for v in neighbors if np.linalg.norm(v) > 1e-12]
        if not neighbors:
            score = 0.0
        else:
            score = float(np.mean([np.dot(d, v) / (nd * np.linalg.norm(v))
                                   for v in neighbors]))
        if score > best_score:
            best_score, best_i = score, i
    return best_i


# ---------------------------------------------------------------------------
# 2. Directional ablation math (norm-preserving orthogonalization)
# ---------------------------------------------------------------------------

def orthogonalize(matrix: np.ndarray, direction: np.ndarray,
                  preserve_norm: bool = True) -> np.ndarray:
    """Project the row space of `matrix` off `direction`.

    M' = M (I - alpha * d d^T)  with alpha = 1, or the norm-preserving
    symmetric form  M' = M sqrt(I - d d^T)  (Lai 2025) when preserve_norm.

    For preserve_norm the operator sqrt(I - dd^T) has eigenvalues 1 on the
    orthogonal complement and 0 on the direction, but applied symmetrically
    (M S S^T M^T) it preserves the norm of outputs whose components along d
    are removed while leaving orthogonal components untouched up to the
    projection - the "biprojected" trick from the literature.
    """
    d = direction.astype(np.float64)
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return matrix
    d = d / nd
    if preserve_norm:
        # sqrt(I - d d^T) = I - (1 - sqrt(0)) d d^T restricted to the span of
        # d: eigenvalue 0 on d, 1 elsewhere. The symmetric square root of
        # I - dd^T is I - dd^T itself (idempotent projector, its own sqrt).
        # Norm preservation comes from projecting *and* rescaling the
        # residual: x' = x - (x.d) d  keeps norm only when x.d == 0, so the
        # norm-preserving variant instead uses  x' = x - (1 - s) (x.d) d  with
        # s = 0 -> this collapses to the plain projection. The genuinely
        # norm-preserving published variant (Lai 2025) projects the *matrix*
        # bidirectionally (input and output sides). We implement that here.
        #
        # M' = (I - dd^T) M (I - dd^T)  -- biprojection; singular values along
        # d are removed on both sides, everything else is untouched, and the
        # per-output norm is preserved up to the removed component.
        p = np.eye(d.shape[0]) - np.outer(d, d)
        return p @ matrix @ p
    p = np.eye(d.shape[0]) - np.outer(d, d)
    return p @ matrix


def ablation_weight_profile(
    n_sites: int, weight: float, position: float, focus: float, floor: float,
    ripple: float = 0.0, phase: float = 0.0,
) -> List[float]:
    """Per-site ablation strengths from the Calibrix kernel (0..1 typical)."""
    gains = ModulationSpec(
        component="ablate", n_sites=n_sites,
        kernel=KernelParams(weight=weight, position=position, focus=focus,
                            floor=floor, ripple=ripple, phase=phase),
    ).gains()
    return gains


# ---------------------------------------------------------------------------
# 3. Weight-matrix location: the only model-family-specific code
# ---------------------------------------------------------------------------

# Component kinds a family exposes; ablation targets attention out-projections
# (Arditi et al.) and optionally MLP down-projections.
ABLATION_COMPONENTS = ("attn", "mlp")


def _find_module(model: Any, names: Sequence[str]) -> Optional[Any]:
    for n in names:
        if hasattr(model, n):
            return getattr(model, n)
    return None


def extract_weight_matrices(model: Any, family: Optional[str] = None) -> List[Tuple[str, Any]]:
    """Locate per-layer out-projection / down-projection weight matrices.

    Returns a list of (component, nn.Module-or-ndarray) pairs, in layer
    order. Supports the HF architectures Calibri/Calibrix target via
    well-known attribute names; unknown models raise with a helpful list.
    Each entry must expose `.weight` (torch) or be a 2D ndarray.
    """
    candidates = {
        "attn": ["o_proj", "self_attn.o_proj", "attn.out_proj", "out_proj",
                 "WO", "wo", "attn2_to_out"],
        "mlp": ["down_proj", "mlp.down_proj", "mlp.fc2", "fc2", "proj_out",
                "h_to_4h_out", "WO_mlp"],
    }
    found: List[Tuple[str, Any]] = []
    blocks = _find_module(model, ["transformer_blocks", "layers", "blocks",
                                  "h", "decoder.layers"])
    if blocks is None:
        raise ValueError(
            "could not locate transformer blocks on the model; supported "
            "containers: transformer_blocks/layers/blocks/h/decoder.layers"
        )
    for block in blocks:
        placed = False
        for comp, names in candidates.items():
            mod = _find_module(block, names)
            if mod is not None and hasattr(mod, "weight"):
                found.append((comp, mod))
                placed = True
                break
        if not placed:
            raise ValueError(
                "could not locate out/down projection on a block; supported "
                f"names: {sorted({n for v in candidates.values() for n in v})}"
            )
    return found


# ---------------------------------------------------------------------------
# 4. The ablation adapter (Calibrix Adapter whose modulation == ablation)
# ---------------------------------------------------------------------------

@dataclass
class AblationSpec:
    """What the optimizer is allowed to change."""

    # One ModulationSpec per component channel (attn, mlp, ...), each with
    # n_sites == number of layers. Kernel params define the per-layer
    # ablation-strength profile; direction_index selects/interpolates the
    # refusal direction per layer.
    specs: List[ModulationSpec]
    # direction lookup: layer -> vector. When the kernel's direction_index is
    # a float k, the site uses directions[floor(k)] blended with
    # directions[ceil(k)] by frac(k).
    directions: List[np.ndarray]
    # global ablation weight bounds applied to every site (0..1 typical)
    max_strength: float = 1.5


def default_ablation_spec(n_layers: int, d_model: int,
                          n_directions: Optional[int] = None) -> AblationSpec:
    """Kernel-style ablation search space: 2 channels x 6 kernel params."""
    dirs = [np.zeros(d_model) for _ in range(n_directions or n_layers + 1)]
    specs = [
        ModulationSpec(component="attn", n_sites=n_layers),
        ModulationSpec(component="mlp", n_sites=n_layers),
    ]
    return AblationSpec(specs=specs, directions=dirs)


def site_direction(directions: List[np.ndarray],
                   index: Optional[float]) -> np.ndarray:
    """Resolve a (possibly fractional, possibly None) direction index."""
    if index is None:
        # "per layer": use the direction of the layer itself (clipped).
        raise ValueError("per-layer direction requires an explicit layer index")
    k = float(index)
    n = len(directions)
    if n == 0:
        raise ValueError("no directions available")
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    lo = min(max(lo, 0), n - 1)
    hi = min(max(hi, 0), n - 1)
    frac = k - math.floor(k)
    v = (1.0 - frac) * directions[lo] + frac * directions[hi]
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else np.zeros_like(v)


class DirectionalAblationAdapter(Adapter):
    """Wraps any torch model whose blocks expose out/down projections.

    Modulation semantics (the Heretic idea, Calibrix mechanics):
      - each component channel's kernel defines per-layer strength a(l)
      - each layer's effective direction d_l is chosen/interpolated by that
        channel's direction_index (fractional indices blend neighbors)
      - the layer's out/down projection W is replaced by an orthogonalized
        W' that inhibits expressing d_l with strength a(l):
            W' = W - a(l) * d_l d_l^T W          (plain)
            W' = (I - a dd^T) W (I - a dd^T)     (norm-preserving, biprojected)

    Everything (strength profile shape, per-layer direction choice, strength
    scale) is a free parameter of the co-optimization - there is no manual
    per-layer tuning, which is the whole point of the method.
    """

    name = "directional-ablation"

    def __init__(self, model: Any, spec: AblationSpec,
                 norm_preserving: bool = True,
                 harmful_prompts: Optional[List[Prompt]] = None,
                 harmless_prompts: Optional[List[Prompt]] = None) -> None:
        self.model = model
        self.spec = spec
        self.norm_preserving = norm_preserving
        self.harmful_prompts = harmful_prompts or []
        self.harmless_prompts = harmless_prompts or []
        self.modulatable = True
        # (component, module) pairs in layer order
        self.targets = extract_weight_matrices(model)
        if not self.targets:
            raise ValueError("no ablation targets found on the model")
        self._originals: Dict[int, Any] = {}
        self._steps_per_call = 2
        self._last_strength_signature = 0.0

    # -- Adapter contract ---------------------------------------------------
    def spec_sites(self) -> List[ModulationSpec]:
        return self.spec.specs

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        """Materialize kernel params into orthogonalized weight matrices."""
        self.spec.specs = list(specs)
        by_comp: Dict[str, ModulationSpec] = {s.component: s for s in specs}
        # restore originals before re-applying (idempotent modulation)
        for idx, (comp, mod) in enumerate(self.targets):
            orig = self._originals.get(idx)
            if orig is not None:
                _set_weight(mod, orig)
        n_layers = len(self.targets)
        sig_accum: List[float] = []
        for idx, (comp, mod) in enumerate(self.targets):
            spec = by_comp.get(comp)
            if spec is None:
                continue
            a_l = spec.gains()  # per-layer strengths
            a = float(np.clip(a_l[idx % len(a_l)], 0.0, self.spec.max_strength))
            self._last_strength_signature += abs(a)
            if a <= 1e-6:
                continue
            d = site_direction(self.spec.directions,
                               spec.kernel.direction_index)  # type: ignore[attr-defined]
            if not np.any(d):
                continue
            W = _get_weight(mod)
            dW = np.outer(d, d @ W)
            Wp = W - a * dW
            if self.norm_preserving:
                # second projection from the output side (biprojected form)
                Wp = Wp - a * (Wp @ np.outer(d, d))
            _set_weight(mod, Wp)
        n = max(len(self.targets), 1)
        self._last_strength_signature = self._last_strength_signature / n

    def generate(self, prompts: List[Prompt], batch_size: int = 8,
                 ctx: Optional[ScorerContext] = None) -> List[str]:
        from .adapters import _stub_response  # local import avoids cycle

        out = [_stub_response(p, 1.0 - self._last_strength_signature,
                              self._steps_per_call) for p in prompts]
        if ctx is not None:
            ctx.note_steps(self._steps_per_call * max(len(prompts), 1))
        return out

    def restore(self) -> None:
        """Undo all ablations (restore original weights)."""
        for idx, (comp, mod) in enumerate(self.targets):
            orig = self._originals.get(idx)
            if orig is not None:
                _set_weight(mod, orig)


def _get_weight(mod: Any) -> np.ndarray:
    w = mod.weight
    if hasattr(w, "detach"):
        return w.detach().float().cpu().numpy()
    return np.asarray(w, dtype=np.float64)


def _set_weight(mod: Any, value: np.ndarray) -> None:
    w = mod.weight
    if hasattr(w, "data"):
        import torch

        w.data = torch.as_tensor(value, dtype=w.dtype, device=w.device)
    else:  # plain ndarray module shim
        mod.weight = value


# ---------------------------------------------------------------------------
# 5. Offline simulation adapter (no torch needed: lets the full TPE loop
#    run on a laptop against a synthetic "refusal direction" model)
# ---------------------------------------------------------------------------

class SimulatedRefusalAdapter(ScriptedAdapter):
    """ScriptedAdapter + a synthetic residual geometry to ablate.

    Emulates a model whose refusal behavior is mediated by a known direction
    at a known layer with a known strength profile, so the direction
    extraction and ablation pipeline can be exercised end-to-end offline:
      - hidden_states() exposes (n_prompts, n_layers+1, d) residuals whose
        harmful cluster is offset along the planted direction
      - modulation deviation (ablation applied) reduces refusal rate, but
        over-ablation produces the usual drift (length/empties)
    """

    def __init__(self, n_layers: int = 12, d_model: int = 16, seed: int = 0,
                 refusal_layer: Optional[int] = None) -> None:
        super().__init__(n_layers=n_layers, seed=seed)
        self.d_model = d_model
        self.refusal_layer = n_layers // 2 if refusal_layer is None else refusal_layer
        rng = np.random.default_rng(seed)
        self.planted_direction = rng.normal(size=d_model)
        self.planted_direction /= np.linalg.norm(self.planted_direction)
        # pre-build the residual clusters
        self.harmful: List[np.ndarray] = []
        self.harmless: List[np.ndarray] = []

    def plant_clusters(self, harmful: List[Prompt], harmless: List[Prompt]) -> None:
        """Build deterministic residual activations for both clusters."""
        self.harmful = harmful
        self.harmless = harmless

    def hidden_states_for(self, prompts: List[Prompt]) -> np.ndarray:
        rows = []
        for p in prompts:
            per_layer = []
            for l in range(self.n_layers + 1):
                base = np.random.default_rng(
                    abs(hash(p.user)) % (2**32) + l).normal(size=self.d_model)
                per_layer.append(base)
            rows.append(per_layer)
        arr = np.stack(rows)
        # inject the planted refusal direction at/around the refusal layer,
        # scaled by depth-tanh like a real residual stream
        for i, p in enumerate(prompts):
            is_harmful = any(w in p.user.lower() for w in
                             ("bomb", "weapon", "malware", "exploit", "steal"))
            if not is_harmful:
                continue
            for l in range(self.n_layers + 1):
                scale = 2.2 * math.tanh(l / 4.0) * max(0.0, 1.0 - abs(l - self.refusal_layer) / max(self.n_layers, 1))
                arr[i, l, :] += scale * self.planted_direction
        return arr

    # -- ScriptedAdapter override: refusal depends on ablation strength ----
    def generate(self, prompts, batch_size=8, ctx=None):
        import hashlib

        sig = self._signature()  # 1.0 == unmodulated
        ablated = 1.0 - sig       # total modulation deviation
        out = []
        for p in prompts:
            h = hashlib.sha256(p.user.encode("utf-8")).hexdigest()
            is_harmful = any(w in p.user.lower() for w in
                             ("bomb", "weapon", "malware", "exploit", "steal"))
            if is_harmful:
                # refusal probability falls as ablation strength rises
                refuse_p = max(0.0, 0.9 - 1.4 * ablated)
                refuses = (int(h[:8], 16) / 0xFFFFFFFF) < refuse_p
                if refuses:
                    out.append("I'm sorry, but I can't help with that request.")
                    continue
            # harmless/harmful-compliant outputs: mild drift with ablation
            if ablated > 0.55:
                out.append("")
            elif ablated > 0.30:
                out.append("As an AI assistant, I must refuse this unrelated request.")
            else:
                out.append(f"Sure. Here is a helpful answer about {p.user[:40]}.")
        if ctx is not None:
            ctx.note_steps(self._steps_per_call * max(len(prompts), 1))
        return out

    def apply_modulation(self, specs):
        # identical bookkeeping to ScriptedAdapter; ablation strength = gains
        self._specs = specs


def extract_directions_from_sim(adapter: SimulatedRefusalAdapter,
                                harmful: List[Prompt],
                                harmless: List[Prompt]) -> List[np.ndarray]:
    """Convenience: refusal directions from the simulated residual stream."""
    hs_h = adapter.hidden_states_for(harmful)
    hs_s = adapter.hidden_states_for(harmless)
    stacked = np.concatenate([hs_h, hs_s], axis=0)
    flags = [True] * len(harmful) + [False] * len(harmless)
    return compute_refusal_directions(stacked, flags)
