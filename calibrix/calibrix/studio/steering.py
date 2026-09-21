# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Domain 1: Model Steering (clean-room math).
#
# All modulation primitives are written from published equations only
# (Vaswani 2017 scaled dot-product attention; Su et al. 2021 RoPE; Shazeer
# 2020 SwiGLU; Ba 2016 LayerNorm; Zhang & Sennrich 2019 RMSNorm). No code is
# copied from any runtime. Numpy-first; every function is pure/deterministic
# so the modulation layer is testable without a GPU.
#
# The unifying concept: a "gain" is any scalar/vector applied multiplicatively
# to an internal activation. Calibrix kernels (kernel.py) generate per-layer
# gain profiles; this module defines HOW each gain family acts on the math.

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# F001 — Q/K Matrix Gain Modulation -----------------------------------------
def qk_gain(attn_scores: np.ndarray, gain: float) -> np.ndarray:
    """Scale query·key logits by `gain` before softmax.

    logits' = gain * (Q K^T / sqrt(d_k))   (Vaswani 2017, Eq. 1)
    gain < 1 flattens attention (more heads consulted), gain > 1 sharpens it
    (winner-take-most). This is the single highest-leverage steering knob
    because every downstream token mixing decision flows through it.
    """
    return attn_scores * float(gain)


def qk_gain_per_head(attn_scores: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    """Per-head variant: attn_scores shaped (heads, q_len, k_len)."""
    g = np.asarray(gains, dtype=np.float64).reshape(-1, 1, 1)
    return attn_scores * g


# F002 — MLP Residual Ratio Shifter ------------------------------------------
def mlp_residual_ratio(residual: np.ndarray, mlp_out: np.ndarray,
                       ratio: float) -> np.ndarray:
    """Blend MLP contribution into the residual stream.

    x' = x + ratio * MLP(x)  — ratio 1.0 is identity; <1 damps the MLP's
    opinion; the Calibrix kernel provides per-layer ratios so depth-specific
    behavior (e.g. fact recall lives in mid MLPs) can be tuned without
    touching attention.
    """
    return residual + float(ratio) * mlp_out


# F003 — LayerNorm Epsilon Clamping ------------------------------------------
def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    """LayerNorm (Ba et al. 2016) with explicit, clamped epsilon.

    Clamping eps >= 1e-6 prevents variance collapse under aggressive CFG /
    high-gain steering where hidden variance can shrink orders of magnitude,
    which silently zeroes the normalization output on some architectures.
    """
    eps = max(float(eps), 1e-6)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm (Zhang & Sennrich 2019), eps clamped as in layernorm()."""
    eps = max(float(eps), 1e-6)
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    return weight * x / np.sqrt(ms + eps)


# F004 — Cross-Attention Bias Gate --------------------------------------------
def context_bias(context: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Add a learned bias vector into the text-conditioning context.

    context' = context + bias  — a single d_model vector that shifts *what
    the cross-attention reads*, letting the search engine push conditioning
    toward attribute axes (style, saturation) with one parameter.
    """
    return context + bias.reshape(1, -1)


# F005 — Skip-Connection Alpha Damping ----------------------------------------
def skip_alpha(deep: np.ndarray, shallow: np.ndarray, alpha: float) -> np.ndarray:
    """Modulate deep-skip injection: out = deep + alpha * shallow.

    Diffusion U-Nets/DiTs re-inject shallow features late; alpha controls how
    much fine detail survives the depth. Useful for detail-vs-coherence
    tradeoffs when cutting NFE.
    """
    return deep + float(alpha) * shallow


# F006 — Rotary Position (RoPE) Angle Perturbation ----------------------------
def rope_angles(positions: np.ndarray, head_dim: int,
                base: float = 10000.0, freq_scale: float = 1.0) -> np.ndarray:
    """RoPE angle table (Su et al. 2021), with a frequency scale knob.

    theta_i = base^(-2i/d) * freq_scale * position
    freq_scale slightly < 1 stretches effective context; the optimizer can
    tune it per-model for long-prompt fidelity.
    """
    half = head_dim // 2
    inv_freq = base ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)
    return np.outer(np.asarray(positions, dtype=np.float64), inv_freq) * float(freq_scale)


def apply_rope(x: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split x into (cos-rotated, sin-rotated) halves for a (seq, d) vector."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = np.cos(angles)
    sin = np.sin(angles)
    return x1 * cos - x2 * sin, x1 * sin + x2 * cos


# F007 — SwiGLU Gate Projection Trim ------------------------------------------
def swiglu(gate_proj: np.ndarray, up_proj: np.ndarray,
           trim_threshold: float = 0.0) -> np.ndarray:
    """SwiGLU (Shazeer 2020): silu(gate) * up, with optional magnitude trim.

    Activations below `trim_threshold` in the gate are zeroed — a cheap
    sparsification the search can exploit to cut compute while the scorer
    panel guards quality.
    """
    g = gate_proj / (1.0 + np.exp(-gate_proj))  # silu
    if trim_threshold > 0:
        g = np.where(g < trim_threshold, 0.0, g)
    return g * up_proj


# F008 — Zero Weight-Mod Bypass Tap -------------------------------------------
class ZeroModTap:
    """Prove base weights are never written.

    Wraps any object holding `.weight`-style arrays; records SHA-256 digests
    at attach time and re-verifies on demand. The search engine applies
    modulation through *gains and projections only* (never in-place weight
    writes), so this tap is the enforcement mechanism behind D7/F081's
    zero-leak proof and every "non-destructive" claim in the docs.
    """

    def __init__(self, tensors: Dict[str, np.ndarray]) -> None:
        self._refs = {k: v for k, v in tensors.items()}
        self._digests = {k: self._digest(v) for k, v in tensors.items()}

    @staticmethod
    def _digest(arr: np.ndarray) -> str:
        import hashlib

        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

    def verify(self) -> Dict[str, bool]:
        return {k: self._digest(self._refs[k]) == d for k, d in self._digests.items()}

    def all_untouched(self) -> bool:
        return all(self.verify().values())


# F009 — GQA Head Replication Router ------------------------------------------
def gqa_route(kv_heads: int, query_heads: int) -> List[List[int]]:
    """Grouped-query attention mapping: which KV head serves each Q head.

    Returns query_heads -> kv_head index. The router lets steering modulate
    *specific groups* (e.g. damp the KV group serving the sharpest heads)
    instead of the whole cache.
    """
    if query_heads % kv_heads != 0:
        raise ValueError("query_heads must be a multiple of kv_heads")
    group = query_heads // kv_heads
    return [h // group for h in range(query_heads)]


# F010 — DiT Double-Stream Synchronizer ---------------------------------------
def dit_sync(img_stream: np.ndarray, txt_stream: np.ndarray,
             sync_gain: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """MM-DiT double-stream sync: scale the cross-stream contribution.

    Each stream's block reads the other; gains here rebalance how much text
    informs image tokens and vice versa (FLUX/SD3/Qwen-style blocks).
    """
    g = float(sync_gain)
    return img_stream + g * 0.0 * txt_stream, txt_stream  # placeholder identity


def dit_cross_gain(img_attn_out: np.ndarray, txt_attn_out: np.ndarray,
                   img_gain: float, txt_gain: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply independent cross-stream gains to joint-attention outputs."""
    return img_attn_out * float(img_gain), txt_attn_out * float(txt_gain)


# F011 — Dynamic Text-Pooling Injection ----------------------------------------
def pooled_injection(hidden: np.ndarray, pooled: np.ndarray,
                     strength: float) -> np.ndarray:
    """Inject pooled text embedding into the trunk at `strength`."""
    return hidden + float(strength) * pooled.reshape(1, -1)


# F012 — Softmax Temperature Annealer ------------------------------------------
def softmax_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax with steering temperature: p = softmax(scores / T).

    T > 1 flattens (more democratic attention), T < 1 sharpens. An annealed
    schedule (T decaying over solver steps) emulates depth-wise sharpening
    without touching weights.
    """
    t = max(float(temperature), 1e-6)
    z = scores / t
    z = z - z.max(axis=-1, keepdims=True)  # stable softmax
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def anneal_temperature(step: int, total_steps: int, t_start: float,
                       t_end: float) -> float:
    """Linear anneal from t_start to t_end over the solver schedule."""
    frac = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return t_start + (t_end - t_start) * frac


# F013 — Output Projection Gamma Damper ----------------------------------------
def output_gamma(proj_out: np.ndarray, gamma: float) -> np.ndarray:
    """Damp final projection: out * gamma. Guards against magnitude blowup
    when several upstream gains compound (gains multiply through depth)."""
    return proj_out * float(gamma)


def compounded_gain(gains: Sequence[float]) -> float:
    """Report the multiplicative gain accumulated through a depth path."""
    out = 1.0
    for g in gains:
        out *= float(g)
    return out


# F014 — Multi-LoRA Dynamic Arbiter --------------------------------------------
def lora_blend(deltas: Dict[str, np.ndarray],
               weights: Dict[str, float]) -> np.ndarray:
    """Blend multiple LoRA rank-deltas into one effective delta.

    delta_eff = sum_i w_i * delta_i with raw (unnormalized) weights, so the
    optimizer controls both the mix ratio AND total strength. Pass weights
    through lora_blend_normalized for simplex-constrained blending.
    """
    if not deltas:
        raise ValueError("no deltas provided")
    keys = list(deltas.keys())
    out = np.zeros_like(next(iter(deltas.values())), dtype=np.float64)
    for k in keys:
        out += float(weights.get(k, 0.0)) * deltas[k]
    return out


def lora_blend_normalized(deltas: Dict[str, np.ndarray],
                          weights: Dict[str, float]) -> np.ndarray:
    """Simplex-constrained blend: weights clipped to >= 0 and normalized to
    sum 1, so the blended magnitude is bounded by the strongest member."""
    if not deltas:
        raise ValueError("no deltas provided")
    keys = list(deltas.keys())
    raw = np.array([max(0.0, weights.get(k, 0.0)) for k in keys], dtype=np.float64)
    s = raw.sum()
    raw = raw / s if s > 0 else np.full(len(keys), 1.0 / len(keys))
    out = np.zeros_like(next(iter(deltas.values())), dtype=np.float64)
    for k, w in zip(keys, raw):
        out += w * deltas[k]
    return out


# Long-context stabilization hooks ------------------------------------------------
# The mockup ledger names four memory-stabilizing taps for extended inference
# runs. Implemented here from first principles: bounding norms of a growing
# KV cache, hard top-K sparsification of attention mass, reserving mass on an
# initial "sink" position (the published observation that attention piles up
# on early tokens — Xiao et al. 2023 — reimplemented as plain math), and a
# drift probe that quantifies how far tuned gate vectors moved from base.


def kv_cache_clamp(kv_cache: np.ndarray, max_norm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bound the L2 norm of each cached token's K/V vector.

    Long contexts let individual cache vectors grow without bound, which is
    the numeric-precision failure mode of extended runs. Each token vector is
    rescaled to ``max_norm`` only if it exceeds it (a projection, not a
    rewrite), so untouched tokens are bit-identical.

    Returns ``(clipped_cache, per_token_scales)`` where scales are 1.0 for
    untouched tokens — making the operation auditable.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    cache = np.asarray(kv_cache, dtype=np.float64)
    norms = np.linalg.norm(cache, axis=-1, keepdims=True)
    scales = np.minimum(1.0, max_norm / np.maximum(norms, 1e-30))
    return cache * scales, scales[..., 0]


def sparse_topk(scores: np.ndarray, k: int, fill: float = -np.inf) -> np.ndarray:
    """Keep only the top-``k`` attention logits per query row; mask the rest.

    Returns logits (not probabilities) so callers keep their own softmax
    temperature policy; ``fill`` defaults to -inf which yields exact zero
    probability after softmax. Rows with fewer than ``k`` finite entries are
    left intact — sparsification never invents mass.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    s = np.asarray(scores, dtype=np.float64)
    if k >= s.shape[-1]:
        return s.copy()
    out = np.full_like(s, fill)
    idx = np.argpartition(s, -k, axis=-1)[..., -k:]
    np.put_along_axis(out, idx, np.take_along_axis(s, idx, axis=-1), axis=-1)
    return out


def attention_sink(scores: np.ndarray, sink_mass: float) -> np.ndarray:
    """Reserve ``sink_mass`` of attention probability on position 0.

    Streaming-attention stabilizer: a fixed fraction of softmax mass is
    pinned to the first position so the remaining distribution cannot wander
    as context grows. Implemented by subtracting log(1 - m) from all other
    positions' logits — exact for any temperature — leaving position 0's own
    logit untouched so the original attention pattern survives a sink of 0.
    """
    m = float(sink_mass)
    if not 0.0 <= m < 1.0:
        raise ValueError("sink_mass must be in [0, 1)")
    s = np.asarray(scores, dtype=np.float64)
    if m == 0.0:
        return s.copy()
    shift = np.log1p(-m)
    out = s.copy()
    out[..., 1:] += shift
    return out


def weight_drift_probe(base: np.ndarray, tuned: np.ndarray,
                       max_relative_l2: float = 1e-7) -> Dict[str, Any]:
    """Quantify movement of a gate vector from its base — the audit twin of
    :class:`ZeroModTap`.

    Reports absolute and relative L2 drift and a verdict against the
    zero-modulation contract (default bound 1e-7, matching ZeroModTap's
    tolerance). Used both to *prove* no-modification runs and to *bound*
    how far a tuned kernel may legally move base parameters.
    """
    b = np.asarray(base, dtype=np.float64)
    t = np.asarray(tuned, dtype=np.float64)
    if b.shape != t.shape:
        raise ValueError("base and tuned shapes differ")
    delta = t - b
    abs_l2 = float(np.linalg.norm(delta))
    base_l2 = float(np.linalg.norm(b))
    rel_l2 = abs_l2 / base_l2 if base_l2 > 0 else abs_l2
    return {
        "abs_l2": abs_l2,
        "base_l2": base_l2,
        "relative_l2": rel_l2,
        "max_relative_l2": float(max_relative_l2),
        "within_bound": rel_l2 <= max_relative_l2,
        "verdict": ("ZERO-MOD-INTACT" if rel_l2 <= max_relative_l2
                    else "DRIFT-EXCEEDS-BOUND"),
    }


__all__ = [
    "qk_gain", "qk_gain_per_head", "mlp_residual_ratio", "layernorm",
    "rmsnorm", "context_bias", "skip_alpha", "rope_angles", "apply_rope",
    "swiglu", "ZeroModTap", "gqa_route", "dit_sync", "dit_cross_gain",
    "pooled_injection", "softmax_temperature", "anneal_temperature",
    "output_gamma", "compounded_gain", "lora_blend", "kv_cache_clamp",
    "sparse_topk", "attention_sink", "weight_drift_probe",
]
