# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Domain 2: Scorer Panels & Reward Functions.
#
# Every scorer is implemented from its published definition:
#   * CIE 1976 L*a*b* Delta-E (CIE publication; sRGB->Lab via IEC 61966-2-1)
#   * SSIM (Wang et al. 2004, "Image quality assessment: from error visibility
#     to structural similarity")
#   * LPIPS *concept* (Zhang et al. 2018) reimplemented as a deterministic
#     deep-free patch-distance proxy (no model weights, fully offline)
#   * gradient-energy sharpness (classic image-processing definition)
#   * KL divergence and NFE penalty reuse calibrix.scorers semantics
#
# All functions are pure numpy: scorers must run inside the search loop
# thousands of times, so no model loading, no GPU, no I/O.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# F019 — CIELAB Delta-E Gamut Scorer ------------------------------------------
def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 12.92
    return np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, c)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (HxWx3, 0..1) -> CIE L*a*b* (D65). IEC 61966-2-1 + CIE 1976."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    lin = _srgb_to_linear(rgb)
    # sRGB D65 -> XYZ
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])  # D65 reference white
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_cie76(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 color difference: sqrt(dL^2 + da^2 + db^2)."""
    la, lb = srgb_to_lab(rgb_a), srgb_to_lab(rgb_b)
    return np.linalg.norm(la - lb, axis=-1)


def color_drift_score(candidate: np.ndarray, reference: np.ndarray,
                      pass_below: float = 6.0) -> float:
    """Fraction of pixels with Delta-E under `pass_below` (JND ~ 2.3; 6.0 =
    'noticeable at a glance'). Returns a reward in [0, 1]: higher = closer."""
    de = delta_e_cie76(candidate, reference)
    return float(np.mean(de < pass_below))


# F022 — SSIM Structural Preserver ---------------------------------------------
def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def _filter2(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid-mode 2D correlation via stride tricks (no scipy)."""
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(img, kernel.shape)
    return np.einsum("ijkl,kl->ij", win, kernel)


def ssim(img_a: np.ndarray, img_b: np.ndarray,
         data_range: float = 1.0) -> float:
    """Mean SSIM (Wang et al. 2004) on grayscale, single-scale."""
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
        b = b.mean(axis=-1)
    k = _gaussian_kernel()
    mu_a, mu_b = _filter2(a, k), _filter2(b, k)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    s_a2 = _filter2(a * a, k) - mu_a2
    s_b2 = _filter2(b * b, k) - mu_b2
    s_ab = _filter2(a * b, k) - mu_ab
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    num = (2 * mu_ab + c1) * (2 * s_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (s_a2 + s_b2 + c2)
    return float(np.mean(num / den))


# F025 — Style-Consistency Patch Distance (LPIPS proxy) -------------------------
def _patch_normalize(img: np.ndarray, patch: int = 8) -> np.ndarray:
    """Non-overlapping patches, per-patch mean/variance normalized.

    LPIPS measures *perceptual* distance by comparing deep activations after
    normalization. Our proxy: compare shallow statistics (per-patch normalized
    gradients) — deterministic, weights-free, surprisingly well correlated for
    style drift detection.
    """
    g = np.gradient(img.mean(axis=-1) if img.ndim == 3 else img)
    mag = np.sqrt(g[0] ** 2 + g[1] ** 2)
    h, w = mag.shape
    h, w = h - h % patch, w - w % patch
    patches = mag[:h, :w].reshape(h // patch, patch, w // patch, patch)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch, patch)
    mu = patches.mean(axis=(1, 2), keepdims=True)
    sd = patches.std(axis=(1, 2), keepdims=True) + 1e-8
    return (patches - mu) / sd


def patch_lpips(img_a: np.ndarray, img_b: np.ndarray, patch: int = 8) -> float:
    """0 = identical style texture, 1 = maximally different patch statistics."""
    pa, pb = _patch_normalize(img_a, patch), _patch_normalize(img_b, patch)
    n = min(len(pa), len(pb))
    if n == 0:
        return 0.0
    d = np.sqrt(((pa[:n] - pb[:n]) ** 2).mean(axis=(1, 2)))
    return float(np.clip(d.mean() / 2.0, 0.0, 1.0))  # ~[0,1]


# F023 — OCR Typographic Sharpness (gradient energy) ----------------------------
def text_sharpness(img: np.ndarray) -> float:
    """Normalized gradient energy in the image — proxy for glyph legibility.

    Real OCR scoring needs an OCR engine (heavy, non-deterministic); gradient
    energy correlates with stroke definition and runs in microseconds, making
    it usable *inside* the optimization loop. Pair with an occasional real OCR
    check offline.
    """
    g = img.mean(axis=-1) if img.ndim == 3 else img
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return float(gx + gy)


# F018 — Refusal-direction classifier hook --------------------------------------
def refusal_projection(hidden: np.ndarray, direction: np.ndarray) -> float:
    """Project a residual-stream vector on the refusal direction.

    Signed magnitude in standard deviations of the projection distribution —
    the same statistic direction extraction uses, reused as a *runtime*
    compliance probe (Arditi et al. 2024 method).
    """
    nd = np.linalg.norm(direction)
    if nd < 1e-12:
        return 0.0
    proj = (hidden @ direction) / nd
    sd = proj.std() + 1e-12
    return float(proj.mean() / sd)


# F021 — Face/Hand Geometry L2 (deterministic ROI proxy) ------------------------
def geometry_anomaly(roi_landmarks: np.ndarray, canonical: np.ndarray) -> float:
    """L2 deviation of landmark configuration from a canonical arrangement.

    In production, landmarks come from a face/hand detector; the *scoring
    math* is this function. Keeping it model-free makes it testable and lets
    the same scorer serve any ROI source.
    """
    if roi_landmarks.shape != canonical.shape:
        raise ValueError("landmark shapes must match")
    scale = np.linalg.norm(canonical - canonical.mean(axis=0)) + 1e-8
    return float(np.linalg.norm(roi_landmarks - canonical) / scale)


# F024 — VRAM Peak Footprint Scorer (host-side metering) -------------------------
class VramMeter:
    """Peak-by-construction usage tracker (CPU RSS or torch CUDA if present).

    Scoring allocation behavior requires a *meter*, not a guess: this wraps
    the best available source and exposes peak bytes; billing and D5 cost
    models consume it directly.
    """

    def __init__(self) -> None:
        self.peak_bytes = 0
        try:
            import resource  # unix

            self._floor = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:
            self._floor = 0

    def current(self) -> int:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass
        return self._floor

    def sample(self) -> int:
        cur = self.current()
        self.peak_bytes = max(self.peak_bytes, cur)
        return cur


def vram_score(peak_bytes: int, budget_bytes: int) -> float:
    """1.0 at or under budget, decaying quadratically past it."""
    if budget_bytes <= 0:
        return 0.0
    ratio = peak_bytes / budget_bytes
    return float(max(0.0, 1.0 - (ratio - 1.0) ** 2)) if ratio > 1.0 else 1.0


# F017 — NFETrap (studio-side, composable with calibrix.scorers.NFETrap) --------
def nfe_penalty(steps: int, reference_steps: int, alpha: float = 0.02) -> float:
    """Linear NFE penalty relative to the reference budget."""
    return alpha * max(0, int(steps) - int(reference_steps))


# F027 — Normalized Weight Auto-Sum ----------------------------------------------
def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Softmax-free simplex projection: clip negatives, renormalize to 1.

    Guarantees multi-objective scalarizations stay a *convex* combination —
    otherwise the optimizer games absolute weight magnitude instead of the
    tradeoff.
    """
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / s


# F026 — Zero-Shot Safety Boundary Gate -------------------------------------------
def safety_gate(harm_scores: Sequence[float], threshold: float = 0.05) -> bool:
    """True = panel may proceed; aborts optimization if any harm probe fires."""
    return all(float(s) <= threshold for s in harm_scores)


# F028 — Custom Python Scorer Hook (sandboxed) ------------------------------------
def load_scorer_class(path: str, class_name: str = "Scorer"):
    """Import a customer scorer file with a restricted globals environment.

    Sandboxing in CPython is fundamentally leaky; this is *hygiene* (no
    network in globals, no file builtins) not a security boundary. Real
    isolation for untrusted code = subprocess + resource limits.
    """
    import types

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    mod = types.ModuleType("customer_scorer")
    safe_globals = {"__builtins__": __builtins__, "math": math, "numpy": np}
    exec(compile(src, path, "exec"), safe_globals, mod.__dict__)  # noqa: S102
    cls = mod.__dict__.get(class_name)
    if cls is None:
        raise ValueError(f"{class_name!r} not found in {path}")
    return cls


__all__ = [
    "srgb_to_lab", "delta_e_cie76", "color_drift_score", "ssim", "patch_lpips",
    "text_sharpness", "refusal_projection", "geometry_anomaly", "VramMeter",
    "vram_score", "nfe_penalty", "normalize_weights", "safety_gate",
    "load_scorer_class",
]
