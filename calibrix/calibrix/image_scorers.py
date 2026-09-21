# Calibrix image scorers.
#
# Reuses the Calibri reward surface (HPSv3, Q-Align, PickScore, ImageReward,
# CLIP) but adds offline, dependency-free versions so image-model calibration
# runs on any machine. The heavy Calibri reward functions
# (src/metrics/rewards.py) plug in via `make_reward_scorer`.

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, List, Optional, Sequence

from .scorers import Prompt, Scorer, ScorerContext, Score


# ---------------------------------------------------------------------------
# Image payload plumbing
# ---------------------------------------------------------------------------

def images_from_ctx(ctx: ScorerContext, prompts: Sequence[Prompt]) -> List[Any]:
    """Fetch generated images for prompts through the cached context.

    Adapters that produce images expose `generate_images(prompts) -> list`.
    The ScorerContext caches by prompt tuple like responses.
    """
    key = ("__images__",) + tuple(p.as_tuple() for p in prompts)
    cache = getattr(ctx, "_response_cache", None)
    if cache is not None and key in cache:
        return cache[key]
    adapter = ctx._adapter
    if not hasattr(adapter, "generate_images"):
        raise NotImplementedError(
            f"{type(adapter).__name__} does not produce images "
            "(needs generate_images)."
        )
    imgs = adapter.generate_images(list(prompts), batch_size=ctx.batch_size(), ctx=ctx)
    if cache is not None:
        cache[key] = imgs
    return imgs


def _image_stats(img: Any) -> List[float]:
    """Cheap deterministic statistics of a PIL image / ndarray.

    Works without numpy: converts via PIL getdata on a downsampled grid.
    Returns [mean_lum, std_lum, mean_sat, edge_density] in [0, 1].
    """
    # PIL image path
    if hasattr(img, "resize") and hasattr(img, "convert"):
        small = img.convert("RGB").resize((16, 16))
        px = list(small.getdata())
    elif hasattr(img, "shape") and hasattr(img, "tolist"):
        # ndarray (H, W, 3) 0..255
        arr = img.tolist() if hasattr(img, "tolist") else img
        h = len(arr)
        w = len(arr[0]) if h else 0
        px = []
        for y in range(0, h, max(h // 16, 1)):
            for x in range(0, w, max(w // 16, 1)):
                r, g, b = arr[y][x][:3]
                px.append((r, g, b))
    else:
        raise TypeError(f"unsupported image type: {type(img).__name__}")

    lums, sats = [], []
    for r, g, b in px:
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        mx, mn = max(r, g, b), min(r, g, b)
        sats.append(0.0 if mx == 0 else (mx - mn) / mx)
        lums.append(lum / 255.0)

    mean_lum = statistics.fmean(lums)
    std_lum = statistics.pstdev(lums) if len(lums) > 1 else 0.0
    mean_sat = statistics.fmean(sats)

    # crude edge density: horizontal luminance differences on the 16x16 grid
    n = len(lums)
    side = int(math.isqrt(n)) or 1
    edges = 0
    for y in range(side):
        for x in range(1, side):
            i = y * side + x
            if i < n and (i - 1) < n:
                if abs(lums[i] - lums[i - 1]) > 0.12:
                    edges += 1
    edge_density = edges / max(n, 1)

    return [mean_lum, std_lum, mean_sat, edge_density]


def _prompt_complexity(prompt: Prompt) -> float:
    """Deterministic 0..1 target complexity from prompt text length/detail."""
    words = prompt.user.split()
    return min(len(words) / 24.0, 1.0)


def _quality_from_stats(stats: List[float], target_complexity: float) -> float:
    """Map image statistics to a 0..1 quality proxy.

    Penalizes near-flat images (low contrast), over/under-exposure, and
    mismatch between texture density and prompt complexity. Deterministic
    and cheap; designed to respond to modulation-induced degradation the
    same way real aesthetic rewards do (sharply, once output collapses).
    """
    mean_lum, std_lum, mean_sat, edge_density = stats
    exposure = 1.0 - abs(mean_lum - 0.5) * 2.0          # 1 at mid-gray
    contrast = min(std_lum / 0.18, 1.0)                  # saturates at 0.18
    saturation = 1.0 - abs(mean_sat - 0.35) * 1.8        # pleasant zone ~0.35
    texture_fit = 1.0 - abs(edge_density - (0.05 + 0.35 * target_complexity)) * 2.5
    quality = (
        0.30 * max(exposure, 0.0)
        + 0.30 * max(contrast, 0.0)
        + 0.15 * max(saturation, 0.0)
        + 0.25 * max(texture_fit, 0.0)
    )
    return max(0.0, min(quality, 1.0))


# ---------------------------------------------------------------------------
# Offline image quality scorer (no GPU, no downloads)
# ---------------------------------------------------------------------------

class OfflineImageQuality(Scorer):
    """Image quality proxy from exposure/contrast/saturation/texture.

    Gap patched: Calibri's rewards (HPSv3 etc.) need GPUs and model
    downloads; this lets image calibration start free and offline, with the
    heavy rewards swapped in later via make_reward_scorer.
    """

    optimization = "maximize"

    def __init__(self, prompts: Sequence[Prompt]) -> None:
        self.prompts_spec = list(prompts)

    def init(self, ctx: ScorerContext) -> None:
        self.prompts = list(self.prompts_spec)

    def get_score(self, ctx: ScorerContext) -> Score:
        images = images_from_ctx(ctx, self.prompts)
        qualities = [
            _quality_from_stats(_image_stats(img), _prompt_complexity(p))
            for img, p in zip(images, self.prompts)
        ]
        mean = statistics.fmean(qualities) if qualities else 0.0
        return Score(value=mean, display=f"imgq {mean:.3f}")


# ---------------------------------------------------------------------------
# Prompt-image alignment, offline
# ---------------------------------------------------------------------------

_COLOR_ANCHORS = {
    "red": (200, 30, 30), "green": (30, 180, 60), "blue": (30, 60, 200),
    "yellow": (220, 200, 40), "purple": (120, 40, 180), "orange": (230, 120, 30),
    "black": (15, 15, 15), "white": (240, 240, 240), "pink": (235, 120, 160),
    "brown": (120, 80, 40), "gray": (128, 128, 128), "grey": (128, 128, 128),
}


class OfflineColorAlignment(Scorer):
    """Checks that colors named in the prompt appear in the image.

    Gap patched: color/attribute binding is the classic DiT failure mode
    that CLIP scores partially measure; this is a direct, free, interpretable
    version for the offline tier.
    """

    optimization = "maximize"

    def __init__(self, prompts: Sequence[Prompt]) -> None:
        self.prompts_spec = list(prompts)

    def init(self, ctx: ScorerContext) -> None:
        self.prompts = list(self.prompts_spec)

    @staticmethod
    def _requested_colors(prompt: Prompt) -> List[str]:
        low = prompt.user.lower()
        return [c for c in _COLOR_ANCHORS if c in low]

    def _coverage(self, img: Any, colors: List[str]) -> float:
        if not colors:
            return 1.0  # nothing requested -> nothing to fail
        if hasattr(img, "resize") and hasattr(img, "convert"):
            px = list(img.convert("RGB").resize((24, 24)).getdata())
        else:
            return 0.5  # unknown format: neutral
        hits = 0
        for name in colors:
            tr, tg, tb = _COLOR_ANCHORS[name]
            if any(
                (r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2 < 90 ** 2
                for r, g, b in px
            ):
                hits += 1
        return hits / len(colors)

    def get_score(self, ctx: ScorerContext) -> Score:
        images = images_from_ctx(ctx, self.prompts)
        covs = [self._coverage(img, self._requested_colors(p))
                for img, p in zip(images, self.prompts)]
        mean = statistics.fmean(covs) if covs else 0.0
        return Score(value=mean, display=f"color {mean:.3f}")


# ---------------------------------------------------------------------------
# Bridge to Calibri's heavy rewards (optional deps)
# ---------------------------------------------------------------------------

def make_reward_scorer(
    reward_names: List[str],
    prompts: Sequence[Prompt],
    device: str = "cuda",
    weights: Optional[dict] = None,
) -> Scorer:
    """Wrap Calibri's multi_score reward functions as a Calibrix Scorer.

    Example:
        make_reward_scorer(["hpsv3_remote", "pickscore"], prompts)

    Requires the Calibri environment (torch, reward deps, possibly the
    HPSv3/Q-Align servers). Raises ImportError with guidance when missing.
    """
    try:
        import sys as _sys
        from pathlib import Path as _Path

        # make src.* importable regardless of cwd
        root = _Path(__file__).resolve().parents[3]
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))

        from src.metrics.rewards import multi_score  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Calibri rewards unavailable. Run inside the Calibri environment "
            "(uv sync at repo root) or use OfflineImageQuality instead."
        ) from e

    weights = weights or {name: 1.0 for name in reward_names}

    class CalibriReward(Scorer):
        optimization = "maximize"
        reproducible = False  # remote servers / heavy models

        def init(self, ctx: ScorerContext) -> None:
            self.prompts = list(self.prompts_spec)
            self._fn = multi_score(device, weights)

        def get_score(self, ctx: ScorerContext) -> Score:
            images = images_from_ctx(ctx, self.prompts)
            details, _ = self._fn(images, [p.user for p in self.prompts], {})
            avg = details.get("avg", [])
            mean = statistics.fmean([float(s) for s in avg]) if len(avg) else 0.0
            return Score(value=mean, display=f"reward {mean:.3f}")

    scorer = CalibriReward(prompts)
    scorer.prompts_spec = list(prompts)
    scorer._reward_names = reward_names
    return scorer
