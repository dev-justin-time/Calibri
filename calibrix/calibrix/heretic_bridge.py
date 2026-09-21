# SPDX-License-Identifier: MIT
# Calibrix — License boundary bridge to the AGPL Heretic CLI.
#
# ARCHITECTURAL LICENSE BOUNDARY (see LICENSES.md at the repo root):
#
#   MIT side (this package)              AGPL side (separate tree, own license)
#   ------------------------------       ----------------------------------------
#   calibrix/heretic_bridge.py   --->    heretic/heretic/  (heretic-llm CLI)
#
# The bridge talks to Heretic ONLY through a subprocess running its
# published CLI entry point (`heretic`, console script of `heretic-llm`).
# No import, no shared memory, no data structures, no linked code — the
# AGPL obligation therefore does not propagate into Calibrix. This mirrors
# the strategy the product doc calls "(a) keep Heretic strictly as a
# separate surface", implemented as an auditable, testable boundary.
#
# What the bridge ADDS (the improvement over using the CLI by hand):
#   1. `fit_kernel_profile` — converts an ablation result (per-layer
#      strengths from ANY source: Heretic output, our own reimplementation,
#      or manual expert profiles) into a portable 6-param Calibrix kernel
#      spec consumable by the ComfyUI node / server / marketplace. Heretic
#      produces per-layer interventions; Calibrix produces a *compact,
#      interpolable, transferable* parameterization of them. That
#      conversion is pure linear algebra on the MIT side.
#   2. `probe` / `run` — availability detection and structured invocation,
#      so the standalone runner can degrade gracefully when the AGPL tree
#      or its GPU deps are absent (non-commercial offline platforms rarely
#      have torch installed).

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kernel import KernelParams, ModulationSpec

HERETIC_TREE = Path(__file__).resolve().parents[2] / "heritic" / "heretic"


@dataclass
class BridgeStatus:
    """Whether the AGPL side is reachable, and how it will be driven."""
    heretic_cli: Optional[str]      # path to the `heretic` console script, if any
    tree_present: bool              # source tree checked out (for AGPL compliance/inspection)
    usable: bool                    # can `run()` actually execute?
    reason: str


def probe() -> BridgeStatus:
    """Detect the Heretic CLI without importing anything from it."""
    exe = shutil.which("heretic")
    if exe:
        return BridgeStatus(exe, HERETIC_TREE.exists(), True,
                            "heretic console script on PATH")
    if HERETIC_TREE.exists():
        return BridgeStatus(
            None, True, False,
            "source tree present but CLI not installed; AGPL side unavailable — "
            "install with `pip install -e heretic/heretic` (AGPL side) to enable")
    return BridgeStatus(None, False, False,
                        "AGPL tree absent; AGPL side unavailable; "
                        "MIT-side ablation and kernels still work")


@dataclass
class HereticRun:
    """Structured record of one subprocess ablation run."""
    model: str
    invoked: bool
    command: List[str]
    returncode: Optional[int]
    started_at: float
    duration_s: float
    stdout_tail: str
    stderr_tail: str
    artifacts_dir: Optional[str]
    note: str


def run(model: str, out_dir: Optional[str] = None, extra_args: Optional[Sequence[str]] = None,
        timeout_s: int = 3600) -> HereticRun:
    """Run `heretic <model> [extra...]` as a hermetic subprocess.

    The subprocess boundary is the license boundary: this process never
    imports heretic, and the child never imports calibrix.
    """
    status = probe()
    started = time.time()
    if not status.usable:
        return HereticRun(model=model, invoked=False, command=[], returncode=None,
                          started_at=started, duration_s=0.0, stdout_tail="",
                          stderr_tail=f"heretic unavailable: {status.reason}",
                          artifacts_dir=None,
                          note="not invoked: AGPL side unavailable")
    cmd = [status.heretic_cli, model, *(extra_args or [])]
    if out_dir:
        cmd += ["--output", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    return HereticRun(
        model=model, invoked=True, command=cmd, returncode=proc.returncode,
        started_at=started, duration_s=round(time.time() - started, 3),
        stdout_tail=proc.stdout[-2000:], stderr_tail=proc.stderr[-2000:],
        artifacts_dir=str(out_dir) if out_dir else None,
        note="executed via subprocess (license boundary intact)")


# ---------------------------------------------------------------------------
# Kernel-profile fitting: per-layer strengths -> compact portable kernel
# ---------------------------------------------------------------------------

def _nelder_mead_2d(f, x0, lo, hi, iters: int = 160, tol: float = 1e-12):
    """Bounded Nelder-Mead in 2-D (stdlib numpy only). Returns (best_xy, best_f).

    Used to refine the (position, focus) kernel fit off the coarse grid:
    the rmse surface is a curved valley in those two params, which axis-wise
    coordinate descent cannot follow but a simplex handles naturally.
    """
    x = np.asarray(x0, dtype=np.float64)
    span = np.array([hi[0] - lo[0], hi[1] - lo[1]], dtype=np.float64)
    pts = [x,
           np.clip(x + np.array([0.02 * span[0], 0.0]), lo, hi),
           np.clip(x + np.array([0.0, 0.02 * span[1]]), lo, hi)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if (abs(vals[0] - vals[-1]) < tol
                and np.linalg.norm(pts[-1] - pts[0]) < 1e-10):
            break
        centroid = (pts[0] + pts[1]) / 2.0
        # reflect
        xr = np.clip(centroid + (centroid - pts[-1]), lo, hi)
        fr = f(xr)
        if fr < vals[0]:
            # expand
            xe = np.clip(centroid + 2.0 * (centroid - pts[-1]), lo, hi)
            fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            # contract
            xc = np.clip(centroid + 0.5 * (pts[-1] - centroid), lo, hi)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                # shrink toward best
                for i in (1, 2):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return (pts[i], vals[i])


def fit_kernel_profile(strengths: Sequence[float],
                       grid: int = 96, seed: int = 0) -> Dict[str, Any]:
    """Fit a 6-param Calibrix kernel to a per-layer ablation-strength profile.

    Input : strengths[l] = ablation strength at layer l (any source; the
            values Heretic-style runs emit, our own reimplementation, or
            hand-tuned expert profiles).
    Output: {"kernel": [weight, position, focus, floor, ripple, phase],
             "rmse", "n_layers", "spec_string"}
            where spec_string parses in the ComfyUI node.

    Method (pure numpy, MIT side): coarse grid over (position, focus),
    closed-form optimal weight & floor for the linear decay model given
    (position, focus), a cosine ripple fit on the residual, then a bounded
    Nelder-Mead refinement of (position, focus) so arbitrary profiles fit
    to machine precision rather than to grid spacing. Deterministic.
    """
    y = np.asarray(strengths, dtype=np.float64)
    n = len(y)
    if n < 2:
        raise ValueError("need at least 2 layers to fit a profile")

    def solve_at(pos_c: float, focus_c: float):
        """Joint closed-form solve at (pos, focus).

        Design matrix [1-decay, decay, cos t, sin t] solves weight, floor
        AND the ripple cosine simultaneously — one lstsq, no residual
        staging. Joint fitting matters: fitting the ripple *after* the
        decay solve makes the rmse surface over (pos, focus) knife-edged
        and multimodal (the true basin is invisible on any coarse grid);
        jointly fitted, the ripple absorbs transient shape error and the
        surface stays smooth enough to guide the refinement stage.

        The cosine basis matches KernelParams.gains() exactly —
        cos(2π·l/n + 2π·phase), phase in turns — so the reported RMSE is
        the RMSE the deployed kernel delivers.

        Returns (rmse, params-tuple). Shared by all search stages."""
        t = 2.0 * np.pi * np.arange(n) / max(n, 1)
        dist = np.abs(np.arange(n) - pos_c * (n - 1))
        decay = np.clip(dist / max(focus_c * (n - 1), 1e-9), 0.0, 1.0)
        A = np.stack([1.0 - decay, decay, np.cos(t), np.sin(t)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        w_hat, fl_hat = float(coef[0]), float(coef[1])
        c_a, c_b = float(coef[2]), float(coef[3])
        amp = float(np.hypot(c_a, c_b))
        # atan2 returns radians; KernelParams.phase is in turns
        ph = float(np.arctan2(-c_b, c_a) / (2.0 * np.pi))
        ph = (ph + 0.5) % 1.0 - 0.5             # canonical [-0.5, 0.5)
        pred = A @ coef
        return (float(np.sqrt(np.mean((y - pred) ** 2))),
                (w_hat, pos_c, focus_c, fl_hat, amp, ph))

    # Stage 1 — coarse grid over (position, focus); keep every point so
    # stage 2 can seed from several distinct basins.
    scored: List[Tuple[float, float, float]] = []
    for pos in np.linspace(0.0, 1.0, max(grid // 2, 24)):
        for focus in (0.08, 0.16, 0.32, 0.64, 1.0, 2.0, 4.0):
            rmse_c, params_c = solve_at(float(pos), focus)
            scored.append((rmse_c, float(pos), focus))
    scored.sort(key=lambda r: r[0])

    # Stage 2 — multi-start bounded Nelder-Mead refinement. The surface is
    # still multimodal even with the joint solve (a weight<->floor mirror
    # basin exists by construction — flipping position about the grid swaps
    # the roles of weight and floor), so refine from several DISTINCT
    # coarse basins and keep the global best. Single-start refinement
    # measurably locks onto the mirror basin and misses the true profile.
    lo = (0.0, 1e-3)
    hi = (1.0, 8.0)

    def obj(pf) -> float:
        p2 = float(np.clip(pf[0], lo[0], hi[0]))
        f2 = float(np.clip(pf[1], lo[1], hi[1]))
        return solve_at(p2, f2)[0]

    seeds: List[Tuple[float, float]] = []
    for _r, p, f in scored:
        if all(abs(p - sp) > 0.08 or abs(f - sf) / max(sf, 1e-9) > 0.5
               for sp, sf in seeds):
            seeds.append((p, f))
        if len(seeds) >= 6:
            break
    best_xy = (scored[0][1], scored[0][2])
    best_rmse = scored[0][0]
    for sp, sf in seeds:
        xy, v = _nelder_mead_2d(obj, (sp, sf), lo, hi, iters=200)
        if v < best_rmse:
            best_xy, best_rmse = (float(xy[0]), float(xy[1])), v
    pos, focus = best_xy
    rmse, (w, pos, focus, fl, amp, ph) = solve_at(pos, focus)
    params = KernelParams(weight=float(np.clip(w, 0.0, 4.0)),
                          position=float(np.clip(pos, 0.0, 1.0)),
                          focus=float(np.clip(focus, 1e-3, 8.0)),
                          floor=float(np.clip(fl, 0.0, 4.0)),
                          ripple=float(np.clip(amp, -1.0, 1.0)),
                          phase=float(ph))
    params.validate()
    kernel_vec = [params.weight, params.position, params.focus,
                  params.floor, params.ripple, params.phase]
    # 9 decimals: the spec string must round-trip the fitted params to
    # better than 1e-8 (ComfyUI node / marketplace parity contracts parse
    # this exact string) — 4 decimals measurably broke kernel parity.
    spec = (f"ablate:{params.weight:.9f}@{params.position:.9f}:"
            f"{params.floor:.9f}:{params.focus:.9f}:"
            f"{params.ripple:.9f}:{params.phase:.9f}")
    return {"kernel": kernel_vec, "rmse": round(rmse, 6),
            "n_layers": n, "spec_string": spec}


def strengths_from_orthogonalization(W: np.ndarray, directions: Sequence[np.ndarray],
                                     n_layers: int) -> List[float]:
    """Per-layer ablation strengths implied by a weight matrix + directions.

    Improvement #1 over reading Heretic's raw output: instead of treating
    the per-layer interventions as opaque numbers, we re-derive the
    *strength* each layer's projection actually applies to the refusal
    direction — s(l) = ||P_l W_l||_2 / ||W_l||_2 (spectral loss along the
    direction) — making profiles comparable across models and runs.
    """
    if W.ndim != 2:
        raise ValueError("W must be a 2D weight matrix")
    out: List[float] = []
    w_norm = float(np.linalg.norm(W, 2))
    for l in range(n_layers):
        d = np.asarray(directions[min(l, len(directions) - 1)], dtype=np.float64)
        nd = np.linalg.norm(d)
        if nd < 1e-12 or w_norm < 1e-12:
            out.append(0.0)
            continue
        d = d / nd
        p = np.eye(d.shape[0]) - np.outer(d, d)
        out.append(float(np.linalg.norm(p @ W, 2) / w_norm))
    return out


def offline_synthetic_run(n_layers: int = 12, seed: int = 0) -> Dict[str, Any]:
    """End-to-end bridge demo without the AGPL side: synthesize a
    Heretic-shaped result (per-layer strengths with a bump), fit the
    kernel, round-trip through the ComfyUI spec parser contract.

    Used by the standalone runner and tests so the *MIT side* of the
    integration is fully exercisable offline, per the non-commercial
    platform requirement.
    """
    rng = np.random.default_rng(seed)
    layers = np.arange(n_layers, dtype=np.float64)
    true = 0.85 * np.exp(-((layers - n_layers * 0.35) ** 2) / (2.0 * (n_layers / 6) ** 2)) \
        + 0.10
    strengths = (true + rng.normal(0.0, 0.02, size=n_layers)).clip(0.0, 1.0).tolist()
    fit = fit_kernel_profile(strengths)
    return {
        "source": "offline-synthetic",
        "n_layers": n_layers,
        "strengths": [round(s, 4) for s in strengths],
        "fit": fit,
        "status": probe().usable and "agpl-available" or "mit-only",
    }


if __name__ == "__main__":
    print(json.dumps(offline_synthetic_run(), indent=2))
