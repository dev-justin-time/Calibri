# SPDX-License-Identifier: MIT
# Calibrix accel — optional Rust acceleration for pure-math hot paths.
#
# The Rust core (rust-core/, MIT) reimplements two hot paths:
#   * kernel gain materialization (calibrix_gains)
#   * kernel spec parsing (calibrix_parse_spec)
# plus a DE benchmark entry (calibrix_bench_de) for measuring the
# interpreter-overhead delta. Python remains the canonical implementation;
# the Rust side is a drop-in accelerator loaded via ctypes when the built
# cdylib is present. No hard dependency: everything degrades to pure
# Python with identical results (parity is test-enforced).

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .kernel import KernelParams, ModulationSpec, parse_kernel_spec

_LIB_NAMES = {
    "win32": ("calibrix_core.dll", "libcalibrix_core.dll"),
    "darwin": ("libcalibrix_core.dylib",),
    "linux": ("libcalibrix_core.so",),
}


def _candidate_paths() -> List[Path]:
    here = Path(__file__).resolve().parent            # calibrix/calibrix
    root = here.parent                                # calibrix/
    names = _LIB_NAMES.get(sys.platform, ("libcalibrix_core.so",))
    out: List[Path] = []
    for base in (root / "rust-core" / "target" / "release",
                 here / "_native", root / "_native"):
        for n in names:
            out.append(base / n)
    return out


class _RustCore:
    def __init__(self) -> None:
        self.lib = None
        self.path: Optional[str] = None
        for cand in _candidate_paths():
            if cand.exists():
                try:
                    lib = ctypes.CDLL(str(cand))
                    lib.calibrix_gains.restype = ctypes.c_int
                    lib.calibrix_parse_spec.restype = ctypes.c_int
                    lib.calibrix_bench_de.restype = ctypes.c_int
                    self.lib = lib
                    self.path = str(cand)
                    return
                except OSError:
                    continue


_CORE = _RustCore()


def rust_available() -> bool:
    return _CORE.lib is not None


def rust_library_path() -> Optional[str]:
    return _CORE.path


def rust_gains(weight: float, position: float, focus: float, floor: float,
               ripple: float, phase: float, n_sites: int) -> List[float]:
    """Per-site gains via the Rust core. Raises RuntimeError if absent."""
    if _CORE.lib is None:
        raise RuntimeError("rust core not available")
    out = (ctypes.c_double * n_sites)()
    rc = _CORE.lib.calibrix_gains(
        ctypes.c_double(weight), ctypes.c_double(position), ctypes.c_double(focus),
        ctypes.c_double(floor), ctypes.c_double(ripple), ctypes.c_double(phase),
        ctypes.c_size_t(n_sites), out)
    if rc != 0:
        raise ValueError(f"rust calibrix_gains failed: rc={rc}")
    return list(out)


def rust_bench_de(dim: int, popsize: int, generations: int, seed: int,
                  lo: float = -2.0, hi: float = 2.0) -> float:
    """Best fitness of the Rust DE on the sphere (interpreter-delta benchmark)."""
    if _CORE.lib is None:
        raise RuntimeError("rust core not available")
    best = ctypes.c_double(0.0)
    rc = _CORE.lib.calibrix_bench_de(
        ctypes.c_size_t(dim), ctypes.c_size_t(popsize), ctypes.c_size_t(generations),
        ctypes.c_uint64(seed), ctypes.c_double(lo), ctypes.c_double(hi),
        ctypes.byref(best))
    if rc != 0:
        raise RuntimeError(f"rust calibrix_bench_de failed: rc={rc}")
    return best.value


def spec_gains(spec: str, n_sites: int) -> Dict[str, List[float]]:
    """Spec string -> per-channel gains, via Rust when available.

    {component: gains_list}. Identical values to the pure-Python path
    either way; which engine ran is reported by rust_available().
    """
    channels = parse_kernel_spec(spec)
    if rust_available():
        return {comp: rust_gains(p.weight, p.position, p.focus, p.floor,
                                 p.ripple, p.phase, n_sites)
                for comp, p in channels}
    return {comp: ModulationSpec(component=comp, n_sites=n_sites,
                                 kernel=p).gains()
            for comp, p in channels}


def parity_check(spec: str = "attn:1.2@0.5:0.9:0.3|mlp:0.8@0.25:1.0:4.0",
                 n_sites: int = 21) -> dict:
    """Cross-language parity report: Rust gains vs canonical Python gains.

    Returns {"rust": bool, "channels": int, "max_abs_diff": float,
    "parity": bool}. When the Rust core is absent the comparison compares
    Python against Python (trivially true) and reports rust=False.
    """
    report = {"rust": rust_available(), "channels": 0,
              "max_abs_diff": 0.0, "parity": True}
    for comp, p in parse_kernel_spec(spec):
        py = ModulationSpec(component=comp, n_sites=n_sites, kernel=p).gains()
        if rust_available():
            ru = rust_gains(p.weight, p.position, p.focus, p.floor,
                            p.ripple, p.phase, n_sites)
            diff = max((abs(a - b) for a, b in zip(py, ru)), default=0.0)
        else:
            diff = 0.0
        report["channels"] += 1
        report["max_abs_diff"] = max(report["max_abs_diff"], diff)
        if diff > 1e-9:
            report["parity"] = False
    return report
